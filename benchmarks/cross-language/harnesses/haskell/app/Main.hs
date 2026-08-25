{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}

-- | Haskell harness for the cross-language benchmark program.
--
-- Implements harnesses/common/PROTOCOL.md over the hsc2hs facades
-- (liblevenshtein + libdictenstein). The runner
-- stages RELEASE cdylibs plus pkg-config metadata under
-- .stage/haskell-native and provides LD_LIBRARY_PATH at run time.
--
-- Fairness notes (PROTOCOL.md §10): default RTS, single capability (built
-- without -threaded); System.Clock 'Monotonic' is the pinned §9 source.
module Main (main) where

import Control.Monad (forM_, unless, when)
import Data.Array (Array, bounds, listArray, (!))
import Data.Array.IO (IOUArray, getElems, newArray, writeArray)
import Data.Bits (shiftR, xor, (.&.))
import Data.ByteString (ByteString)
import qualified Data.ByteString as BS
import Data.IORef (IORef, newIORef, readIORef, writeIORef)
import Data.Int (Int64)
import Data.List (intercalate)
import Data.Text (Text)
import qualified Data.Text.Encoding as TextEncoding
import qualified Data.Text.Foreign as TextForeign
import Data.Time.Clock (getCurrentTime)
import Data.Time.Format (defaultTimeLocale, formatTime)
import Data.Version (showVersion)
import Data.Word (Word64, Word8)
import Numeric (showHex)
import System.Clock (Clock (Monotonic), getTime, toNanoSecs)
import System.Directory (createDirectoryIfMissing)
import System.Environment (getArgs)
import System.Exit (ExitCode (ExitFailure), exitWith)
import System.FilePath (takeBaseName, takeDirectory)
import System.Info (fullCompilerVersion)
import System.IO (hPutStrLn, stderr)
import qualified VinaryTree.Libdictenstein as Dict
import qualified VinaryTree.Liblevenshtein as Lev

wallCapSeconds :: Double
wallCapSeconds = 300.0

sampleDefinition :: String
sampleDefinition =
  "one full pass over the query set; every cursor fully drained and "
    <> "(term, distance) materialized"

baseNotes :: [String]
baseNotes =
  [ "hsc2hs facade; default RTS, single capability (built without -threaded)" ]

die :: String -> IO a
die message = do
  hPutStrLn stderr ("bench-cross-haskell: " <> message)
  exitWith (ExitFailure 2)

-- ---------------------------------------------------------------------
-- Checksum primitives (PROTOCOL.md §8) — Word64 wraps by definition
-- ---------------------------------------------------------------------

fnvOffset, fnvPrime :: Word64
fnvOffset = 0xcbf29ce484222325
fnvPrime = 0x100000001b3

fnvUpdate :: Word64 -> Word8 -> Word64
fnvUpdate !hash !byte = (hash `xor` fromIntegral byte) * fnvPrime
{-# INLINE fnvUpdate #-}

fnv1a64 :: ByteString -> Word64
fnv1a64 = BS.foldl' fnvUpdate fnvOffset

-- | entry(term, distance) over utf8(term) ‖ 0x00 ‖ LE64(distance).
entryHash :: Text -> Int -> Word64
entryHash term distance =
  let !afterTerm = BS.foldl' fnvUpdate fnvOffset (TextEncoding.encodeUtf8 term)
      !afterSeparator = fnvUpdate afterTerm 0x00
      !distance64 = fromIntegral distance :: Word64
      step !hash i = fnvUpdate hash (fromIntegral ((distance64 `shiftR` (8 * i)) .&. 0xff))
  in foldl' step afterSeparator [0 .. 7 :: Int]

-- | 16 lowercase hex digits, zero padded; showHex on Word64 is unsigned.
checksumHex :: Word64 -> String
checksumHex value =
  let digits = showHex value ""
  in replicate (16 - length digits) '0' <> digits

selfTest :: IO ()
selfTest = do
  let expect actual wanted label =
        unless (actual == wanted) . die $
          "checksum self-test failed for " <> label <> ": got "
            <> checksumHex actual <> ", want " <> checksumHex wanted
  expect (fnv1a64 "") 0xcbf29ce484222325 "fnv1a64(\"\")"
  expect (fnv1a64 "a") 0xaf63dc4c8601ec8c "fnv1a64(\"a\")"
  expect (entryHash "cat" 1) 0x9697fa3e50464bc4 "entry(cat,1)"
  expect (entryHash "cat" 0) 0xb592c1475b3595e5 "entry(cat,0)"
  expect (entryHash "cot" 1) 0xb8acc5d3816bcdea "entry(cot,1)"
  expect (entryHash "cat" 0 + entryHash "cot" 1) 0x6e3f871adca163cf "checksum{2}"
  expect 0 0 "checksum{}"
  unless (checksumHex maxBound == "ffffffffffffffff") $
    die "hex serialization of high-bit checksums is not unsigned"

-- ---------------------------------------------------------------------
-- CLI contract (PROTOCOL.md §1)
-- ---------------------------------------------------------------------

data Args = Args
  { argMode :: !String
  , argAlgorithm :: !(Maybe String)
  , argMaxDistance :: !Int
  , argDictionary :: !(Maybe FilePath)
  , argQueries :: !(Maybe FilePath)
  , argBackend :: !String
  , argOut :: !(Maybe FilePath)
  , argSamples :: !Int
  , argWarmupSeconds :: !Double
  , argGateLimit :: !Int
  , argReps :: !Int
  , argCells :: !(Maybe FilePath)
  }

defaultArgs :: Args
defaultArgs = Args
  { argMode = ""
  , argAlgorithm = Nothing
  , argMaxDistance = -1
  , argDictionary = Nothing
  , argQueries = Nothing
  , argBackend = ""
  , argOut = Nothing
  , argSamples = 30
  , argWarmupSeconds = 3.0
  , argGateLimit = 200
  , argReps = 10
  , argCells = Nothing
  }

parseIntArg :: String -> String -> IO Int
parseIntArg flag value = case reads value of
  [(parsed, "")] -> pure parsed
  _ -> die (flag <> " expects an integer, got " <> show value)

parseDoubleArg :: String -> String -> IO Double
parseDoubleArg flag value = case reads value of
  [(parsed, "")] -> pure parsed
  _ -> die (flag <> " expects a number, got " <> show value)

parseArgs :: [String] -> IO Args
parseArgs = go defaultArgs
  where
    go args [] = finish args
    go _ [dangling] = die ("dangling argument: " <> dangling)
    go args (flag : value : rest) = case flag of
      "--mode" -> go args { argMode = value } rest
      "--algorithm" -> go args { argAlgorithm = Just value } rest
      "--max-distance" -> do
        parsed <- parseIntArg flag value
        go args { argMaxDistance = parsed } rest
      "--dictionary" -> go args { argDictionary = Just value } rest
      "--queries" -> go args { argQueries = Just value } rest
      "--backend" -> go args { argBackend = value } rest
      "--out" -> go args { argOut = Just value } rest
      "--samples" -> do
        parsed <- parseIntArg flag value
        go args { argSamples = parsed } rest
      "--warmup-seconds" -> do
        parsed <- parseDoubleArg flag value
        go args { argWarmupSeconds = parsed } rest
      "--gate-limit" -> do
        parsed <- parseIntArg flag value
        go args { argGateLimit = parsed } rest
      "--reps" -> do
        parsed <- parseIntArg flag value
        go args { argReps = parsed } rest
      "--cells" -> go args { argCells = Just value } rest
      other -> die ("unknown flag: " <> other)
    finish args
      | null (argMode args) || argDictionary args == Nothing
          || null (argBackend args) =
          die "--mode, --dictionary, --backend are required"
      | otherwise = pure args

-- ---------------------------------------------------------------------
-- Input loading (PROTOCOL.md §3)
-- ---------------------------------------------------------------------

readLinesBytes :: FilePath -> IO [ByteString]
readLinesBytes path = do
  raw <- BS.readFile path
  let entries = filter (not . BS.null) (BS.split 10 raw)
  when (null entries) (die (path <> " contains no lines"))
  pure entries

-- Ord ByteString is byte-lexicographic: exactly the §3 invariant.
assertStrictlySorted :: [ByteString] -> FilePath -> IO ()
assertStrictlySorted entries path = go (1 :: Int) entries
  where
    go _ [] = pure ()
    go _ [_] = pure ()
    go !lineNumber (previous : current : rest) = do
      unless (previous < current) . die $
        path <> " is not strictly byte-sorted at line " <> show lineNumber
          <> ": " <> show previous <> " >= " <> show current
      go (lineNumber + 1) (current : rest)

-- ---------------------------------------------------------------------
-- Dictionary, transducer, and the pass (PROTOCOL.md §4–5)
-- ---------------------------------------------------------------------

data Side = Side
  { sidePrepared :: ![(ByteString, Maybe Word64)] -- entry prep happens once
  , sideDictionary :: !(IORef (Maybe Dict.Dictionary))
  , sideTransducer :: !(IORef (Maybe Lev.Transducer))
  }

buildDictionary :: Side -> String -> IO ()
buildDictionary side backend = case backend of
  "dynamic_dawg" -> do
    dictionary <- Dict.dynamicDawg Dict.UnicodeScalar
    inserted <- Dict.putManyBytes dictionary (sidePrepared side)
    let expected = length (sidePrepared side)
    unless (inserted == expected) . die $
      "batch insert count mismatch: " <> show inserted <> " != " <> show expected
    writeIORef (sideDictionary side) (Just dictionary)
  other ->
    die ("unsupported backend for the Haskell target (dynamic_dawg only): " <> other)

freeDictionary :: Side -> IO ()
freeDictionary side = do
  readIORef (sideTransducer side) >>= \case
    Just transducer -> do
      Lev.closeTransducer transducer
      writeIORef (sideTransducer side) Nothing
    Nothing -> pure ()
  readIORef (sideDictionary side) >>= \case
    Just dictionary -> do
      Dict.close dictionary
      writeIORef (sideDictionary side) Nothing
    Nothing -> pure ()

algorithmOfName :: String -> IO Lev.Algorithm
algorithmOfName = \case
  "standard" -> pure Lev.Standard
  "transposition" -> pure Lev.Transposition
  "merge_and_split" -> pure Lev.MergeAndSplit
  "damerau_levenshtein" -> pure Lev.DamerauLevenshtein
  other -> die ("unknown algorithm: " <> other)

createTransducer :: Side -> String -> IO ()
createTransducer side algorithmName = do
  readIORef (sideTransducer side) >>= \case
    Just previous -> Lev.closeTransducer previous
    Nothing -> pure ()
  readIORef (sideDictionary side) >>= \case
    Nothing -> die "dictionary must be built before the transducer"
    Just dictionary -> do
      algorithm <- algorithmOfName algorithmName
      shared <- Dict.resource dictionary
      transducer <- Lev.transducer algorithm shared
      writeIORef (sideTransducer side) (Just transducer)

-- Strict pass accumulator: the O(1) triple plus the gate checksum.
data Accumulator = Accumulator !Int !Int !Int !Word64

-- | One full pass (§5): every cursor drained through the facade's batched
-- reducer (256 matches per crossing — the declared batch_size), summing the
-- O(1) triple; the FNV checksum only in untimed gate contexts.
-- Data.Text.Foreign.lengthWord8 is the O(1) UTF-8 byte length (text >= 2.0
-- stores UTF-8 internally).
fullPass :: Side -> Array Int Text -> Int -> Bool -> IO Accumulator
fullPass side queries maxDistance withChecksum = do
  transducer <- readIORef (sideTransducer side) >>= \case
    Just transducer -> pure transducer
    Nothing -> die "createTransducer must run before fullPass"
  let (low, high) = bounds queries
      reducer (Accumulator !m !b !d !cs) matches' =
        let step (Accumulator !m' !b' !d' !cs') matched = case Lev.term matched of
              Lev.TextTerm text ->
                let !distance' = Lev.editDistance matched
                    !checksum' =
                      if withChecksum
                        then cs' + entryHash text distance'
                        else cs'
                in Accumulator (m' + 1) (b' + TextForeign.lengthWord8 text)
                     (d' + distance') checksum'
              _ -> error "unexpected non-text match for a text query"
        in pure (foldl' step (Accumulator m b d cs) matches')
      loop !index !accumulator
        | index > high = pure accumulator
        | otherwise = do
            cursor <- Lev.queryText Lev.Traversal transducer (queries ! index) maxDistance
            accumulator' <- Lev.foldBatches 256 cursor accumulator reducer
            Lev.closeCursor cursor
            loop (index + 1) accumulator'
  loop low (Accumulator 0 0 0 0)

tripleEquals :: Accumulator -> Accumulator -> Bool
tripleEquals (Accumulator m1 b1 d1 _) (Accumulator m2 b2 d2 _) =
  m1 == m2 && b1 == b2 && d1 == d2

-- ---------------------------------------------------------------------
-- Monotonic clock (PROTOCOL.md §9: System.Clock getTime Monotonic)
-- ---------------------------------------------------------------------

nowNs :: IO Int64
nowNs = fromIntegral . toNanoSecs <$> getTime Monotonic

-- ---------------------------------------------------------------------
-- Result JSON (PROTOCOL.md §11 — runner post-fills run_id, sha256s,
-- cell_snapshot, environment_ref)
-- ---------------------------------------------------------------------

escapeJson :: String -> String
escapeJson = concatMap escape
  where
    escape '"' = "\\\""
    escape '\\' = "\\\\"
    escape '\n' = "\\n"
    escape '\r' = "\\r"
    escape '\t' = "\\t"
    escape c
      | c < ' ' = let hex = showHex (fromEnum c) ""
                  in "\\u" <> replicate (4 - length hex) '0' <> hex
      | otherwise = [c]

data Emission
  = EmitMeasurements ![Int64] !Accumulator
  | EmitConstruct ![Int64]

writeResult
  :: FilePath -> Args -> String -> String -> Int -> FilePath -> Int -> Int
  -> Maybe Int64 -> Int -> Emission -> String -> [String] -> IO ()
writeResult outPath args mode algorithm maxDistance queriesPath queryCount
    termCount constructNs warmupPasses emission status notes = do
  timestamp <- formatTime defaultTimeLocale "%Y-%m-%dT%H:%M:%SZ" <$> getCurrentTime
  let queryset = takeBaseName queriesPath
      samplesRequested = case mode of
        "construct" -> argReps args
        "query" -> argSamples args
        _ -> 0 :: Int
      constructField = case constructNs of
        Just value -> ",\n    \"construct_ns\": " <> show value <> "\n"
        Nothing -> "\n"
      payload = case emission of
        EmitConstruct times ->
          "  \"construct\": {\n"
            <> "    \"reps\": " <> show (length times) <> ",\n"
            <> "    \"times_ns\": [" <> intercalate ", " (map show times) <> "],\n"
            <> "    \"term_count\": " <> show termCount <> "\n"
            <> "  },\n"
        EmitMeasurements samples (Accumulator matches termBytes distanceSum checksum) ->
          "  \"measurements\": {\n"
            <> "    \"samples_ns\": [" <> intercalate ", " (map show samples) <> "],\n"
            <> "    \"sample_count\": " <> show (length samples) <> ",\n"
            <> "    \"matches_per_pass\": " <> show matches <> ",\n"
            <> "    \"term_bytes_per_pass\": " <> show termBytes <> ",\n"
            <> "    \"distance_sum_per_pass\": " <> show distanceSum <> ",\n"
            <> "    \"checksum_hex\": \"" <> checksumHex checksum <> "\"\n"
            <> "  },\n"
      rendered =
        "{\n"
          <> "  \"schema_version\": \"1.0.0\",\n"
          <> "  \"suite\": \"cross-language-v1\",\n"
          <> "  \"timestamp_utc\": \"" <> timestamp <> "\",\n"
          <> "  \"target\": {\n"
          <> "    \"language\": \"haskell\",\n"
          <> "    \"implementation\": \"vinary-tree\",\n"
          <> "    \"backend\": \"hsc2hs\",\n"
          <> "    \"runtime_version\": \"GHC "
          <> escapeJson (showVersion fullCompilerVersion) <> "\",\n"
          <> "    \"library_version\": \"0.10.0\",\n"
          <> "    \"artifact\": { \"kind\": \"local-build\", \"id\": "
          <> "\"liblevenshtein@0.10.0 (cabal project)\" }\n"
          <> "  },\n"
          <> "  \"dictionary\": {\n"
          <> "    \"file\": \""
          <> escapeJson (maybe "" id (argDictionary args)) <> "\",\n"
          <> "    \"term_count\": " <> show termCount <> ",\n"
          <> "    \"structure\": \"dynamic_dawg\",\n"
          <> "    \"unit_domain\": \"unicode_scalar\"" <> constructField
          <> "  },\n"
          <> "  \"workload\": {\n"
          <> "    \"queryset\": \"" <> escapeJson queryset <> "\",\n"
          <> "    \"file\": \"" <> escapeJson queriesPath <> "\",\n"
          <> "    \"query_count\": " <> show queryCount <> "\n"
          <> "  },\n"
          <> "  \"algorithm\": \"" <> algorithm <> "\",\n"
          <> "  \"max_distance\": " <> show maxDistance <> ",\n"
          <> "  \"mode\": \""
          <> (if mode == "memory-child" then "memory" else mode) <> "\",\n"
          <> "  \"protocol\": {\n"
          <> "    \"timer\": \"monotonic\",\n"
          <> "    \"harness\": \"self-timed\",\n"
          <> "    \"warmup_seconds_min\": " <> show (argWarmupSeconds args) <> ",\n"
          <> "    \"warmup_passes\": " <> show warmupPasses <> ",\n"
          <> "    \"samples_requested\": " <> show samplesRequested <> ",\n"
          <> "    \"sample_definition\": \"" <> escapeJson sampleDefinition <> "\",\n"
          <> "    \"batch_size\": 256,\n"
          <> "    \"wall_cap_seconds\": " <> show (round wallCapSeconds :: Int) <> "\n"
          <> "  },\n"
          <> payload
          <> "  \"status\": \"" <> status <> "\",\n"
          <> "  \"notes\": ["
          <> intercalate ", " (map (\note -> "\"" <> escapeJson note <> "\"") notes)
          <> "]\n}\n"
  createDirectoryIfMissing True (takeDirectory outPath)
  writeFile outPath rendered

-- ---------------------------------------------------------------------
-- Modes (PROTOCOL.md §6) and the batch driver (§7)
-- ---------------------------------------------------------------------

runConstruct :: Args -> Side -> Int -> IO ()
runConstruct args side termCount = do
  outPath <- maybe (die "--out is required for construct mode") pure (argOut args)
  buildDictionary side (argBackend args) -- warmup build
  freeDictionary side
  let reps = max 1 (argReps args)
  times <- newArray (0, reps - 1) 0 :: IO (IOUArray Int Int64)
  forM_ [0 .. reps - 1] $ \rep -> do
    started <- nowNs
    buildDictionary side (argBackend args)
    finished <- nowNs
    writeArray times rep (finished - started)
    freeDictionary side
  recorded <- getElems times
  writeResult outPath args "construct" "standard" 1
    (maybe "workload/queries/hits.txt" id (argQueries args)) 1 termCount
    Nothing 1 (EmitConstruct recorded) "ok"
    (baseNotes
       <> [ "construct mode: timed region is the build from the pre-sorted \
            \in-memory list only" ])

runQueryCell
  :: Args -> Side -> Array Int Text -> Int -> String -> Int -> FilePath
  -> FilePath -> Int -> Int64 -> IO ()
runQueryCell args side queries queryCount algorithm maxDistance queriesPath
    outPath termCount constructNs = do
  gate <- fullPass side queries maxDistance True
  warmStart <- nowNs
  let warmupBudget = round (argWarmupSeconds args * 1e9) :: Int64
      warmupLoop !passes !lastNs = do
        elapsed <- subtract warmStart <$> nowNs
        if elapsed < warmupBudget || passes < (2 :: Int)
          then do
            started <- nowNs
            triple <- fullPass side queries maxDistance False
            finished <- nowNs
            unless (tripleEquals triple gate)
              (die "nondeterministic result during warmup")
            warmupLoop (passes + 1) (finished - started)
          else pure (passes, lastNs)
  (warmupPasses, lastPassNs) <- warmupLoop 0 0
  let lastPassSeconds = fromIntegral lastPassNs / 1e9 :: Double
      requested = argSamples args
      capped = fromIntegral requested * lastPassSeconds > wallCapSeconds
      sampleCount =
        if capped
          then max 10 (floor (wallCapSeconds / lastPassSeconds))
          else requested
      status = if capped then "degraded" else "ok"
      notes =
        baseNotes
          <> [ "samples reduced from " <> show requested <> " to "
                 <> show sampleCount <> " by the 300s wall cap (estimated pass "
                 <> show lastPassSeconds <> "s)"
             | capped ]
  samples <- newArray (0, sampleCount - 1) 0 :: IO (IOUArray Int Int64)
  forM_ [0 .. sampleCount - 1] $ \index -> do
    started <- nowNs
    triple <- fullPass side queries maxDistance False
    finished <- nowNs
    writeArray samples index (finished - started)
    unless (tripleEquals triple gate)
      (die "nondeterministic result during measurement")
  recorded <- getElems samples
  writeResult outPath args "query" algorithm maxDistance queriesPath queryCount
    termCount (Just constructNs) warmupPasses (EmitMeasurements recorded gate)
    status notes

runOne :: Args -> Side -> String -> Int -> FilePath -> FilePath -> Int -> Int64 -> IO ()
runOne args side algorithm maxDistance queriesPath outPath termCount constructNs = do
  createTransducer side algorithm
  queryLines <- readLinesBytes queriesPath
  let decoded = map TextEncoding.decodeUtf8 queryLines
      totalQueries = length decoded
  case argMode args of
    "verify" -> do
      let limit = min (argGateLimit args) totalQueries
          subset = listArray (0, limit - 1) (take limit decoded)
      gate <- fullPass side subset maxDistance True
      writeResult outPath args "verify" algorithm maxDistance queriesPath limit
        termCount (Just constructNs) 0 (EmitMeasurements [] gate) "ok" baseNotes
    "memory-child" -> do
      let queries = listArray (0, totalQueries - 1) decoded
      gate <- fullPass side queries maxDistance True
      writeResult outPath args "memory-child" algorithm maxDistance queriesPath
        totalQueries termCount (Just constructNs) 0 (EmitMeasurements [] gate)
        "ok" baseNotes
    "query" -> do
      let queries = listArray (0, totalQueries - 1) decoded
      runQueryCell args side queries totalQueries algorithm maxDistance
        queriesPath outPath termCount constructNs
    other -> die ("unknown mode: " <> other)

runCells :: Args -> Side -> FilePath -> Int -> Int64 -> IO ()
runCells args side cellsPath termCount constructNs = do
  raw <- readFile cellsPath
  let rows =
        [ row
        | row <- map trim (lines raw)
        , not (null row)
        , take 1 row /= "#"
        ]
      trim = dropWhile (== ' ') . reverse . dropWhile (\c -> c == ' ' || c == '\r') . reverse
  forM_ rows $ \row -> case splitOnTab row of
    [algorithm, distanceField, queriesPath, outPath] -> do
      distance <- parseIntArg "--cells max_distance" distanceField
      runOne args side algorithm distance queriesPath outPath termCount constructNs
    _ -> die ("cells row needs 4 fields: " <> row)
  where
    splitOnTab value = case break (== '\t') value of
      (field, "") -> [field]
      (field, _ : rest) -> field : splitOnTab rest

main :: IO ()
main = do
  selfTest
  args <- getArgs >>= parseArgs
  dictionaryPath <- maybe (die "--dictionary is required") pure (argDictionary args)
  terms <- readLinesBytes dictionaryPath
  assertStrictlySorted terms dictionaryPath
  dictionaryRef <- newIORef Nothing
  transducerRef <- newIORef Nothing
  let side = Side
        { sidePrepared = map (\term -> (term, Nothing)) terms
        , sideDictionary = dictionaryRef
        , sideTransducer = transducerRef
        }
      termCount = length terms
  case argMode args of
    "construct" -> runConstruct args side termCount
    mode | mode `elem` ["query", "verify", "memory-child"] -> do
      buildStart <- nowNs
      buildDictionary side (argBackend args)
      buildEnd <- nowNs
      let constructNs = buildEnd - buildStart
      case argCells args of
        Just cellsPath -> runCells args side cellsPath termCount constructNs
        Nothing -> do
          algorithm <- maybe (die "--algorithm is required") pure (argAlgorithm args)
          queriesPath <- maybe (die "--queries is required") pure (argQueries args)
          outPath <- maybe (die "--out is required") pure (argOut args)
          when (argMaxDistance args < 0) (die "--max-distance is required")
          runOne args side algorithm (argMaxDistance args) queriesPath outPath
            termCount constructNs
    other -> die ("unknown mode: " <> other)
