{-# LANGUAGE CApiFFI #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
module VinaryTree.Liblevenshtein
  ( Algorithm(..)
  , QueryOrder(..)
  , Term(..)
  , Match(..)
  , Transducer
  , QueryCache
  , QueryCacheStats(..)
  , Cursor
  , PhoneticPattern
  , PhoneticRules
  , transducer
  , closeTransducer
  , queryCache
  , closeQueryCache
  , queryCacheStats
  , clearQueryCache
  , resetQueryCacheStats
  , cachedQueryText
  , cachedQueryBytes
  , cachedQueryU64
  , queryText
  , queryBytes
  , queryU64
  , queryPattern
  , closeCursor
  , next
  , nextBatch
  , foldBatches
  , regexPattern
  , llrePattern
  , patternMatches
  , patternSize
  , closePattern
  , phoneticRules
  , rulesLength
  , applyRules
  , closeRules
  , distance
  , distanceThreshold
  , damerauDistance
  , damerauDistanceThreshold
  , trueDamerauDistance
  , trueDamerauDistanceThreshold
  ) where

import Prelude hiding (maximum)
import Control.Exception (onException, throwIO)
import Data.ByteString (ByteString)
import qualified Data.ByteString as BS
import qualified Data.ByteString.Unsafe as BSU
import Data.IORef (IORef, atomicModifyIORef', newIORef)
import Data.Text (Text)
import qualified Data.Text.Encoding as Text
import Data.Word (Word8, Word32, Word64)
import Foreign.C.String (peekCString)
import Foreign.C.Types (CChar, CInt(..), CSize(..))
import Foreign.ForeignPtr
  (ForeignPtr, finalizeForeignPtr, newForeignPtr, withForeignPtr)
import Foreign.Marshal.Alloc (allocaBytes)
import Foreign.Marshal.Array (peekArray, withArray)
import Foreign.Marshal.Utils (fillBytes)
import Foreign.Ptr (FunPtr, Ptr, castPtr, plusPtr)
import Foreign.Storable (peekByteOff)
import VinaryTree.Interop
  (DictionaryResource, VtResource, withDictionaryResource)

#include "liblevenshtein.h"

data CTransducer
data CQueryCache
data CQueryCacheStats
data CCursor
data CPattern
data CRules
data CMatch
data CBatch
data COwnedString

data Algorithm = Standard | Transposition | MergeAndSplit | DamerauLevenshtein
  deriving stock (Eq, Ord, Show, Enum, Bounded)
data QueryOrder = Traversal | DistanceThenTerm
  deriving stock (Eq, Ord, Show, Enum, Bounded)
data Term = BytesTerm !ByteString | TextTerm !Text | TokensTerm ![Word64]
  deriving stock (Eq, Show)
data Match = Match { term :: !Term, editDistance :: !Int, identifier :: !(Maybe Word64) }
  deriving stock (Eq, Show)
newtype Transducer = Transducer (ForeignPtr CTransducer)
newtype QueryCache = QueryCache (ForeignPtr CQueryCache)
data QueryCacheStats = QueryCacheStats
  { requests :: !Word64
  , hits :: !Word64
  , misses :: !Word64
  , admissions :: !Word64
  , rejections :: !Word64
  , evictions :: !Word64
  , residentEntries :: !Int
  , residentWeight :: !Int
  } deriving stock (Eq, Show)
data Cursor = Cursor !(ForeignPtr CCursor) !(IORef [Match])
newtype PhoneticPattern = PhoneticPattern (ForeignPtr CPattern)
newtype PhoneticRules = PhoneticRules (ForeignPtr CRules)
type Status = CInt

foreign import ccall unsafe "llev_last_error_message" cLastError :: IO (Ptr CChar)
foreign import ccall unsafe "&llev_transducer_free"
  cTransducerFree :: FunPtr (Ptr CTransducer -> IO ())
foreign import ccall unsafe "&llev_query_cursor_free"
  cCursorFree :: FunPtr (Ptr CCursor -> IO ())
foreign import ccall unsafe "&llev_query_cache_free"
  cQueryCacheFree :: FunPtr (Ptr CQueryCache -> IO ())
foreign import ccall unsafe "&llev_phonetic_pattern_free"
  cPatternFree :: FunPtr (Ptr CPattern -> IO ())
foreign import ccall unsafe "&llev_phonetic_rules_free"
  cRulesFree :: FunPtr (Ptr CRules -> IO ())
foreign import ccall unsafe "llev_transducer_new"
  cTransducerNew :: Ptr VtResource -> Word32 -> Ptr (Ptr CTransducer) -> IO Status
foreign import ccall unsafe "llev_transducer_query_utf8"
  cQueryText :: Ptr CTransducer -> Ptr CChar -> CSize -> CSize -> Word32
             -> Ptr (Ptr CCursor) -> IO Status
foreign import ccall unsafe "llev_transducer_query_bytes"
  cQueryBytes :: Ptr CTransducer -> Ptr Word8 -> CSize -> CSize -> Word32
              -> Ptr (Ptr CCursor) -> IO Status
foreign import ccall unsafe "llev_transducer_query_u64"
  cQueryU64 :: Ptr CTransducer -> Ptr Word64 -> CSize -> CSize -> Word32
            -> Ptr (Ptr CCursor) -> IO Status
foreign import ccall unsafe "llev_query_cache_new"
  cQueryCacheNew :: Ptr CTransducer -> CSize -> CSize
                 -> Ptr (Ptr CQueryCache) -> IO Status
foreign import ccall unsafe "llev_query_cache_clear"
  cQueryCacheClear :: Ptr CQueryCache -> IO Status
foreign import ccall unsafe "llev_query_cache_reset_stats"
  cQueryCacheResetStats :: Ptr CQueryCache -> IO Status
foreign import ccall unsafe "llev_query_cache_stats"
  cQueryCacheStats :: Ptr CQueryCache -> Ptr CQueryCacheStats -> IO Status
foreign import ccall unsafe "llev_query_cache_query_utf8"
  cCachedQueryText :: Ptr CQueryCache -> Ptr CChar -> CSize -> CSize -> Word32
                   -> Ptr (Ptr CCursor) -> IO Status
foreign import ccall unsafe "llev_query_cache_query_bytes"
  cCachedQueryBytes :: Ptr CQueryCache -> Ptr Word8 -> CSize -> CSize -> Word32
                    -> Ptr (Ptr CCursor) -> IO Status
foreign import ccall unsafe "llev_query_cache_query_u64"
  cCachedQueryU64 :: Ptr CQueryCache -> Ptr Word64 -> CSize -> CSize -> Word32
                  -> Ptr (Ptr CCursor) -> IO Status
foreign import ccall unsafe "llev_transducer_query_pattern"
  cQueryPattern :: Ptr CTransducer -> Ptr CPattern -> Word8
                -> Ptr (Ptr CCursor) -> IO Status
foreign import ccall safe "llev_query_cursor_next_batch"
  cNextBatch :: Ptr CCursor -> CSize -> Ptr CBatch -> IO Status
foreign import ccall unsafe "llev_query_cursor_release_batch"
  cReleaseBatch :: Ptr CCursor -> Word64 -> IO Status
foreign import ccall unsafe "llev_phonetic_pattern_compile_regex"
  cRegex :: Ptr CChar -> CSize -> Ptr (Ptr CPattern) -> IO Status
foreign import ccall unsafe "llev_phonetic_pattern_compile_llre"
  cLlre :: Ptr CChar -> CSize -> Ptr (Ptr CPattern) -> IO Status
foreign import ccall unsafe "llev_phonetic_pattern_matches"
  cPatternMatches :: Ptr CPattern -> Ptr CChar -> CSize -> Ptr Word8 -> IO Status
foreign import ccall unsafe "llev_phonetic_pattern_size"
  cPatternSize :: Ptr CPattern -> Ptr CSize -> Ptr CSize -> IO Status
foreign import ccall unsafe "llev_phonetic_rules_builtin"
  cRulesBuiltin :: Word32 -> Ptr (Ptr CRules) -> IO Status
foreign import ccall unsafe "llev_phonetic_rules_parse"
  cRulesParse :: Ptr CChar -> CSize -> Ptr (Ptr CRules) -> IO Status
foreign import ccall unsafe "llev_phonetic_rules_len"
  cRulesLength :: Ptr CRules -> Ptr CSize -> IO Status
foreign import ccall unsafe "llev_phonetic_rules_apply"
  cRulesApply :: Ptr CRules -> Ptr CChar -> CSize -> Ptr COwnedString -> IO Status
foreign import ccall unsafe "llev_owned_string_free"
  cOwnedStringFree :: Ptr COwnedString -> IO ()
foreign import ccall safe "llev_distance"
  cDistance :: Ptr CChar -> CSize -> Ptr CChar -> CSize -> IO CSize
foreign import ccall safe "llev_distance_threshold"
  cDistanceThreshold :: Ptr CChar -> CSize -> Ptr CChar -> CSize -> CSize -> IO CSize
foreign import ccall safe "llev_damerau_distance"
  cDamerau :: Ptr CChar -> CSize -> Ptr CChar -> CSize -> IO CSize
foreign import ccall safe "llev_damerau_distance_threshold"
  cDamerauThreshold :: Ptr CChar -> CSize -> Ptr CChar -> CSize -> CSize -> IO CSize
foreign import ccall safe "llev_true_damerau_distance"
  cTrueDamerau :: Ptr CChar -> CSize -> Ptr CChar -> CSize -> IO CSize
foreign import ccall safe "llev_true_damerau_distance_threshold"
  cTrueDamerauThreshold :: Ptr CChar -> CSize -> Ptr CChar -> CSize -> CSize -> IO CSize

check :: Status -> IO ()
check 0 = pure ()
check _ = cLastError >>= peekCString >>= throwIO . userError

withTransducer :: Transducer -> (Ptr CTransducer -> IO a) -> IO a
withTransducer (Transducer value) = withForeignPtr value
withCursor :: Cursor -> (Ptr CCursor -> IO a) -> IO a
withCursor (Cursor value _) = withForeignPtr value
withQueryCache :: QueryCache -> (Ptr CQueryCache -> IO a) -> IO a
withQueryCache (QueryCache value) = withForeignPtr value
withPattern :: PhoneticPattern -> (Ptr CPattern -> IO a) -> IO a
withPattern (PhoneticPattern value) = withForeignPtr value
withRules :: PhoneticRules -> (Ptr CRules -> IO a) -> IO a
withRules (PhoneticRules value) = withForeignPtr value

transducer :: Algorithm -> DictionaryResource -> IO Transducer
transducer algorithm resource = withDictionaryResource resource $ \resourcePointer ->
  allocaBytes #{size void *} $ \output -> do
    cTransducerNew resourcePointer (fromIntegral (fromEnum algorithm)) output >>= check
    Transducer <$> (peekByteOff output 0 >>= newForeignPtr cTransducerFree)

closeTransducer :: Transducer -> IO ()
closeTransducer (Transducer value) = finalizeForeignPtr value

queryCache :: Int -> Int -> Transducer -> IO QueryCache
queryCache maximumEntries maximumWeight value
  | maximumEntries < 0 || maximumWeight < 0 =
      throwIO (userError "query-cache limits must be non-negative")
  | otherwise = withTransducer value $ \native ->
      allocaBytes #{size void *} $ \output -> do
        cQueryCacheNew native (fromIntegral maximumEntries)
          (fromIntegral maximumWeight) output >>= check
        QueryCache <$> (peekByteOff output 0 >>= newForeignPtr cQueryCacheFree)

closeQueryCache :: QueryCache -> IO ()
closeQueryCache (QueryCache value) = finalizeForeignPtr value

queryCacheStats :: QueryCache -> IO QueryCacheStats
queryCacheStats value = withQueryCache value $ \native ->
  allocaBytes #{size LlevQueryCacheStats} $ \output -> do
    cQueryCacheStats native output >>= check
    QueryCacheStats
      <$> #{peek LlevQueryCacheStats, requests} output
      <*> #{peek LlevQueryCacheStats, hits} output
      <*> #{peek LlevQueryCacheStats, misses} output
      <*> #{peek LlevQueryCacheStats, admissions} output
      <*> #{peek LlevQueryCacheStats, rejections} output
      <*> #{peek LlevQueryCacheStats, evictions} output
      <*> (fromIntegral <$> (#{peek LlevQueryCacheStats, resident_entries} output :: IO CSize))
      <*> (fromIntegral <$> (#{peek LlevQueryCacheStats, resident_weight} output :: IO CSize))

clearQueryCache :: QueryCache -> IO ()
clearQueryCache value = withQueryCache value $ \native -> cQueryCacheClear native >>= check

resetQueryCacheStats :: QueryCache -> IO ()
resetQueryCacheStats value =
  withQueryCache value $ \native -> cQueryCacheResetStats native >>= check

newCursor :: Ptr CCursor -> IO Cursor
newCursor pointer = Cursor <$> newForeignPtr cCursorFree pointer <*> newIORef []

queryBytesWith :: (Ptr CTransducer -> Ptr Word8 -> CSize -> CSize -> Word32
                 -> Ptr (Ptr CCursor) -> IO Status)
               -> QueryOrder -> Transducer -> ByteString -> Int -> IO Cursor
queryBytesWith action order value input maximum
  | maximum < 0 = throwIO (userError "maximum distance must be non-negative")
  | otherwise = withTransducer value $ \native -> BSU.unsafeUseAsCStringLen input $ \(p, n) ->
      allocaBytes #{size void *} $ \output -> do
        action native (castPtr p) (fromIntegral n) (fromIntegral maximum)
          (fromIntegral (fromEnum order)) output >>= check
        peekByteOff output 0 >>= newCursor

queryText :: QueryOrder -> Transducer -> Text -> Int -> IO Cursor
queryText order value input maximum =
  queryBytesWith (\t p n m o out -> cQueryText t (castPtr p) n m o out)
    order value (Text.encodeUtf8 input) maximum
queryBytes :: QueryOrder -> Transducer -> ByteString -> Int -> IO Cursor
queryBytes = queryBytesWith cQueryBytes
queryU64 :: QueryOrder -> Transducer -> [Word64] -> Int -> IO Cursor
queryU64 order value input maximum
  | maximum < 0 = throwIO (userError "maximum distance must be non-negative")
  | otherwise = withTransducer value $ \native -> withArray input $ \tokens ->
      allocaBytes #{size void *} $ \output -> do
        cQueryU64 native tokens (fromIntegral (length input)) (fromIntegral maximum)
          (fromIntegral (fromEnum order)) output >>= check
        peekByteOff output 0 >>= newCursor

cachedQueryBytesWith :: (Ptr CQueryCache -> Ptr Word8 -> CSize -> CSize -> Word32
                       -> Ptr (Ptr CCursor) -> IO Status)
                     -> QueryOrder -> QueryCache -> ByteString -> Int -> IO Cursor
cachedQueryBytesWith action order value input maximum
  | maximum < 0 = throwIO (userError "maximum distance must be non-negative")
  | otherwise = withQueryCache value $ \native ->
      BSU.unsafeUseAsCStringLen input $ \(p, n) ->
        allocaBytes #{size void *} $ \output -> do
          action native (castPtr p) (fromIntegral n) (fromIntegral maximum)
            (fromIntegral (fromEnum order)) output >>= check
          peekByteOff output 0 >>= newCursor

cachedQueryText :: QueryOrder -> QueryCache -> Text -> Int -> IO Cursor
cachedQueryText order value input maximum =
  cachedQueryBytesWith
    (\cache p n m o out -> cCachedQueryText cache (castPtr p) n m o out)
    order value (Text.encodeUtf8 input) maximum

cachedQueryBytes :: QueryOrder -> QueryCache -> ByteString -> Int -> IO Cursor
cachedQueryBytes = cachedQueryBytesWith cCachedQueryBytes

cachedQueryU64 :: QueryOrder -> QueryCache -> [Word64] -> Int -> IO Cursor
cachedQueryU64 order value input maximum
  | maximum < 0 = throwIO (userError "maximum distance must be non-negative")
  | otherwise = withQueryCache value $ \native -> withArray input $ \tokens ->
      allocaBytes #{size void *} $ \output -> do
        cCachedQueryU64 native tokens (fromIntegral (length input))
          (fromIntegral maximum) (fromIntegral (fromEnum order)) output >>= check
        peekByteOff output 0 >>= newCursor

queryPattern :: Transducer -> PhoneticPattern -> Int -> IO Cursor
queryPattern value pattern maximum
  | maximum < 0 || maximum > 255 = throwIO (userError "pattern distance is outside u8")
  | otherwise = withTransducer value $ \native -> withPattern pattern $ \p ->
      allocaBytes #{size void *} $ \output -> do
        cQueryPattern native p (fromIntegral maximum) output >>= check
        peekByteOff output 0 >>= newCursor

closeCursor :: Cursor -> IO ()
closeCursor (Cursor value _) = finalizeForeignPtr value

peekMatch :: Ptr CMatch -> IO Match
peekMatch pointer = do
  domain <- #{peek LlevMatch, unit_domain} pointer :: IO Word32
  dataPointer <- #{peek LlevMatch, term_data} pointer :: IO (Ptr Word8)
  termLength <- #{peek LlevMatch, term_len} pointer :: IO CSize
  byteLength <- #{peek LlevMatch, byte_len} pointer :: IO CSize
  distance' <- #{peek LlevMatch, distance} pointer :: IO CSize
  identifier' <- #{peek LlevMatch, id} pointer :: IO Word64
  hasIdentifier <- #{peek LlevMatch, has_id} pointer :: IO Word8
  term' <- case domain of
    1 -> BytesTerm <$> BS.packCStringLen (castPtr dataPointer, fromIntegral byteLength)
    2 -> TextTerm . Text.decodeUtf8 <$> BS.packCStringLen
           (castPtr dataPointer, fromIntegral byteLength)
    _ -> TokensTerm <$> peekArray (fromIntegral termLength) (castPtr dataPointer)
  pure (Match term' (fromIntegral distance')
        (if hasIdentifier == 0 then Nothing else Just identifier'))

nextBatch :: Int -> Cursor -> IO (Maybe [Match])
nextBatch maximum cursor
  | maximum <= 0 = throwIO (userError "batch size must be positive")
  | otherwise = withCursor cursor $ \native -> allocaBytes #{size LlevMatchBatchView} $ \view -> do
      fillBytes view 0 #{size LlevMatchBatchView}
      status <- cNextBatch native (fromIntegral maximum) view
      if status == 1 then pure Nothing else do
        check status
        matches <- #{peek LlevMatchBatchView, matches} view :: IO (Ptr CMatch)
        count <- #{peek LlevMatchBatchView, len} view :: IO CSize
        generation <- #{peek LlevMatchBatchView, generation} view :: IO Word64
        let release = cReleaseBatch native generation >>= check
        values <- mapM (\index -> peekMatch
          (matches `plusPtr` (index * #{size LlevMatch}))) [0 .. fromIntegral count - 1]
          `onException` release
        release
        pure (Just values)

next :: Cursor -> IO (Maybe Match)
next cursor@(Cursor _ buffered) = do
  available <- atomicModifyIORef' buffered $ \values -> case values of
    [] -> ([], Nothing)
    item : rest -> (rest, Just item)
  case available of
    Just item -> pure (Just item)
    Nothing -> nextBatch 1 cursor >>= \case
      Nothing -> pure Nothing
      Just [] -> pure Nothing
      Just (item : rest) -> atomicModifyIORef' buffered (const (rest, ())) >> pure (Just item)

foldBatches :: Int -> Cursor -> state -> (state -> [Match] -> IO state) -> IO state
foldBatches maximum cursor = go
  where
    go state reducer = nextBatch maximum cursor >>= \case
      Nothing -> pure state
      Just values -> reducer state values >>= \nextState -> go nextState reducer

withUtf8 :: Text -> (Ptr CChar -> CSize -> IO a) -> IO a
withUtf8 input action = BSU.unsafeUseAsCStringLen (Text.encodeUtf8 input) $ \(p, n) ->
  action p (fromIntegral n)

compilePattern :: (Ptr CChar -> CSize -> Ptr (Ptr CPattern) -> IO Status)
               -> Text -> IO PhoneticPattern
compilePattern action source = withUtf8 source $ \p n ->
  allocaBytes #{size void *} $ \output -> do
    action p n output >>= check
    PhoneticPattern <$> (peekByteOff output 0 >>= newForeignPtr cPatternFree)
regexPattern, llrePattern :: Text -> IO PhoneticPattern
regexPattern = compilePattern cRegex
llrePattern = compilePattern cLlre
patternMatches :: PhoneticPattern -> Text -> IO Bool
patternMatches pattern input = withPattern pattern $ \native -> withUtf8 input $ \p n ->
  allocaBytes 1 $ \output -> cPatternMatches native p n output >>= check
    >> ((/= 0) <$> (peekByteOff output 0 :: IO Word8))
patternSize :: PhoneticPattern -> IO (Int, Int)
patternSize pattern = withPattern pattern $ \native ->
  allocaBytes #{size size_t} $ \states -> allocaBytes #{size size_t} $ \transitions -> do
    cPatternSize native states transitions >>= check
    (,) <$> (fromIntegral <$> (peekByteOff states 0 :: IO CSize))
        <*> (fromIntegral <$> (peekByteOff transitions 0 :: IO CSize))
closePattern :: PhoneticPattern -> IO ()
closePattern (PhoneticPattern value) = finalizeForeignPtr value

phoneticRules :: Text -> IO PhoneticRules
phoneticRules source = allocaBytes #{size void *} $ \output -> do
  status <- if source == "english-orthography" then cRulesBuiltin 0 output
            else if source == "english-phonetic" then cRulesBuiltin 1 output
            else withUtf8 source $ \p n -> cRulesParse p n output
  check status
  PhoneticRules <$> (peekByteOff output 0 >>= newForeignPtr cRulesFree)
rulesLength :: PhoneticRules -> IO Int
rulesLength rules = withRules rules $ \native -> allocaBytes #{size size_t} $ \output ->
  cRulesLength native output >>= check >> (fromIntegral <$> (peekByteOff output 0 :: IO CSize))
applyRules :: PhoneticRules -> Text -> IO Text
applyRules rules input = withRules rules $ \native -> withUtf8 input $ \p n ->
  allocaBytes #{size LlevOwnedString} $ \output -> do
    fillBytes output 0 #{size LlevOwnedString}
    cRulesApply native p n output >>= check
    dataPointer <- #{peek LlevOwnedString, data} output :: IO (Ptr CChar)
    length' <- #{peek LlevOwnedString, len} output :: IO CSize
    result <- Text.decodeUtf8 <$> BS.packCStringLen (dataPointer, fromIntegral length')
    cOwnedStringFree output
    pure result
closeRules :: PhoneticRules -> IO ()
closeRules (PhoneticRules value) = finalizeForeignPtr value

binaryDistance :: (Ptr CChar -> CSize -> Ptr CChar -> CSize -> IO CSize)
               -> Text -> Text -> IO Int
binaryDistance action source target = withUtf8 source $ \sp sn -> withUtf8 target $ \tp tn ->
  fromIntegral <$> action sp sn tp tn
thresholdDistance :: (Ptr CChar -> CSize -> Ptr CChar -> CSize -> CSize -> IO CSize)
                  -> Text -> Text -> Int -> IO Int
thresholdDistance action source target threshold
  | threshold < 0 = throwIO (userError "threshold must be non-negative")
  | otherwise = withUtf8 source $ \sp sn -> withUtf8 target $ \tp tn ->
      fromIntegral <$> action sp sn tp tn (fromIntegral threshold)
distance, damerauDistance, trueDamerauDistance :: Text -> Text -> IO Int
distance = binaryDistance cDistance
damerauDistance = binaryDistance cDamerau
trueDamerauDistance = binaryDistance cTrueDamerau
distanceThreshold, damerauDistanceThreshold, trueDamerauDistanceThreshold
  :: Text -> Text -> Int -> IO Int
distanceThreshold = thresholdDistance cDistanceThreshold
damerauDistanceThreshold = thresholdDistance cDamerauThreshold
trueDamerauDistanceThreshold = thresholdDistance cTrueDamerauThreshold
