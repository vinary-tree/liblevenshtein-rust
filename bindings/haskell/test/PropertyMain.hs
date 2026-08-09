{-# LANGUAGE OverloadedStrings #-}

-- | C8 native property-based tests for the Haskell facade (QuickCheck).
--
-- Every property is checked against an in-language brute-force Levenshtein
-- oracle:
--
--   (a) distance symmetry @d(a,b)==d(b,a)@, identity @d(a,a)==0@, and threshold
--       consistency @dThreshold(a,b,k) == (if d<=k then d else -2)@ across the
--       Levenshtein\/Damerau\/true-Damerau variants, plus standard oracle
--       agreement. Distances are 'Int'; the native usize::MAX-1 over-bound
--       sentinel reads as @-2@.
--   (b) a query's result set at distance @k@ equals
--       @{t in dict : lev(query,t) <= k}@ over a random libdictenstein
--       DynamicDawg dictionary, with exact distances and value round-trips.
--   (c) u64 value round-trips, with 0 and 'maxBound' pinned.
--
-- The RNG is replayed from a fixed seed, so a failing run reproduces. Distance
-- operands may be empty: an empty 'Text.Text' can reach the native distance
-- entry point as a null pointer, and as of the LLEV-B18 fix that NULL+0 is
-- accepted as the empty string (the transducer query path always accepted it).
module Main (main) where

import Control.Monad (unless)
import Data.Function (on)
import qualified Data.List as List
import qualified Data.Text as Text
import Data.Text.Encoding (encodeUtf8)
import Data.Word (Word64)
import System.Exit (exitFailure)
import Test.QuickCheck
import Test.QuickCheck.Random (mkQCGen)
import qualified VinaryTree.Libdictenstein as Dict
import qualified VinaryTree.Liblevenshtein as Lev

alphabet :: String
alphabet = "abcé"

genTerm :: Gen Text.Text
genTerm = Text.pack <$> (choose (0, 6) >>= \n -> vectorOf n (elements alphabet))

genValue :: Gen (Maybe Word64)
genValue = frequency [(1, pure Nothing), (3, Just <$> arbitrary)]

genDictionary :: Gen [(Text.Text, Maybe Word64)]
genDictionary = do
  n <- choose (0, 8)
  pairs <- vectorOf n ((,) <$> genTerm <*> genValue)
  pure (List.nubBy ((==) `on` fst) pairs)

-- | Reference Levenshtein distance over Unicode scalars (each alphabet
-- character is a single scalar, so 'Text.unpack' matches the native semantics).
levenshtein :: Text.Text -> Text.Text -> Int
levenshtein a b = last (List.foldl' nextRow [0 .. length right] left)
  where
    left = Text.unpack a
    right = Text.unpack b
    nextRow previous leftChar =
      case previous of
        (headCell : _) -> scanl compute (headCell + 1) (zip3 right previous (drop 1 previous))
        [] -> []
      where
        compute west (rightChar, northwest, north) =
          minimum [north + 1, west + 1, northwest + (if leftChar == rightChar then 0 else 1)]

drainMatches :: Lev.Cursor -> IO [Lev.Match]
drainMatches cursor = go []
  where
    go acc = do
      value <- Lev.next cursor
      case value of
        Nothing -> Lev.closeCursor cursor >> pure (reverse acc)
        Just m -> go (m : acc)

textTerm :: Lev.Match -> Text.Text
textTerm m = case Lev.term m of
  Lev.TextTerm t -> t
  _ -> error "expected text term"

runQuery :: [(Text.Text, Maybe Word64)] -> Text.Text -> Int -> IO [(Text.Text, (Int, Maybe Word64))]
runQuery entries query k = do
  dictionary <- Dict.dynamicDawg Dict.UnicodeScalar
  mapM_ (\(term, value) -> Dict.putBytes dictionary (encodeUtf8 term) value) entries
  resource <- Dict.resource dictionary
  transducer <- Lev.transducer Lev.Standard resource
  cursor <- Lev.queryText Lev.Traversal transducer query k
  matches <- drainMatches cursor
  Lev.closeTransducer transducer
  Dict.close dictionary
  pure (map (\m -> (textTerm m, (Lev.editDistance m, Lev.identifier m))) matches)

distanceProperty
  :: (Text.Text -> Text.Text -> IO Int)
  -> (Text.Text -> Text.Text -> Int -> IO Int)
  -> Property
distanceProperty dist thr =
  forAll genTerm $ \a ->
    forAll genTerm $ \b ->
      forAll (choose (0, 3)) $ \k -> ioProperty $ do
        full <- dist a b
        backward <- dist b a
        identity <- dist a a
        bounded <- thr a b k
        pure (full == backward && identity == 0 && bounded == (if full <= k then full else -2))

oracleProperty :: Property
oracleProperty =
  forAll genTerm $ \a ->
    forAll genTerm $ \b -> ioProperty $ do
      d <- Lev.distance a b
      pure (d == levenshtein a b)

queryProperty :: Property
queryProperty =
  forAll genDictionary $ \entries ->
    forAll genTerm $ \query ->
      forAll (choose (0, 3)) $ \k -> ioProperty $ do
        got <- runQuery entries query k
        let expected = [(t, v) | (t, v) <- entries, levenshtein query t <= k]
            keysMatch = List.sort (map fst got) == List.sort (map fst expected)
            valuesMatch =
              all (\(t, (d, i)) -> d == levenshtein query t && lookup t entries == Just i) got
        pure (keysMatch && valuesMatch)

u64Property :: Property
u64Property =
  forAll (oneof [elements [0, 1, 2 ^ (63 :: Int), maxBound], arbitrary]) $ \value ->
    ioProperty $ do
      got <- runQuery [("term", Just value)] "term" 0
      pure (got == [("term", (0, Just value))])

main :: IO ()
main = do
  let args = stdArgs {replay = Just (mkQCGen 20260809, 0), maxSuccess = 200, chatty = False}
      variants =
        [ distanceProperty Lev.distance Lev.distanceThreshold
        , distanceProperty Lev.damerauDistance Lev.damerauDistanceThreshold
        , distanceProperty Lev.trueDamerauDistance Lev.trueDamerauDistanceThreshold
        ]
  results <-
    mapM (quickCheckWithResult args) (variants ++ [oracleProperty, queryProperty, u64Property])
  unless (all isSuccess results) exitFailure
  putStrLn "Haskell property tests passed"
