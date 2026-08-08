{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE LambdaCase #-}
module Main (main) where

import Control.Monad (forM_, unless)
import qualified Data.ByteString.Char8 as BS
import Data.Maybe (fromJust)
import qualified Data.Text as Text
import qualified Data.Text.Encoding as TextEncoding
import System.Directory (getTemporaryDirectory, removeFile)
import System.FilePath ((</>), replaceExtension)
import System.IO.Error (catchIOError)
import qualified VinaryTree.Libdictenstein as Dict
import qualified VinaryTree.Liblevenshtein as Lev

assertIO :: Bool -> String -> IO ()
assertIO condition message = unless condition (ioError (userError message))

collect :: Lev.Cursor -> IO [Lev.Match]
collect cursor = go []
  where
    go values = Lev.next cursor >>= \case
      Nothing -> Lev.closeCursor cursor >> pure (reverse values)
      Just value -> go (value : values)

textTerm :: Lev.Match -> Text.Text
textTerm value = case Lev.term value of
  Lev.TextTerm text -> text
  _ -> error "expected text term"

removeIfPresent :: FilePath -> IO ()
removeIfPresent path = removeFile path `catchIOError` const (pure ())

main :: IO ()
main = do
  forM_ [0 .. 63 :: Int] $ \trace -> do
    dictionary <- Dict.dynamicDawg Dict.UnicodeScalar
    let entries =
          [ (BS.pack ("t" <> show trace <> "-" <> show index), Just (fromIntegral index))
          | index <- [0 .. 15 :: Int]
          ]
    inserted <- Dict.putManyBytes dictionary entries
    assertIO (inserted == 16) "batch insert count"
    shared <- Dict.resource dictionary
    automaton <- Lev.transducer Lev.Standard shared
    expected <- Lev.queryText Lev.DistanceThenTerm automaton "" 64 >>= collect
    cursor <- Lev.queryText Lev.DistanceThenTerm automaton "" 64
    first <- fromJust <$> Lev.next cursor
    _ <- Dict.removeBytes dictionary (BS.pack ("t" <> show trace <> "-1"))
    _ <- Dict.putBytes dictionary (BS.pack ("t" <> show trace <> "-2")) (Just 999)
    second <- fromJust <$> Lev.next cursor
    Dict.clear dictionary
    _ <- Dict.compact dictionary
    _ <- Dict.putBytes dictionary (BS.pack ("after-" <> show trace)) (Just 1000)
    remainder <- collect cursor
    assertIO (first : second : remainder == expected) "query-start snapshot changed"
    Dict.close dictionary
    fresh <- Lev.queryText Lev.Traversal automaton "" 64 >>= collect
    assertIO (map textTerm fresh == [Text.pack ("after-" <> show trace)])
      "retained resource did not observe fresh state"
    Lev.closeTransducer automaton

  let cafe = TextEncoding.encodeUtf8 "café"
  dat <- Dict.doubleArrayTrie Dict.UnicodeScalar
    [(cafe, Just 7), ("caff", Nothing)]
  lookupDat <- Dict.getBytes dat cafe
  assertIO (Dict.mappedValue lookupDat == Just 7) "DAT lookup"
  Dict.close dat

  suffixes <- Dict.scdawg Dict.UnicodeScalar
  _ <- Dict.putBytes suffixes "cat" (Just 1)
  _ <- Dict.putBytes suffixes "cot" (Just 2)
  containsSuffix <- Dict.containsSubstring suffixes "ot"
  frequency <- Dict.substringFrequency suffixes "t"
  assertIO (containsSuffix && frequency == 2) "SCDAWG substring"
  Dict.close suffixes

  tokens <- Dict.dynamicDawg Dict.U64
  _ <- Dict.putU64 tokens [1, 2, 3] (Just 12)
  tokenResource <- Dict.resource tokens
  tokenAutomaton <- Lev.transducer Lev.Standard tokenResource
  tokenCursor <- Lev.queryU64 Lev.Traversal tokenAutomaton [1, 2, 4] 1
  tokenMatch <- fromJust <$> Lev.next tokenCursor
  assertIO (Lev.term tokenMatch == Lev.TokensTerm [1, 2, 3]
            && Lev.editDistance tokenMatch == 1
            && Lev.identifier tokenMatch == Just 12) "u64 query"
  Lev.closeCursor tokenCursor
  Dict.close tokens
  Lev.closeTransducer tokenAutomaton

  temporary <- getTemporaryDirectory
  let persistentPath = temporary </> "vinary-tree-haskell.artrie"
      vocabularyPath = temporary </> "vinary-tree-haskell-vocabulary.vocab"
      persistenceArtifacts =
        [ persistentPath, replaceExtension persistentPath "wal", persistentPath <> ".wlock"
        , vocabularyPath, vocabularyPath <> ".wal", vocabularyPath <> ".wlock"
        ]
  mapM_ removeIfPresent
    persistenceArtifacts
  persistent <- Dict.createPersistentARTrie Dict.UnicodeScalar persistentPath
  _ <- Dict.putBytes persistent "durable" (Just 17)
  Dict.checkpoint persistent
  Dict.close persistent
  reopened <- Dict.openPersistentARTrie Dict.UnicodeScalar persistentPath
  durable <- Dict.getBytes reopened "durable"
  assertIO (Dict.mappedValue durable == Just 17) "persistent ARTrie reopen"
  Dict.close reopened

  vocabulary <- Dict.createPersistentVocabulary vocabularyPath
  _ <- Dict.putBytes vocabulary "alpha" (Just 0)
  Dict.checkpoint vocabulary
  term <- Dict.vocabularyTerm vocabulary 0
  assertIO (term == Just "alpha") "vocabulary reverse lookup"
  Dict.close vocabulary
  reopenedVocabulary <- Dict.openPersistentVocabulary vocabularyPath
  alpha <- Dict.getBytes reopenedVocabulary "alpha"
  assertIO (Dict.mappedValue alpha == Just 0) "vocabulary reopen"
  Dict.close reopenedVocabulary

  pattern' <- Lev.regexPattern "c[ao]t"
  matchesCat <- Lev.patternMatches pattern' "cat"
  matchesCut <- Lev.patternMatches pattern' "cut"
  (states, transitions) <- Lev.patternSize pattern'
  assertIO (matchesCat && not matchesCut && states > 0 && transitions > 0)
    "phonetic pattern"
  Lev.closePattern pattern'
  rules <- Lev.phoneticRules "english-orthography"
  ruleCount <- Lev.rulesLength rules
  _ <- Lev.applyRules rules "phone"
  assertIO (ruleCount > 0) "phonetic rules"
  Lev.closeRules rules
  standard <- Lev.distance "kitten" "sitting"
  standardThreshold <- Lev.distanceThreshold "kitten" "sitting" 3
  damerau <- Lev.damerauDistance "ca" "ac"
  damerauThreshold <- Lev.damerauDistanceThreshold "ca" "ac" 1
  trueDamerau <- Lev.trueDamerauDistance "ca" "ac"
  trueThreshold <- Lev.trueDamerauDistanceThreshold "ca" "ac" 1
  assertIO (standard == 3 && standardThreshold == 3) "Levenshtein distances"
  assertIO (damerau == 1 && damerauThreshold == 1) "Damerau distances"
  assertIO (trueDamerau == 1 && trueThreshold == 1) "true Damerau distances"

  mapM_ removeIfPresent persistenceArtifacts
  putStrLn "Haskell binding snapshot integration passed"
