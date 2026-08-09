{-# LANGUAGE OverloadedStrings #-}

-- | C9 leak-discipline tests for the Haskell facade (performGC + weak pointers).
--
-- A >=10,000-cycle create\/use\/free loop over transducers (and phonetic
-- patterns) takes a weak pointer to each facade handle, then forces a major
-- collection with 'performGC' and asserts every weak pointer is dead. A
-- retained handle -- one not reclaimed after its last strong reference drops --
-- would keep its weak pointer live, revealing the leak. Native resources are
-- released at close; this confirms the managed wrappers are collected rather
-- than accumulating.
module Main (main) where

import Control.Monad (forM_, unless)
import Data.IORef (modifyIORef', newIORef, readIORef)
import Data.Maybe (isJust)
import qualified Data.Text as Text
import Data.Text.Encoding (encodeUtf8)
import Data.Word (Word64)
import System.Exit (exitFailure)
import System.Mem (performGC)
import System.Mem.Weak (Weak, deRefWeak, mkWeakPtr)
import qualified VinaryTree.Libdictenstein as Dict
import qualified VinaryTree.Liblevenshtein as Lev

cycles :: Int
cycles = 10000

fixture :: [(Text.Text, Maybe Word64)]
fixture = [("cat", Just 1), ("cot", Just 2), ("cut", Just 3), ("scat", Nothing)]

drain :: Lev.Cursor -> IO ()
drain cursor = do
  value <- Lev.next cursor
  case value of
    Nothing -> Lev.closeCursor cursor
    Just _ -> drain cursor

countAlive :: [Weak a] -> IO Int
countAlive references = length . filter isJust <$> mapM deRefWeak references

main :: IO ()
main = do
  transducerWeaks <- newIORef ([] :: [Weak Lev.Transducer])
  forM_ [1 .. cycles] $ \_ -> do
    dictionary <- Dict.dynamicDawg Dict.UnicodeScalar
    forM_ fixture $ \(term, value) -> Dict.putBytes dictionary (encodeUtf8 term) value
    resource <- Dict.resource dictionary
    transducer <- Lev.transducer Lev.Standard resource
    cursor <- Lev.queryText Lev.Traversal transducer "cat" 2
    drain cursor
    Lev.closeTransducer transducer
    Dict.close dictionary
    weak <- mkWeakPtr transducer Nothing
    modifyIORef' transducerWeaks (weak :)

  patternWeaks <- newIORef ([] :: [Weak Lev.PhoneticPattern])
  forM_ [1 .. cycles] $ \_ -> do
    pattern <- Lev.regexPattern "c[ao]t"
    _ <- Lev.patternMatches pattern "cat"
    Lev.closePattern pattern
    weak <- mkWeakPtr pattern Nothing
    modifyIORef' patternWeaks (weak :)

  performGC
  aliveTransducers <- readIORef transducerWeaks >>= countAlive
  alivePatterns <- readIORef patternWeaks >>= countAlive
  unless (aliveTransducers == 0 && alivePatterns == 0) $ do
    putStrLn
      ( "leak: "
          ++ show aliveTransducers
          ++ " transducers and "
          ++ show alivePatterns
          ++ " patterns reachable after performGC"
      )
    exitFailure
  putStrLn "Haskell leak tests passed"
