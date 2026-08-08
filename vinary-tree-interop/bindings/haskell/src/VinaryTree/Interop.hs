{-# LANGUAGE DerivingStrategies #-}
module VinaryTree.Interop
  ( DictionaryResource
  , VtResource
  , UnitDomain(..)
  , fromOwnedResource
  , withDictionaryResource
  ) where

import Foreign.ForeignPtr (ForeignPtr, newForeignPtr, withForeignPtr)
import Foreign.Ptr (FunPtr, Ptr)

data VtResource
newtype DictionaryResource = DictionaryResource (ForeignPtr VtResource)

data UnitDomain = Byte | UnicodeScalar | U64
  deriving stock (Eq, Ord, Show, Enum, Bounded)

foreign import ccall unsafe "&vt_hs_resource_free"
  resourceFinalizer :: FunPtr (Ptr VtResource -> IO ())

fromOwnedResource :: Ptr VtResource -> IO DictionaryResource
fromOwnedResource pointer = DictionaryResource <$> newForeignPtr resourceFinalizer pointer

withDictionaryResource :: DictionaryResource -> (Ptr VtResource -> IO a) -> IO a
withDictionaryResource (DictionaryResource resource) = withForeignPtr resource
