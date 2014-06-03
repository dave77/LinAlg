{-# LANGUAGE ForeignFunctionInterface #-}
module Bvls(
	bvls
) where

-- Foreign Function Interface Includes
import Foreign.Ptr
import Foreign.C.Types
import Foreign.Marshal.Alloc
import Foreign.Marshal.Array
import System.IO.Unsafe
import Data.List

foreign import ccall unsafe "bvls.h bvls"
	c_bvls :: CInt -> CInt -> Ptr CDouble -> Ptr CDouble -> Ptr CDouble
		-> Ptr CDouble -> Ptr CDouble -> IO CInt

convert :: (Fractional b, Real a) => [a] -> [b]
convert = map realToFrac

bvls :: [[Double]] -> [Double] -> [Double] -> [Double] -> Maybe [Double] 
bvls a b lb ub = unsafePerformIO $ do
	x <- mallocArray n
	rc <-   withArray (convert $ pack2d a)	$ \a' ->
		withArray (convert b)		$ \b' ->
		withArray (convert lb)		$ \lb' ->
		withArray (convert ub)		$ \ub' ->
		c_bvls m' n' a' b' lb' ub' x
	let res = case rc of
		0 -> do
			tmp <- peekArray n x
			return (Just $ convert tmp) 
		_ -> return Nothing
	lift <- res
	free x
	return lift
	where
		pack2d = concat . transpose
		m' = fromIntegral $ length a
		n = length $ head a
		n' = fromIntegral n::CInt
