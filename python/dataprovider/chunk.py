import numpy as np
from patch_mask import PatchMask

#from typing import Tuple
#Offset = Tuple[int, int, int]

class Chunk(object):
    """
        Chunk
    a chunk of big array with offset
    this is a python alternative of Julia OffsetArrays.jl
    https://github.com/JuliaArrays/OffsetArrays.jl
    """
    def __init__(self, arr, offset):
        isinstance(arr, np.ndarray)
        self.arr = arr
        self.offset = offset

    def __getitem__(self, slices):
        internalSlices = self._get_internal_slices( self, slices )
        return self.arr[internalSlices]

    def __setitem__(self, slices, inputArr):
        internalSlices = self._get_internal_slices(self, slices)
        self.arr[internalSlices] = inputArr

    @property
    def indices(self):
        return map(lambda o,s: range(o,o+s), self.offset, self.shape)

    @property
    def parent(self):
        return self.arr

    def __iadd__(self, other):
        indices1 = self.indices
        indices2 = other.indices
        overlap_slices = map(lambda i1,i2: slice(   max(i1.start, i2.start), \
                                                    min(i1.stop,  i2.stop)), \
                                                            indices1, indices2)
        self[overlap_slices] += other[overlap_slices]

    def __itruediv__(self, otherArray):
        assert isinstance(otherArray, np.ndarray)
        self.arr /= otherArray

    def _get_internal_slices(self, slices):
        return map(lambda s,o: slice(s.start-o, s.stop-o), slices, self.offset)


class Patch(Chunk):
    """
        Patch
    3D convnet inference input/output patch
    """
    def __init__(self, arr, offset):
        super(Patch, self).__init__(arr, offset)

    def __itruediv__(self, patchMask):
        isinstance(mask, PatchMask)


    def normalize_by_mask(self, mask):
        self /= mask

