import numpy as np
# from typing import Tuple
# Offset = Tuple[int, int, int]
from cloudvolume.lib import Vec, Bbox

class OffsetArray(np.ndarray):
    """
        OffsetArray
    a chunk of big array with offset
    this is a python alternative of Julia OffsetArrays.jl
    https://github.com/JuliaArrays/OffsetArrays.jl
    implementation following an example in (ndarray subclassing)
    [https://docs.scipy.org/doc/numpy/user/basics.subclassing.html]
    """
    def __new__(cls, array, global_offset=(0, 0, 0)):
        assert array.ndim == len(global_offset)
        obj = np.asarray(array).view(cls)
        obj.global_offset = global_offset
        return obj

    @classmethod
    def from_bbox(cls, array, bbox):
        global_offset = (bbox.minpt.z, bbox.minpt.y, bbox.minpt.x)
        return OffsetArray(array, global_offset=global_offset)

    def __array_finalize__(self, obj):
        if obj is not None:
            self.info = getattr(obj, 'global_offset', None)

    @property
    def ranges(self):
        return tuple(range(o, o+s)
                     for o, s in zip(self.global_offset, self.shape))

    def where(self, mask):
        """
        find the indexes of masked value as an alternative of np.where function
        args:
            mask (binary ndarray):
        """
        isinstance(mask, np.ndarray)
        assert mask.shape == self.shape
        return (i+o for i, o in zip(np.where(mask), self.global_offset))

    def add_overlap(self, other):
        assert isinstance(other, OffsetArray)
        overlap_slices = self._get_overlap_slices(other.ranges)
        self[overlap_slices] += other[overlap_slices]

    def _get_overlap_slices(self, other_slices):
        return (slice(max(s1.start, s2.start), min(s1.stop, s2.stop))
                for s1, s2 in zip(self.ranges, other_slices))

    def _get_internal_slices(self, slices):
        return (slice(s.start-o, s.stop-o)
                for s, o in zip(slices, self.global_offset))

    def __getitem__(self, slices):
        internalSlices = self._get_internal_slices(self, slices)
        return self[internalSlices]

    def __setitem__(self, slices, inputArr):
        internalSlices = self._get_internal_slices(self, slices)
        self[internalSlices] = inputArr
