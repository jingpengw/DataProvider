import numpy as np

class PatchMask(object):
    """
        PatchMask

    """
    def __init__(size, overlapSize, dtype = np.float32):
        assert len(size) == 3
        assert len(overlapSize) == 3
        assert overlapSize < map(lambda s: s/2, size)
        self.overlapSize = overlapSize
        self.mask = ones(size, dtype)
        self._normalize_overlap()

    @property
    def shape(self):
        return self.mask.shape

    def _normalize_overlap( self ):

def _get_mask(self, key, loc):
         mask = None
         if self.blend:
             assert key in self.max_logits
             max_logit = self.max_logits[key].get_patch(loc)
             mask = self._bump_map(max_logit.shape[-3:], max_logit[0,...])
         return mask

     def _bump_logit(self, z, y, x, t=1.5):
         return -(x*(1-x))**(-t)-(y*(1-y))**(-t)-(z*(1-z))**(-t)

     def _bump_logit_map(self, dim):
         x = range(dim[-1])
         y = range(dim[-2])
         z = range(dim[-3])
         zv, yv, xv = np.meshgrid(z, y, x, indexing='ij')
         xv = (xv+1.0)/(dim[-1]+1.0)
         yv = (yv+1.0)/(dim[-2]+1.0)
         zv = (zv+1.0)/(dim[-3]+1.0)
         return _bump_logit(zv, yv, xv)

     def _bump_map(self, dim, max_logit):
         return np.exp(self._bump_logit_map(dim) - max_logit)
