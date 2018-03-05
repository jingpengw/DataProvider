from __future__ import print_function
import numpy as np
#import time

from ..box import centered_box
from ..tensor import WritableTensorData as WTD, WritableTensorDataWithMask as WTDM


def prepare_outputs(spec, locs, blend=False, blend_mode=''):
    blend_pool = ['','bump']
    b = blend_mode.lower()
    if b not in blend_pool:
        raise RuntimeError('unknown output blend type [%s]' % b)

    if b == '':
        b = 'Blend'
    else:
        b = b[0].capitalize() + b[1:] + 'Blend'
    outputs = eval(b + '(spec, locs, blend)')
    return outputs


class Blend(object):
    """
    Blend interface.
    """

    def __init__(self, spec, locs, blend=False):
        """Initialize Blend."""
        self.spec  = spec
        self.locs  = locs
        self.blend = blend
        self._prepare_data()

    def push(self, loc, sample):
        """Write to data."""
        for k, v in sample.items():
            assert k in self.data
            self.data[k].set_patch(loc, v, op=self.op)

    def get_data(self, key):
        """Get inference output data."""
        assert key in self.data
        return self.data[key].get_data()

    def voxels(self):
        voxels = list()
        for k, v in self.data.items():
            voxels.append(np.prod(v.dim()))
        return min(voxels)

    ####################################################################
    ## Private Methods.
    ####################################################################

    def _prepare_data(self):
        """
        TODO(kisuk): Documentation.
        """
        assert len(self.locs) > 0
        rmin = self.locs[0]
        rmax = self.locs[-1]

        self.data = dict()
        self.op   = None
        for k, v in self.spec.items():
            fov = v[-3:]
            a = centered_box(rmin, fov)
            b = centered_box(rmax, fov)
            c = a.merge(b)
            shape = v[:-3] + tuple(c.size())
            # Inference with overlapping window.
            if self.blend:
                self.data[k] = WTDM(shape, fov, c.min())
                self.op = 'np.add'
            else:
                self.data[k] = WTD(shape, fov, c.min())


class BumpBlend(Blend):
    """
    Blending with bump function.
    """

    def __init__(self, spec, locs, blend=False):
        """Initialize BumpBlend."""
        Blend.__init__(self, spec, locs, blend)

        self.logit_maps = dict()

        # Inference with overlapping window.
        self.max_logits = None
        if blend:
            max_logits = dict()
            # Compute max_logit for numerical stability.
            for k, v in self.data.items():
                fov = tuple(v.fov())
                data = np.full(v.dim(), -np.inf, dtype='float32')
                max_logit = WTD(data, fov, v.offset())
                max_logit_window = self._bump_logit_map(fov)
                for loc in self.locs:
                    max_logit.set_patch(loc, max_logit_window, op='np.maximum')
                max_logits[k] = max_logit
            self.max_logits = max_logits

    def push(self, loc, sample):
        """Blend with data."""
        for k, v in sample.items():
            assert k in self.data
            #t0 = time.time()
            mask = self.get_mask(k, loc)
            #t1 = time.time() - t0
            self.data[k].set_patch(loc, v, op=self.op, mask=mask)
            #t2 = time.time() - t0
            # print('get_mask: %.3f, set_patch: %.3f' % (t1, t2-t1))

    def get_mask(self, key, loc):
        mask = None
        if self.blend:
            assert key in self.max_logits
            max_logit = self.max_logits[key].get_patch(loc)
            mask = self._bump_map(max_logit.shape[-3:], max_logit[0,...])
        return mask

    ####################################################################
    ## Private methods.
    ####################################################################

    def _bump_logit(self, z, y, x, t=1.5):
        return -(x*(1-x))**(-t)-(y*(1-y))**(-t)-(z*(1-z))**(-t)

    def _bump_logit_map(self, dim):
        ret = self.logit_maps.get(dim)
        if ret is None:
            x = range(dim[-1])
            y = range(dim[-2])
            z = range(dim[-3])
            zv, yv, xv = np.meshgrid(z, y, x, indexing='ij')
            xv = (xv+1.0)/(dim[-1]+1.0)
            yv = (yv+1.0)/(dim[-2]+1.0)
            zv = (zv+1.0)/(dim[-3]+1.0)
            ret = self._bump_logit(zv, yv, xv)
            self.logit_maps[dim] = ret
        return ret

    def _bump_map(self, dim, max_logit):
        return np.exp(self._bump_logit_map(dim) - max_logit)


class AlignedBumpBlend(BumpBlend):
    """
    Blending with bump function.
    """

    def __init__(self, spec, locs, blend=False):
        """Initialize BumpBlend."""
        super(AlignedBumpBlend, self).__init__(spec, locs, blend)

        self.patchMask = None
        if self.blend:
            self.fov = self.data.values[0].fov()
            self.patchMask = _make_mask(fov)

    def get_mask(self, key, loc):
        """
        no matter where the patch is, always use the same normalization mask
        """
        return self.patchMask

    ####################################################################
    ## Private methods.
    ####################################################################

    @property
    def stride(self):
        stride_ratio = self.spec['scan_params']['stride']
        return map(lambda f,r:f*r, self.fov, stride_ratio)

    def _make_mask( self ):
        """
            _make_mask( size )
        params:
            size:tuple of int
        return:
            an numpy array with data type of float32. The value was generated
            using a bump function. the overlapping borders and corners were
            normalized according to weight accumulation.
            https://en.wikipedia.org/wiki/Bump_function
        """
        stride = self.stride
        fov = self.fov
        bumpMap = bump_map( fov )
        # use 3x3x3 mask addition to figure out the normalization parameter
		# this is a simulation of blending
        baseMask = np.zeros( tuple(map(lambda f,s: f+2*s, fov, stride)), \
                                                        dtype=np.float32 )
        for nz in range(3):
            for ny in range(3):
                for nx in range(3):
                    baseMask[nz*stride[0]:nz*stride[0]+fov[0], \
                             ny*stride[1]:ny*stride[1]+fov[1], \
                             nx*stride[2]:nx*stride[2]+fov[2]] += bumpMap
        self.patchMask = bumpMap / baseMask[stride[0]:stride[0]+fov[0], \
                                            stride[1]:stride[1]+fov[1], \
                                            stride[2]:stride[2]+fov[2]]

    def _bump_map( dim ):
        x = range(dim[-1])
        y = range(dim[-2])
        z = range(dim[-3])
        zv, yv, xv = np.meshgrid(z, y, x, indexing='ij')
        xv = (xv+1.0)/(dim[-1]+1.0) * 2.0 - 1.0
        yv = (yv+1.0)/(dim[-2]+1.0) * 2.0 - 1.0
        zv = (zv+1.0)/(dim[-3]+1.0) * 2.0 - 1.0
        return 	np.exp(-1.0/(1.0-xv*xv)) * \
				np.exp(-1.0/(1.0-yv*yv)) * \
				np.exp(-1.0/(1.0-zv*zv))
