from __future__ import print_function
import numpy as np
import time
import math

from ..box import centered_box
from ..tensor import WritableTensorData as WTD, \
    WritableTensorDataWithMask as WTDM
from ..emio import imsave

def prepare_outputs(spec, locs, blend=False, blend_mode='', stride=None):
    blend_pool = ['', 'bump', 'aligned-bump']
    b = blend_mode.lower()
    if b not in blend_pool:
        raise RuntimeError('unknown output blend type [%s]' % b)

    if b == '':
        b = 'Blend'
    elif b == 'aligned-bump':
        b = 'AlignedBumpBlend'
    else:
        b = b[0].capitalize() + b[1:] + 'Blend'
    # print('blending mode: {}'.format(b))
    outputs = eval(b + '(spec, locs, blend, stride)')
    return outputs


class Blend(object):
    """
    Blend interface.
    """

    def __init__(self, spec, locs, blend=False, stride=None):
        """Initialize Blend."""
        self.spec = spec
        self.locs = locs
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
        self.op = None
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

    def __init__(self, spec, locs, blend=False, **kwargs):
        """Initialize BumpBlend."""
        super().__init__(self, spec, locs, blend, **kwargs)

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
            t0 = time.time()
            mask = self.get_mask(k, loc)
            t1 = time.time() - t0
            self.data[k].set_patch(loc, v, op=self.op, mask=mask)
            t2 = time.time() - t0
            print('get_mask: %.3f, set_patch: %.3f' % (t1, t2-t1))

    def get_mask(self, key, loc):
        mask = None
        if self.blend:
            assert key in self.max_logits
            max_logit = self.max_logits[key].get_patch(loc)
            mask = self._bump_map(max_logit.shape[-3:], max_logit[0, ...])
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


class AlignedBumpBlend(Blend):
    """
    Blending with bump function with aligned patches.
    """
    def __init__(self, spec, locs, blend=True, stride=None):
        """Initialize BumpBlend."""
        # note that the blend mode is always False in parent class to avoid
        # using the chunk-wise mask
        super().__init__(spec, locs, False)

        self.patch_masks = dict()
        # always add the patches, this will take effect in the push
        # functions of Blend class
        for k, v in self.data.items():
            fov = v.fov()

            assert stride
            if all(np.less_equal(stride, 1.0)):
                # this is in percentile, need to transform to voxel based
                fov = list(self.data.values()).fov()
                stride_by_voxel = (f-math.ceil(f*s) for (f, s) in zip(fov, stride))
            else:
                stride_by_voxel = stride
            print('stride: {}'.format(stride))
            assert all(np.greater_equal(stride_by_voxel, 1))

            mask = self._make_mask(fov, stride_by_voxel)
            assert np.less_equal(mask, 1.0).all()
            self.patch_masks[k] = mask

        self._save_mask()

    def push(self, loc, sample):
        """Write to data."""
        for k, v in sample.items():
            # assert np.less_equal(v, 1.0).all()
            np.multiply(v, self.patch_masks[k], v)
            self.data[k].set_patch(loc, v, op='np.add')

    ####################################################################
    ## Private methods.
    ####################################################################
    def _save_mask(self):
        for k, v in self.patch_masks.items():
            imsave(v, '/tmp/patch_mask_{}.tif'.format(k))

    def _make_mask(self, fov, stride_by_voxel):
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
        stride = stride_by_voxel
        bump_map = self._make_bump_map(fov)
        # use 3x3x3 mask addition to figure out the normalization parameter
        # this is a simulation of blending
        base_mask = np.zeros(tuple(f+2*s for (f, s) in zip(fov, stride)),
                             dtype='float64')
        print('fov: {}, stride: {}'.format(fov, stride))
        print('shape of base mask: {}'.format(base_mask.shape))
        for nz in range(3):
            for ny in range(3):
                for nx in range(3):
                    base_mask[nz*stride[0]:nz*stride[0]+fov[0],
                              ny*stride[1]:ny*stride[1]+fov[1],
                              nx*stride[2]:nx*stride[2]+fov[2]] += bump_map

        bump_map /= base_mask[stride[0]:stride[0]+fov[0],
                              stride[1]:stride[1]+fov[1],
                              stride[2]:stride[2]+fov[2]]

        return np.asarray(bump_map, dtype='float32')

    def _make_bump_map(self, dim):
        x = range(dim[-1])
        y = range(dim[-2])
        z = range(dim[-3])
        zv, yv, xv = np.meshgrid(z, y, x, indexing='ij')
        xv = (xv+1.0)/(dim[-1]+1.0) * 2.0 - 1.0
        yv = (yv+1.0)/(dim[-2]+1.0) * 2.0 - 1.0
        zv = (zv+1.0)/(dim[-3]+1.0) * 2.0 - 1.0
        bump_map = np.exp(-1.0/(1.0-xv*xv) +
                          -1.0/(1.0-yv*yv) +
                          -1.0/(1.0-zv*zv))
        return np.asarray(bump_map, dtype='float64')
