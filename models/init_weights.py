import jittor as jt
from jittor import init
import numpy as np

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.gauss_(m.weight, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.gauss_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.gauss_(m.weight, 1.0, 0.02)
        init.constant_(m.bias, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_gauss_(m.weight, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_gauss_(m.weight, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.gauss_(m.weight, 1.0, 0.02)
        init.constant_(m.bias, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_gauss_(m.weight, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_gauss_(m.weight, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.gauss_(m.weight, 1.0, 0.02)
        init.constant_(m.bias, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        # Implement orthogonal init manually
        shape = m.weight.shape
        if len(shape) < 2:
            # Fallback for 1D or scalar? Usually Conv/Linear are >= 2D
            init.gauss_(m.weight, 0.0, 0.02)
        else:
            rows = shape[0]
            cols = np.prod(shape[1:])
            a = np.random.normal(0.0, 1.0, (rows, cols))
            u, _, v = np.linalg.svd(a, full_matrices=False)
            q = u if u.shape == (rows, cols) else v
            q = q.reshape(shape)
            # Assign the orthogonal weights
            m.weight.assign(jt.array(q).float32())
            
    elif classname.find('BatchNorm') != -1:
        init.gauss_(m.weight, 1.0, 0.02)
        init.constant_(m.bias, 0.0)


def init_weights(net, init_type='normal'):
    # Define a helper to apply initialization
    def apply_fn(m):
        if init_type == 'normal':
            weights_init_normal(m)
        elif init_type == 'xavier':
            weights_init_xavier(m)
        elif init_type == 'kaiming':
            weights_init_kaiming(m)
        elif init_type == 'orthogonal':
            weights_init_orthogonal(m)
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

    # Jittor modules usually support .modules() to iterate over all submodules
    if hasattr(net, 'modules'):
        for m in net.modules():
            apply_fn(m)
    else:
        # Fallback: try to apply to net itself if it's a layer, 
        # or warn/error if structure is unknown. 
        # But usually net is a jt.nn.Module which has modules()
        apply_fn(net)
