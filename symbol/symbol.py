import mxnet as mx
from config.config import config


    
def convolution_module(net, kernel_size, pad_size, filter_count, stride=(1, 1), work_space=2048, batch_norm=True, down_pool=False, up_pool=False, act_type="relu", convolution=True):
    if up_pool:
        net = mx.sym.Deconvolution(net, kernel=(2, 2), pad=(0, 0), stride=(2, 2), num_filter=filter_count, workspace = work_space)
        net = mx.sym.BatchNorm(net)
        if act_type != "":
            net = mx.sym.Activation(net, act_type=act_type)

    if convolution:
        conv = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count, workspace=work_space)
        net = conv

    if batch_norm:
        net = mx.sym.BatchNorm(net)

    if act_type != "":
        net = mx.sym.Activation(net, act_type=act_type)

    if down_pool:
        pool = mx.sym.Pooling(net, pool_type="max", kernel=(2, 2), stride=(2, 2))
        net = pool
    return net

    return net

def get_symbol_180():
    source = mx.sym.Variable("data")
    mask = mx.sym.Variable('label')
    kernel_size = (3, 3)
    pad_size = (1, 1)
    filter_count = 32
    pool1 = convolution_module(source, kernel_size, pad_size, filter_count=filter_count, down_pool=True)
    net = pool1
    pool2 = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 2, down_pool=True)
    net = pool2
    pool3 = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 4, down_pool=True)
    net = pool3
    pool4 = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 4, down_pool=True)
    net = pool4
    net = mx.sym.Dropout(net)
    pool5 = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 8, down_pool=True)
    net = pool5
    net = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 4, up_pool=True)
    net = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 4, up_pool=True)
    net = mx.sym.Concat(*[pool3, net])
    net = mx.sym.Dropout(net)
    net = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 4)
    net = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 4, up_pool=True)

    net = mx.sym.Concat(*[pool2, net])
    net = mx.sym.Dropout(net)
    net = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 4)
    net = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 4, up_pool=True)
    convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 4)
    net = mx.sym.Concat(*[pool1, net])
    net = mx.sym.Dropout(net)
    net = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 2)
    net = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 2, up_pool=True)

    net = convolution_module(net, kernel_size, pad_size, filter_count=1, batch_norm=False, act_type="")
    dice_coef = dice_coef_loss(net,mask)
    return mx.symbol.MakeLoss(dice_coef,grad_scale=1./config.TRAIN.BATCH_SIZE)
    return net

def dice_coef_loss(data,mask):
    pred_f = mx.sym.flatten(data=data,name='flatten_pred')
    mask_f =mx.sym.flatten(data=mask,name='flatten_ture')
    intersection = mx.sym.sum(pred_f * mask_f,axis=1)
    return -(2*intersection + 1) / (mx.sym.sum(pred_f,axis=1) + mx.sym.sum(mask_f,axis=1) + 1)