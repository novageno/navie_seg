import mxnet as mx
import argparse
import logging
from config.config import config
from dataset.LIDC import LIDC
from Loader import FileIter
from symbol.symbol import get_symbol_180
from lr_scheduler import WarmupMultiFactorScheduler
from metric import DiceMetric


def train_net(args,ctx):
    #Create the log
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    imdb = LIDC(config.dataset.imageset,config.dataset.root_path,config.dataset.data_path,config.dataset.train_folds,config.dataset.output_path)
    roidb = imdb.gt_roidb()
    
    train_data = FileIter(roidb,config.dataset.root_path,config.dataset.data_path,config.TRAIN.BATCH_SIZE,shuffle=True)
    #TODO:continue training
    if args.resume:
        logger.warning('Not Implement!')
    else:
        arg_params = None
        aux_params = None
    data_names = train_data.provide_data[0]

    label_names = train_data.provide_label[0]
    sym = get_symbol_180()
    #print(sym.infer_shape(data=(32,3,512,512))[1])
    mod = mx.mod.Module(sym,data_names=('data',),label_names=('label',),
                        logger=logger,context=ctx)
    eval_metric = DiceMetric()
    base_lr = config.TRAIN.lr
    print(base_lr)
    lr_factor = 0.1
    lr_epoch = [float(epoch) for epoch in config.TRAIN.lr_step.split(',')]
    lr_epoch_diff = [config.TRAIN.epoch - config.TRAIN.begin_epoch for epoch in lr_epoch if epoch > config.TRAIN.begin_epoch]
    lr = base_lr * (lr_factor ** (len(lr_epoch) - len(lr_epoch_diff)))
    lr_iters = [int(epoch * len(roidb) / config.TRAIN.BATCH_SIZE) for epoch in lr_epoch_diff]
    lr_scheduler = WarmupMultiFactorScheduler(lr_iters, config.TRAIN.lr_factor, config.TRAIN.warmup, config.TRAIN.warmup_lr, config.TRAIN.warmup_step)
    #batch_end_callback = callback.Speedometer(train_data.batch_size, frequent=args.frequent)
    optimizer_params = {'momentum': config.TRAIN.momentum,
                        'wd': config.TRAIN.wd,
                        'learning_rate': lr,
                        'lr_scheduler': lr_scheduler,
                        'rescale_grad': 1.0,
                        'clip_gradient': None}
    mod.fit(train_data=train_data,eval_metric=eval_metric,
            batch_end_callback = mx.callback.Speedometer(config.TRAIN.BATCH_SIZE,10),
            epoch_end_callback = mx.callback.do_checkpoint(config.model_prefix),
            optimizer='sgd', optimizer_params=optimizer_params,allow_missing=True,initializer=mx.initializer.Xavier(),
            arg_params=arg_params,aux_params=aux_params,num_epoch=config.TRAIN.epoch)
def arg_parse():
    parser = argparse.ArgumentParser(description='Training U-net to get candidates')
    parser.add_argument('--resume',default=False)
    args = parser.parse_args()
    return args
def main():
    args = arg_parse()
    print ('called with argument')
    ctx = mx.gpu(1)
    train_net(args,ctx)

if __name__ =="__main__":
    main()
    