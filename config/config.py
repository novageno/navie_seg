from easydict import EasyDict as easydict

config = easydict()
config.model_prefix = 'u_net'
config.dataset = easydict()
config.dataset.dataset = 'LIDC'
config.dataset.imageset = 'trainval'
config.dataset.root_path = './data/'
config.dataset.train_folds = 'fold1,fold2,fold3,fold4,fold5,fold6,fold7,fold8,fold9'
config.dataset.val_folds = 'fold10'
config.dataset.data_path = '/home/genonova/u-net/data/faster_rcnn_data/v9'
config.dataset.output_path = './output'

config.TRAIN = easydict()
config.TRAIN.BATCH_SIZE = 32
config.TRAIN.lr = 1e-3
config.TRAIN.warmup_lr = 0.00005
config.TRAIN.lr_step = '15'
config.TRAIN.lr_factor = 0.1
config.TRAIN.wd = 0.001
config.TRAIN.begin_epoch = 0
config.TRAIN.epoch = 20
config.TRAIN.warmup = True
config.TRAIN.warmup_step=100
config.TRAIN.momentum = 0.9

