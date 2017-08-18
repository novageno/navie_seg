import mxnet as mx
import numpy as np

class DiceMetric(mx.metric.EvalMetric):
    
    def __init__(self):
        super(DiceMetric,self).__init__('DiceMetric')
        
    def update(self,labels,preds):
        """preds = preds
        labels = labels
        x= mx.nd.flatten(preds)
        y = mx.nd.flatten(labels)
        intersection = mx.nd.sum(x * y,axis=1)
        dice = (2 * intersection + 1) / (mx.nd.sum(x,axis=1) + mx.nd.sum(y,axis=1) + 1)"""
        preds = preds.asnumpy()
        self.sum_metric += np.sum(preds)
        self.num_inst += preds.shape[0]
        