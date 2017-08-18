# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: skip-file
""" file iterator for pasval voc 2012"""
import mxnet as mx
import numpy as np
import sys, os
import scipy.io as sio
from mxnet.io import DataIter
from PIL import Image

class FileIter(DataIter):
    """FileIter object in fcn-xs example. Taking a file list file to get dataiter.
    in this example, we use the whole image training for fcn-xs, that is to say
    we do not need resize/crop the image to the same size, so the batch_size is
    set to 1 here
    Parameters
    ----------
    root_dir : string
        the root dir of image/label lie in
    flist_name : string
        the list file of iamge and label, every line owns the form:
        index \t image_data_path \t image_label_path
    cut_off_size : int
        if the maximal size of one image is larger than cut_off_size, then it will
        crop the image with the minimal size of that image
    data_name : string
        the data name used in symbol data(default data name)
    label_name : string
        the label name used in symbol softmax_label(default label name)
    """
    def __init__(self, roidb, data_root,data_path,batch_size,
                 shuffle=False,rgb_mean = (0.25, 0.25, 0.25),
                 cut_off_size = None,
                 data_name = "data",
                 label_name = "softmax_label"):
        super(FileIter, self).__init__()
        self.data_root = data_root
        self.data_path = data_path
        self.roidb = roidb
        self.size = len(roidb)
        self.index = np.arange(self.size)
        self.mean = np.array(rgb_mean)  # (R, G, B)
        self.batch_size = batch_size
        self.data_name = 'data'
        self.label_name = 'label'
        self.cur = 0
        self.get_batch()

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)
    def get_batch(self):
        cur_form = self.cur
        cur_to = min(self.cur+self.batch_size,self.size)
        roidb = [self.roidb[i] for i in xrange(cur_form,cur_to)]
        data = np.zeros((self.batch_size,3,512,512))
        label = np.zeros((self.batch_size, 1, 512, 512))
        for idx,roi_rec in enumerate(roidb):
            img_path = roi_rec['image']
            mask_path = roi_rec['mask']
            image = sio.loadmat(img_path)['img']
            mask = sio.loadmat(mask_path)['mask']
            #TODO:preprocess and data argument
            image = self._img_process(image)
            mask = self._mask_process(mask)
            data[idx] = image
            #print(mask.shape)
            label[idx] = mask
        self.data = mx.nd.array(data)
        self.label = mx.nd.array(label)
    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        return [mx.io.DataDesc('data',self.data.shape)]

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator"""
        return [mx.io.DataDesc('label',self.label.shape)]

    def get_batch_size(self):
        return self.batch_size
    def iter_next(self):
        self.cur += self.batch_size
        if(self.cur < self.size-1):
            return True
        else:
            return False
    def getindex(self):
        return self.cur/self.batch_size
    def next(self):
        """return one dict which contains "data" and "label" """
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=(self.data,),label=(self.label,),
                                   index = self.getindex(),provide_data=self.provide_data,provide_label=self.provide_label)
        else:
            raise StopIteration
    def _img_process(self,img):
        reshaped_mean = self.mean.reshape(1, 1, 3)
        img = img - reshaped_mean
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)  # (c, h, w)

        return img
    def _mask_process(self,mask):
        mask[mask==255] = 1
        mask = np.swapaxes(mask,0,2)
        mask = np.swapaxes(mask,1,2)
        return mask

