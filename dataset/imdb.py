"""
General image database
An image database creates a list of relative image path called image_set_index and
transform index to absolute image path. As to training, it is necessary that ground
truth and proposals are mixed together for training.
roidb
basic format [image_index]
['image', 'height', 'width', 'flipped',
'boxes', 'gt_classes', 'gt_overlaps', 'max_classes', 'max_overlaps', 'bbox_targets']
"""

import os
import cPickle
import numpy as np
from PIL import Image

def get_flipped_entry_outclass_wrapper(IMDB_instance, seg_rec):
    return IMDB_instance.get_flipped_entry(seg_rec)

class IMDB(object):
    def __init__(self, name, image_set, root_path, dataset_path, result_path=None):
        """
        basic information about an image database
        :param name: name of image database will be used for any output
        :param root_path: root path store cache and proposal data
        :param dataset_path: dataset path store images and image lists
        """
        self.name = name + '_' + image_set
        self.image_set = image_set
        self.root_path = root_path
        self.data_path = dataset_path
        self._result_path = result_path

        # abstract attributes
        self.classes = []
        self.num_classes = 0
        self.image_set_index = []
        self.num_images = 0

        self.config = {}

    def image_path_from_index(self, index):
        raise NotImplementedError

    def gt_roidb(self):
        raise NotImplementedError

    def evaluate_detections(self, detections):
        raise NotImplementedError

    def evaluate_segmentations(self, segmentations):
        raise NotImplementedError

    @property
    def cache_path(self):
        """
        make a directory to store all caches
        :return: cache path
        """
        cache_path = os.path.join(self.root_path, 'cache')
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
        return cache_path

    @property
    def result_path(self):
        if self._result_path and os.path.exists(self._result_path):
            return self._result_path
        else:
            return self.cache_path

    def image_path_at(self, index):
        """
        access image at index in image database
        :param index: image index in image database
        :return: image path
        """
        return self.image_path_from_index(self.image_set_index[index])

    def load_rpn_data(self, full=False):
        if full:
            rpn_file = os.path.join(self.result_path, 'rpn_data', self.name + '_full_rpn.pkl')
        else:
            rpn_file = os.path.join(self.result_path, 'rpn_data', self.name + '_rpn.pkl')
        print 'loading {}'.format(rpn_file)
        assert os.path.exists(rpn_file), 'rpn data not found at {}'.format(rpn_file)
        with open(rpn_file, 'rb') as f:
            box_list = cPickle.load(f)
        return box_list

    def load_rpn_roidb(self, gt_roidb):
        """
        turn rpn detection boxes into roidb
        :param gt_roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        :return: roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        box_list = self.load_rpn_data()
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def rpn_roidb(self, gt_roidb, append_gt=False):
        """
        get rpn roidb and ground truth roidb
        :param gt_roidb: ground truth roidb
        :param append_gt: append ground truth
        :return: roidb of rpn
        """
        if append_gt:
            print 'appending ground truth annotations'
            rpn_roidb = self.load_rpn_roidb(gt_roidb)
            roidb = IMDB.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self.load_rpn_roidb(gt_roidb)
        return roidb

    def create_roidb_from_box_list(self, box_list, gt_roidb):
        """
        given ground truth, prepare roidb
        :param box_list: [image_index] ndarray of [box_index][x1, x2, y1, y2]
        :param gt_roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        :return: roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        assert len(box_list) == self.num_images, 'number of boxes matrix must match number of images'
        roidb = []
        for i in range(self.num_images):
            roi_rec = dict()
            roi_rec['image'] = gt_roidb[i]['image']
            roi_rec['height'] = gt_roidb[i]['height']
            roi_rec['width'] = gt_roidb[i]['width']

            boxes = box_list[i]
            if boxes.shape[1] == 5:
                boxes = boxes[:, :4]
            num_boxes = boxes.shape[0]
            overlaps = np.zeros((num_boxes, self.num_classes), dtype=np.float32)
            if gt_roidb is not None and gt_roidb[i]['boxes'].size > 0:
                gt_boxes = gt_roidb[i]['boxes']
                gt_classes = gt_roidb[i]['gt_classes']
                # n boxes and k gt_boxes => n * k overlap
                gt_overlaps = bbox_overlaps(boxes.astype(np.float), gt_boxes.astype(np.float))
                # for each box in n boxes, select only maximum overlap (must be greater than zero)
                argmaxes = gt_overlaps.argmax(axis=1)
                maxes = gt_overlaps.max(axis=1)
                I = np.where(maxes > 0)[0]
                overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]

            roi_rec.update({'boxes': boxes,
                            'gt_classes': np.zeros((num_boxes,), dtype=np.int32),
                            'gt_overlaps': overlaps,
                            'max_classes': overlaps.argmax(axis=1),
                            'max_overlaps': overlaps.max(axis=1),
                            'flipped': False})

            # background roi => background class
            zero_indexes = np.where(roi_rec['max_overlaps'] == 0)[0]
            assert all(roi_rec['max_classes'][zero_indexes] == 0)
            # foreground roi => foreground class
            nonzero_indexes = np.where(roi_rec['max_overlaps'] > 0)[0]
            assert all(roi_rec['max_classes'][nonzero_indexes] != 0)

            roidb.append(roi_rec)

        return roidb

    def get_flipped_entry(self, seg_rec):
        return {'image': self.flip_and_save(seg_rec['image']),
                'seg_cls_path': self.flip_and_save(seg_rec['seg_cls_path']),
                'height': seg_rec['height'],
                'width': seg_rec['width'],
                'flipped': True}

    def append_flipped_images_for_segmentation(self, segdb):
        """
        append flipped images to an roidb
        flip boxes coordinates, images will be actually flipped when loading into network
        :param segdb: [image_index]['seg_cls_path', 'flipped']
        :return: segdb: [image_index]['seg_cls_path', 'flipped']
        """
        print 'append flipped images to segdb'
        assert self.num_images == len(segdb)
        pool = Pool(processes=1)
        pool_result = []
        for i in range(self.num_images):
            seg_rec = segdb[i]
            pool_result.append(pool.apply_async(get_flipped_entry_outclass_wrapper, args=(self, seg_rec, )))
            #self.get_flipped_entry(seg_rec, segdb_flip, i)
        pool.close()
        pool.join()
        segdb_flip = [res_instance.get() for res_instance in pool_result]
        segdb += segdb_flip
        self.image_set_index *= 2
        return segdb

    def append_flipped_images(self, roidb):
        """
        append flipped images to an roidb
        flip boxes coordinates, images will be actually flipped when loading into network
        :param roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        :return: roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        print 'append flipped images to roidb'
        assert self.num_images == len(roidb)
        for i in range(self.num_images):
            roi_rec = roidb[i]
            boxes = roi_rec['boxes'].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = roi_rec['width'] - oldx2 - 1
            boxes[:, 2] = roi_rec['width'] - oldx1 - 1
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            entry = {'image': roi_rec['image'],
                     'height': roi_rec['height'],
                     'width': roi_rec['width'],
                     'boxes': boxes,
                     'gt_classes': roidb[i]['gt_classes'],
                     'gt_overlaps': roidb[i]['gt_overlaps'],
                     'max_classes': roidb[i]['max_classes'],
                     'max_overlaps': roidb[i]['max_overlaps'],
                     'flipped': True}

            # if roidb has mask
            if 'cache_seg_inst' in roi_rec:
                [filename, extension] = os.path.splitext(roi_rec['cache_seg_inst'])
                entry['cache_seg_inst'] = os.path.join(filename + '_flip' + extension)

            roidb.append(entry)

        self.image_set_index *= 2
        return roidb

    def flip_and_save(self, image_path):
        """
        flip the image by the path and save the flipped image with suffix 'flip'
        :param path: the path of specific image
        :return: the path of saved image
        """
        [image_name, image_ext] = os.path.splitext(os.path.basename(image_path))
        image_dir = os.path.dirname(image_path)
        saved_image_path = os.path.join(image_dir, image_name + '_flip' + image_ext)
        try:
            flipped_image = Image.open(saved_image_path)
        except:
            flipped_image = Image.open(image_path)
            flipped_image = flipped_image.transpose(Image.FLIP_LEFT_RIGHT)
            flipped_image.save(saved_image_path, 'png')
        return saved_image_path

    def evaluate_recall(self, roidb, candidate_boxes=None, thresholds=None):
        """
        evaluate detection proposal recall metrics
        record max overlap value for each gt box; return vector of overlap values
        :param roidb: used to evaluate
        :param candidate_boxes: if not given, use roidb's non-gt boxes
        :param thresholds: array-like recall threshold
        :return: None
        ar: average recall, recalls: vector recalls at each IoU overlap threshold
        thresholds: vector of IoU overlap threshold, gt_overlaps: vector of all ground-truth overlaps
        """
        all_log_info = ''
        area_names = ['all', '0-25', '25-50', '50-100',
                      '100-200', '200-300', '300-inf']
        area_ranges = [[0**2, 1e5**2], [0**2, 25**2], [25**2, 50**2], [50**2, 100**2],
                       [100**2, 200**2], [200**2, 300**2], [300**2, 1e5**2]]
        area_counts = []
        for area_name, area_range in zip(area_names[1:], area_ranges[1:]):
            area_count = 0
            for i in range(self.num_images):
                if candidate_boxes is None:
                    # default is use the non-gt boxes from roidb
                    non_gt_inds = np.where(roidb[i]['gt_classes'] == 0)[0]
                    boxes = roidb[i]['boxes'][non_gt_inds, :]
                else:
                    boxes = candidate_boxes[i]
                boxes_areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
                valid_range_inds = np.where((boxes_areas >= area_range[0]) & (boxes_areas < area_range[1]))[0]
                area_count += len(valid_range_inds)
            area_counts.append(area_count)
        total_counts = float(sum(area_counts))
        for area_name, area_count in zip(area_names[1:], area_counts):
            log_info = 'percentage of {} {}'.format(area_name, area_count / total_counts)
            print log_info
            all_log_info += log_info
        log_info = 'average number of proposal {}'.format(total_counts / self.num_images)
        print log_info
        all_log_info += log_info
        for area_name, area_range in zip(area_names, area_ranges):
            gt_overlaps = np.zeros(0)
            num_pos = 0
            for i in range(self.num_images):
                # check for max_overlaps == 1 avoids including crowd annotations
                max_gt_overlaps = roidb[i]['gt_overlaps'].max(axis=1)
                gt_inds = np.where((roidb[i]['gt_classes'] > 0) & (max_gt_overlaps == 1))[0]
                gt_boxes = roidb[i]['boxes'][gt_inds, :]
                gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0] + 1) * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)
                valid_gt_inds = np.where((gt_areas >= area_range[0]) & (gt_areas < area_range[1]))[0]
                gt_boxes = gt_boxes[valid_gt_inds, :]
                num_pos += len(valid_gt_inds)

                if candidate_boxes is None:
                    # default is use the non-gt boxes from roidb
                    non_gt_inds = np.where(roidb[i]['gt_classes'] == 0)[0]
                    boxes = roidb[i]['boxes'][non_gt_inds, :]
                else:
                    boxes = candidate_boxes[i]
                if boxes.shape[0] == 0:
                    continue

                overlaps = bbox_overlaps(boxes.astype(np.float), gt_boxes.astype(np.float))

                _gt_overlaps = np.zeros((gt_boxes.shape[0]))
                # choose whatever is smaller to iterate
                rounds = min(boxes.shape[0], gt_boxes.shape[0])
                for j in range(rounds):
                    # find which proposal maximally covers each gt box
                    argmax_overlaps = overlaps.argmax(axis=0)
                    # get the IoU amount of coverage for each gt box
                    max_overlaps = overlaps.max(axis=0)
                    # find which gt box is covered by most IoU
                    gt_ind = max_overlaps.argmax()
                    gt_ovr = max_overlaps.max()
                    assert (gt_ovr >= 0), '%s\n%s\n%s' % (boxes, gt_boxes, overlaps)
                    # find the proposal box that covers the best covered gt box
                    box_ind = argmax_overlaps[gt_ind]
                    # record the IoU coverage of this gt box
                    _gt_overlaps[j] = overlaps[box_ind, gt_ind]
                    assert (_gt_overlaps[j] == gt_ovr)
                    # mark the proposal box and the gt box as used
                    overlaps[box_ind, :] = -1
                    overlaps[:, gt_ind] = -1
                # append recorded IoU coverage level
                gt_overlaps = np.hstack((gt_overlaps, _gt_overlaps))

            gt_overlaps = np.sort(gt_overlaps)
            if thresholds is None:
                step = 0.05
                thresholds = np.arange(0.5, 0.95 + 1e-5, step)
            recalls = np.zeros_like(thresholds)

            # compute recall for each IoU threshold
            for i, t in enumerate(thresholds):
                recalls[i] = (gt_overlaps >= t).sum() / float(num_pos)
            ar = recalls.mean()

            # print results
            log_info = 'average recall for {}: {:.3f}'.format(area_name, ar)
            print log_info
            all_log_info += log_info
            for threshold, recall in zip(thresholds, recalls):
                log_info = 'recall @{:.2f}: {:.3f}'.format(threshold, recall)
                print log_info
                all_log_info += log_info

        return all_log_info

    @staticmethod
    def merge_roidbs(a, b):
        """
        merge roidbs into one
        :param a: roidb to be merged into
        :param b: roidb to be merged
        :return: merged imdb
        """
        assert len(a) == len(b)
        for i in range(len(a)):
            a[i]['boxes'] = np.vstack((a[i]['boxes'], b[i]['boxes']))
            a[i]['gt_classes'] = np.hstack((a[i]['gt_classes'], b[i]['gt_classes']))
            a[i]['gt_overlaps'] = np.vstack((a[i]['gt_overlaps'], b[i]['gt_overlaps']))
            a[i]['max_classes'] = np.hstack((a[i]['max_classes'], b[i]['max_classes']))
            a[i]['max_overlaps'] = np.hstack((a[i]['max_overlaps'], b[i]['max_overlaps']))
        return a
