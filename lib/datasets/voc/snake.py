import os
from lib.utils.snake import snake_voc_utils, snake_config, visualize_utils
import cv2
import numpy as np
import math
from lib.utils import data_utils
import torch.utils.data as data
from pycocotools.coco import COCO
from lib.config import cfg
import random 

class Dataset(data.Dataset):
    def __init__(self, ann_file, data_root, split, istrain):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.split = split

        self.coco = COCO(ann_file)
        self.anns = np.array(sorted(self.coco.getImgIds()))
        self.anns = self.anns[:500] if split == 'mini' else self.anns
        self.json_category_id_to_contiguous_id = {v: i for i, v in enumerate(self.coco.getCatIds())}

    def process_info(self, img_id):
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anno = self.coco.loadAnns(ann_ids)
        path = os.path.join(self.data_root, self.coco.loadImgs(int(img_id))[0]['file_name'])
        return anno, path, img_id

    def read_original_data(self, anno, path):
        img = cv2.imread(path)
        instance_polys = [[np.array(poly).reshape(-1, 2) for poly in obj['segmentation']] for obj in anno]
        cls_ids = [self.json_category_id_to_contiguous_id[obj['category_id']] for obj in anno]
        return img, instance_polys, cls_ids

    def transform_original_data(self, instance_polys, flipped, width, trans_output, inp_out_hw):
        output_h, output_w = inp_out_hw[2:]
        instance_polys_ = []
        for instance in instance_polys:
            polys = [poly.reshape(-1, 2) for poly in instance]

            if flipped:
                polys_ = []
                for poly in polys:
                    poly[:, 0] = width - np.array(poly[:, 0]) - 1
                    polys_.append(poly.copy())
                polys = polys_

            polys = snake_voc_utils.transform_polys(polys, trans_output, output_h, output_w)
            instance_polys_.append(polys)
        return instance_polys_

    def get_valid_polys(self, instance_polys, inp_out_hw):
        output_h, output_w = inp_out_hw[2:]
        instance_polys_ = []
        for instance in instance_polys:
            instance = [poly for poly in instance if len(poly) >= 4]
            for poly in instance:
                poly[:, 0] = np.clip(poly[:, 0], 0, output_w - 1)
                poly[:, 1] = np.clip(poly[:, 1], 0, output_h - 1)
            polys = snake_voc_utils.filter_tiny_polys(instance)
            polys = snake_voc_utils.get_cw_polys(polys)
            polys = [poly[np.sort(np.unique(poly, axis=0, return_index=True)[1])] for poly in polys]
            instance_polys_.append(polys)
        return instance_polys_

    def get_extreme_points(self, instance_polys):
        extreme_points = []
        for instance in instance_polys:
            points = [snake_voc_utils.get_extreme_points(poly) for poly in instance]
            extreme_points.append(points)
        return extreme_points

    def prepare_detection(self, box, poly, ct_hm, cls_id, ct_cls, ct_ind):
        ct_hm = ct_hm[cls_id]
        ct_cls.append(cls_id)

        x_min, y_min, x_max, y_max = box
        ct = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2], dtype=np.float32)
        ct = np.round(ct).astype(np.int32)

        h, w = y_max - y_min, x_max - x_min
        radius = data_utils.gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        data_utils.draw_umich_gaussian(ct_hm, ct, radius)

        ct_ind.append(ct[1] * ct_hm.shape[1] + ct[0])

    def prepare_evolution(self, poly, img_gt_polys):
        img_gt_poly = snake_voc_utils.uniformsample(poly, len(poly) * 128)
        idx = self.four_idx(img_gt_poly)
        img_gt_poly = self.get_img_gt(img_gt_poly, idx)
        img_gt_polys.append(img_gt_poly)

    def four_idx(self, img_gt_poly):
        x_min, y_min = np.min(img_gt_poly, axis=0)
        x_max, y_max = np.max(img_gt_poly, axis=0)
        center = [(x_min + x_max) / 2., (y_min + y_max) / 2.]
        can_gt_polys = img_gt_poly.copy()
        can_gt_polys[:, 0] -= center[0]
        can_gt_polys[:, 1] -= center[1]
        distance = np.sum(can_gt_polys ** 2, axis=1, keepdims=True) ** 0.5 + 1e-6
        can_gt_polys /= np.repeat(distance, axis=1, repeats=2)
        idx_bottom = np.argmax(can_gt_polys[:, 1])
        idx_top = np.argmin(can_gt_polys[:, 1])
        idx_right = np.argmax(can_gt_polys[:, 0])
        idx_left = np.argmin(can_gt_polys[:, 0])
        return [idx_bottom, idx_right, idx_top, idx_left]

    def get_img_gt(self, img_gt_poly, idx, t=128):
        align = len(idx)
        pointsNum = img_gt_poly.shape[0]
        r = []
        k = np.arange(0, t / align, dtype=float) / (t / align)
        for i in range(align):
            begin = idx[i]
            end = idx[(i + 1) % align]
            if begin > end:
                end += pointsNum
            r.append((np.round(((end - begin) * k).astype(int)) + begin) % pointsNum)
        r = np.concatenate(r, axis=0)
        return img_gt_poly[r, :]

    def img_poly_to_can_poly(self, img_poly):
        x_min, y_min = np.min(img_poly, axis=0)
        can_poly = img_poly - np.array([x_min, y_min])
        return can_poly

    def __getitem__(self, index):
        ann = self.anns[index]

        anno, path, img_id = self.process_info(ann)
        img, instance_polys, cls_ids = self.read_original_data(anno, path)

        height, width = img.shape[0], img.shape[1]
        orig_img, inp, trans_input, trans_output, flipped, center, scale, inp_out_hw = \
            snake_voc_utils.augment(
                img, self.split,
                snake_config.data_rng, snake_config.eig_val, snake_config.eig_vec,
                snake_config.mean, snake_config.std, instance_polys
            )
        instance_polys = self.transform_original_data(instance_polys, flipped, width, trans_output, inp_out_hw)
        instance_polys = self.get_valid_polys(instance_polys, inp_out_hw)

        # detection
        output_h, output_w = inp_out_hw[2:]
        ct_hm = np.zeros([cfg.heads.ct_hm, output_h, output_w], dtype=np.float32)
        ct_cls = []
        ct_ind = []

        # evolution
        i_gt_pys = []

        cmask = snake_voc_utils.polygon_to_cmask(instance_polys, output_h, output_w)[np.newaxis,:,:]

        for i in range(len(anno)):
            cls_id = cls_ids[i]
            instance_poly = instance_polys[i]

            for j in range(len(instance_poly)):
                poly = instance_poly[j]

                x_min, y_min = np.min(poly[:, 0]), np.min(poly[:, 1])
                x_max, y_max = np.max(poly[:, 0]), np.max(poly[:, 1])
                bbox = [x_min, y_min, x_max, y_max]
                h, w = y_max - y_min + 1, x_max - x_min + 1
                if h <= 1 or w <= 1:
                    continue

                self.prepare_detection(bbox, poly, ct_hm, cls_id, ct_cls, ct_ind)
                self.prepare_evolution(poly, i_gt_pys)

        ret = {'inp': inp, 'cmask': cmask}
        detection = {'ct_hm': ct_hm, 'ct_cls': ct_cls, 'ct_ind': ct_ind}
        evolution = {'i_gt_py': i_gt_pys}
        ret.update(detection)
        ret.update(evolution)

        ct_num = len(ct_ind)
        meta = {'center': center, 'scale': scale, 'img_id': img_id, 'ann': ann, 'ct_num': ct_num}
        ret.update({'meta': meta})

        return ret

    def __len__(self):
        return len(self.anns)

