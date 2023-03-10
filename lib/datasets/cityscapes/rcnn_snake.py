import torch.utils.data as data
from lib.utils.snake import snake_cityscapes_utils, visualize_utils
from lib.utils.snake import snake_voc_utils
import cv2
import numpy as np
import math
from lib.utils import data_utils
from lib.utils.rcnn_snake import rcnn_snake_config as snake_config
import random


class Dataset(data.Dataset):
    def __init__(self, ann_file, data_root, split, istrain):
        super(Dataset, self).__init__()
        print('root', data_root, ann_file)
        print(istrain)
        self.data_root = data_root
        self.split = split

        self.anns = np.array(snake_cityscapes_utils.read_dataset(ann_file)[:])
        self.anns = self.anns[:10] if split == 'mini' else self.anns
        self.json_category_id_to_contiguous_id = snake_cityscapes_utils.JSON_DICT
        
        self.istrain = istrain
        
    def read_original_data(self, anno, path):
        img = cv2.imread(path)
        instance_polys = [np.array(obj['components']) for obj in anno]
        cls_ids = [self.json_category_id_to_contiguous_id[obj['label']] for obj in anno]
        return img, instance_polys, cls_ids

    def transform_original_data(self, instance_polys, flipped, width, trans_output, inp_out_hw):
        output_h, output_w = inp_out_hw[2:]
        instance_polys_ = []
        for instance in instance_polys:
            polys = [np.array(poly['poly']) for poly in instance]

            if flipped:
                polys_ = []
                for poly in polys:
                    poly[:, 0] = width - np.array(poly[:, 0]) - 1
                    polys_.append(poly.copy())
                polys = polys_

            polys = snake_cityscapes_utils.transform_polys(polys, trans_output, output_h, output_w)
            instance_polys_.append(polys)
        return instance_polys_

    def get_valid_polys(self, instance_polys):
        instance_polys_ = []
        for instance in instance_polys:
            instance = [poly for poly in instance if len(poly) >= 4]
            polys = snake_cityscapes_utils.filter_tiny_polys(instance)
            polys = snake_cityscapes_utils.get_cw_polys(polys)
            polys = [poly[np.sort(np.unique(poly, axis=0, return_index=True)[1])] for poly in polys]
            instance_polys_.append(polys)
        return instance_polys_

    def get_amodal_boxes(self, extreme_points):
        boxes = []
        for instance_points in extreme_points:
            if len(instance_points) == 0:
                box = []
            else:
                instance = np.concatenate(instance_points)
                box = np.concatenate([np.min(instance, axis=0), np.max(instance, axis=0)])
            boxes.append(box)
        return boxes

    def get_extreme_points(self, instance_polys):
        extreme_points = []
        for instance in instance_polys:
            points = [snake_cityscapes_utils.get_extreme_points(poly) for poly in instance]
            extreme_points.append(points)
        return extreme_points

    def prepare_adet(self, box, ct_hm, cls_id, wh, ct_ind):
        if len(box) == 0:
            return

        ct_hm = ct_hm[cls_id]

        x_min, y_min, x_max, y_max = box
        ct = np.round([(x_min + x_max) / 2, (y_min + y_max) / 2]).astype(np.int32)

        h, w = y_max - y_min, x_max - x_min
        radius = data_utils.gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        data_utils.draw_umich_gaussian(ct_hm, ct, radius)

        wh.append([w, h])
        ct_ind.append(ct[1] * ct_hm.shape[1] + ct[0])

    def prepare_rcnn(self, abox, instance, cp_hm, cp_wh, cp_ind):
        if len(abox) == 0:
            return

        x_min, y_min, x_max, y_max = abox
        ct = np.round([(x_min + x_max) / 2, (y_min + y_max) / 2]).astype(np.int32)
        h, w = y_max - y_min, x_max - x_min
        abox = np.array([ct[0] - w/2, ct[1] - h/2, ct[0] + w/2, ct[1] + h/2])

        hm = np.zeros([1, snake_config.cp_h, snake_config.cp_w], dtype=np.float32)
        abox_w, abox_h = abox[2] - abox[0], abox[3] - abox[1]
        cp_wh_ = []
        cp_ind_ = []
        ratio = [snake_config.cp_w, snake_config.cp_h] / np.array([abox_w, abox_h])

        decode_boxes = []

        for ex in instance:
            box = np.concatenate([np.min(ex, axis=0), np.max(ex, axis=0)])
            box_w, box_h = box[2] - box[0], box[3] - box[1]
            cp_wh_.append([box_w, box_h])

            center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            shift = center - abox[:2]
            ro_center = shift / [abox_w, abox_h] * [snake_config.cp_w, snake_config.cp_h]
            ro_center = np.floor(ro_center).astype(np.int32)
            cp_ind_.append(ro_center[1] * hm.shape[2] + ro_center[0])

            ro_box_w, ro_box_h = [box_w, box_h] * ratio
            radius = data_utils.gaussian_radius((math.ceil(ro_box_h), math.ceil(ro_box_w)))
            radius = max(0, int(radius))
            data_utils.draw_umich_gaussian(hm[0], ro_center, radius)

            center = ro_center / [snake_config.cp_w, snake_config.cp_h] * [abox_w, abox_h] + abox[:2]
            x_min, y_min = center[0] - box_w / 2, center[1] - box_h / 2
            x_max, y_max = center[0] + box_w / 2, center[1] + box_h / 2
            x_min, y_min = max(x_min, 0), max(y_min, 0)
            decode_boxes.append([x_min, y_min, x_max, y_max])

        cp_hm.append(hm)
        cp_wh.append(cp_wh_)
        cp_ind.append(cp_ind_)

        return decode_boxes

    def prepare_init(self, box, extreme_point, i_it_4pys, c_it_4pys, i_gt_4pys, c_gt_4pys, h, w):
        x_min, y_min = np.min(extreme_point[:, 0]), np.min(extreme_point[:, 1])
        x_max, y_max = np.max(extreme_point[:, 0]), np.max(extreme_point[:, 1])

        if np.random.uniform(0, 1) < 0.5:
            x_shift = x_min - box[0]
            y_shift = y_min - box[1]
            box = [x_min + x_shift, y_min + y_shift, x_max + x_shift, y_max + y_shift]

        img_init_poly = snake_cityscapes_utils.get_init(box)
        img_init_poly = snake_cityscapes_utils.uniformsample(img_init_poly, snake_config.init_poly_num)
        can_init_poly = snake_cityscapes_utils.img_poly_to_can_poly(img_init_poly, x_min, y_min, x_max, y_max)
        img_gt_poly = extreme_point
        can_gt_poly = snake_cityscapes_utils.img_poly_to_can_poly(img_gt_poly, x_min, y_min, x_max, y_max)

        i_it_4pys.append(img_init_poly)
        c_it_4pys.append(can_init_poly)
        i_gt_4pys.append(img_gt_poly)
        c_gt_4pys.append(can_gt_poly)

    def prepare_evolution(self, bbox, poly, extreme_point, img_init_polys, can_init_polys, img_gt_polys, can_gt_polys):
        x_min, y_min = np.min(extreme_point[:, 0]), np.min(extreme_point[:, 1])
        x_max, y_max = np.max(extreme_point[:, 0]), np.max(extreme_point[:, 1])
        
        if self.istrain:
            if random.random() > 0.33:
                aug_para = (np.random.rand(4)) * 3 - 1.5
                bbox = list(np.array(bbox) + aug_para)
                bbox = np.clip(bbox, 0, 199)
            
        extreme_point = snake_voc_utils.get_init(bbox)

        octagon = snake_cityscapes_utils.get_octagon(extreme_point)
        img_init_poly = snake_cityscapes_utils.uniformsample(octagon, snake_config.poly_num)
        can_init_poly = snake_cityscapes_utils.img_poly_to_can_poly(img_init_poly, x_min, y_min, x_max, y_max)

        img_gt_poly = snake_cityscapes_utils.uniformsample(poly, len(poly) * snake_config.gt_poly_num)
        tt_idx = np.argmin(np.power(img_gt_poly - img_init_poly[0], 2).sum(axis=1))
        img_gt_poly = np.roll(img_gt_poly, -tt_idx, axis=0)[::len(poly)]
        can_gt_poly = snake_cityscapes_utils.img_poly_to_can_poly(img_gt_poly, x_min, y_min, x_max, y_max)

        img_init_polys.append(img_init_poly)
        can_init_polys.append(can_init_poly)
        img_gt_polys.append(img_gt_poly)
        can_gt_polys.append(can_gt_poly)

    def __getitem__(self, index):
        ann = self.anns[index]

        anno, path, img_id = snake_cityscapes_utils.process_info(ann, self.data_root)
        img, instance_polys, cls_ids = self.read_original_data(anno, path)

        height, width = img.shape[0], img.shape[1]
        orig_img, inp, trans_input, trans_output, flipped, center, scale, inp_out_hw = \
            snake_cityscapes_utils.augment(
                img, self.split,
                snake_config.data_rng, snake_config.eig_val, snake_config.eig_vec,
                snake_config.mean, snake_config.std, instance_polys
            )
        instance_polys = self.transform_original_data(instance_polys, flipped, width, trans_output, inp_out_hw)
        instance_polys = self.get_valid_polys(instance_polys)
        extreme_points = self.get_extreme_points(instance_polys)
        boxes = self.get_amodal_boxes(extreme_points)

        # detection
        output_h, output_w = inp_out_hw[2:]

        act_hm = np.zeros([8, output_h, output_w], dtype=np.float32)
        awh = []
        act_ind = []

        # component
        cp_hm = []
        cp_wh = []
        cp_ind = []

        # evolution
        i_it_pys = []
        c_it_pys = []
        i_gt_pys = []
        c_gt_pys = []

        for i in range(len(anno)):
            cls_id = cls_ids[i]
            instance_poly = instance_polys[i]
            instance_points = extreme_points[i]
            self.prepare_adet(boxes[i], act_hm, cls_id, awh, act_ind)
            decode_boxes = self.prepare_rcnn(boxes[i], instance_points, cp_hm, cp_wh, cp_ind)

            for j in range(len(instance_poly)):
                poly = instance_poly[j]
                extreme_point = instance_points[j]

                x_min, y_min = np.min(poly[:, 0]), np.min(poly[:, 1])
                x_max, y_max = np.max(poly[:, 0]), np.max(poly[:, 1])
                bbox = [x_min, y_min, x_max, y_max]
                h, w = y_max - y_min + 1, x_max - x_min + 1
                if h <= 1 or w <= 1:
                    continue

                self.prepare_evolution(bbox, poly, extreme_point, i_it_pys, c_it_pys, i_gt_pys, c_gt_pys)

        # the meaning of the returned data
        # inp: image
        # act_hm: 'ct_hm' means the heatmap of the object center; 'a' means 'amodal', which includes the complete object
        # awh: 'wh' means the width and height of the object bounding box
        # act_ind: the index in an image, row * width + col
        # cp_hm: component heatmap
        # cp_ind: the index in an RoI
        # i_it_4py: initial 4-vertex polygon for extreme point prediction, 'i' means 'image', 'it' means 'initial'
        # c_it_4py: normalized initial 4-vertex polygon. 'c' means 'canonical', which indicates that the polygon coordinates are normalized.
        # i_gt_4py: ground-truth 4-vertex polygon.
        # i_it_py: initial n-vertex polygon for contour deformation.

        ret = {'inp': inp}
        adet = {'act_hm': act_hm, 'awh': awh, 'act_ind': act_ind}
        cp = {'cp_hm': cp_hm, 'cp_wh': cp_wh, 'cp_ind': cp_ind}

        evolution = {'i_it_py': i_it_pys, 'c_it_py': c_it_pys, 'i_gt_py': i_gt_pys, 'c_gt_py': c_gt_pys}
        ret.update(adet)
        ret.update(cp)
        ret.update(evolution)
        act_num = len(act_ind)
        ct_num = len(i_gt_pys)
        meta = {'center': center, 'scale': scale, 'img_id': img_id, 'ann': ann, 'act_num': act_num, 'ct_num': ct_num}
        ret.update({'meta': meta})

        return ret

    def __len__(self):
        return len(self.anns)

