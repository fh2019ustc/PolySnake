from .snake import Snake
from .update import BasicUpdateBlock
from lib.utils import data_utils
from lib.utils.snake import snake_gcn_utils, snake_config, snake_decode

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RAFT(nn.Module):
    def __init__(self):
        super(RAFT, self).__init__()
        self.iter = 6  # iteration number

        self.evolve_gcn = Snake(state_dim=128, feature_dim=64 + 2, conv_type='dgrid', need_fea=True)
        self.update_block = BasicUpdateBlock()

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def prepare_training(self, output, batch):
        init = snake_gcn_utils.prepare_training(output, batch)
        output.update({'i_gt_py': init['i_gt_py']})
        return init

    def evolve_poly(self, snake, cnn_feature, i_it_poly, c_it_poly, ind):
        if len(i_it_poly) == 0:
            return torch.zeros_like(i_it_poly)
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        init_feature = snake_gcn_utils.get_gcn_feature(cnn_feature, i_it_poly, ind, h, w)
        init_input = torch.cat([init_feature, c_it_poly.permute(0, 2, 1)], dim=1)
        i_poly_fea = snake(init_input)
        return i_poly_fea

    def use_gt_detection(self, output, batch):
        bacthsize, _, height, width = output['ct_hm'].size()
        wh_pred = output['wh']
        ct_01 = batch['ct_01'].byte()
        ct_ind = batch['ct_ind'][ct_01]
        ct_img_idx = batch['ct_img_idx'][ct_01]
        ct_x, ct_y = ct_ind % width, ct_ind // width
        ct_img_idx = ct_img_idx % bacthsize

        if ct_x.size(0) == 0:
            ct_offset = wh_pred[ct_img_idx, :, ct_y, ct_x].view(ct_x.size(0), 1,  2)  
        else:
            ct_offset = wh_pred[ct_img_idx, :, ct_y, ct_x].view(ct_x.size(0), -1, 2)

        ct_x, ct_y = ct_x[:, None].float(), ct_y[:, None].float()
        ct = torch.cat([ct_x, ct_y], dim=1)

        init_polys = ct_offset + ct.unsqueeze(1).expand(ct_offset.size(0), ct_offset.size(1), ct_offset.size(2))

        output.update({'poly_init': init_polys * snake_config.ro})
        return init_polys

    def clip_to_image(self, poly, h, w):
        poly[..., :2] = torch.clamp(poly[..., :2], min=0)
        poly[..., 0] = torch.clamp(poly[..., 0], max=w - 1)
        poly[..., 1] = torch.clamp(poly[..., 1], max=h - 1)
        return poly

    def decode_detection(self, output, h, w):
        ct_hm = output['ct_hm']
        wh = output['wh']

        poly_init, detection = snake_decode.decode_ct_hm(torch.sigmoid(ct_hm), wh)

        valid = detection[0, :, 2] >= 0.05  # min_ct_score
        poly_init, detection = poly_init[0][valid], detection[0][valid]

        init_polys = self.clip_to_image(poly_init, h, w)
        output.update({'poly_init_infer': init_polys * snake_config.ro, 'detection': detection})
        return poly_init, detection

    def forward(self, output, cnn_feature, batch):
        ret = output
        if batch is not None and 'test' not in batch['meta']:
            with torch.no_grad():
                init = self.prepare_training(output, batch)

            poly_init = self.use_gt_detection(output, batch)
            poly_init = poly_init.detach()
            py_pred = poly_init * snake_config.ro
            c_py_pred = snake_gcn_utils.img_poly_to_can_poly(poly_init)
            i_poly_fea = self.evolve_poly(self.evolve_gcn, cnn_feature, poly_init, c_py_pred, init['py_ind'])  # n*c*128
            net = torch.tanh(i_poly_fea)
            i_poly_fea = F.leaky_relu(i_poly_fea)
            py_preds = []
            for i in range(self.iter):
                net, offset = self.update_block(net, i_poly_fea)
                py_pred = py_pred + snake_config.ro * offset
                py_preds.append(py_pred)

                py_pred_sm = py_pred / snake_config.ro
                c_py_pred = snake_gcn_utils.img_poly_to_can_poly(py_pred_sm)
                i_poly_fea = self.evolve_poly(self.evolve_gcn, cnn_feature, py_pred_sm, c_py_pred, init['py_ind'])
                i_poly_fea = F.leaky_relu(i_poly_fea)
            ret.update({'py_pred': py_preds, 'i_gt_py': output['i_gt_py'] * snake_config.ro})

        if not self.training:
            with torch.no_grad():
                poly_init, detection = self.decode_detection(output, cnn_feature.size(2), cnn_feature.size(3))

                ind = torch.zeros((poly_init.size(0)))

                py_pred = poly_init * snake_config.ro
                c_py_pred = snake_gcn_utils.img_poly_to_can_poly(poly_init)
                i_poly_fea = self.evolve_poly(self.evolve_gcn, cnn_feature, poly_init, c_py_pred,
                                              ind)

                if len(py_pred) != 0:
                    net = torch.tanh(i_poly_fea)
                    i_poly_fea = F.leaky_relu(i_poly_fea)
                    for i in range(self.iter):
                        net, offset = self.update_block(net, i_poly_fea)
                        py_pred = py_pred + snake_config.ro * offset
                        py_pred_sm = py_pred / snake_config.ro

                        if i != (self.iter - 1):                     
                            c_py_pred = snake_gcn_utils.img_poly_to_can_poly(py_pred_sm)
                            i_poly_fea = self.evolve_poly(self.evolve_gcn, cnn_feature, py_pred_sm, c_py_pred, ind) #init['ind'])
                            i_poly_fea = F.leaky_relu(i_poly_fea)
                    py_preds = [py_pred_sm]
                else:
                    py_preds = [i_poly_fea]    
                ret.update({'py': py_preds})
        return output

