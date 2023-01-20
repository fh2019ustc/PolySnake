from .snake import Snake
from .update import BasicUpdateBlock
from lib.utils.snake import snake_gcn_utils, snake_config, snake_decode, active_spline
from lib.utils.rcnn_snake import rcnn_snake_utils

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RAFT(nn.Module):
    def __init__(self):
        super(RAFT, self).__init__()
        self.iter = 6

        self.evolve_gcn = Snake(state_dim=128, feature_dim=64+2, conv_type='dgrid', need_fea=True)
        self.update_block = BasicUpdateBlock()

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def prepare_training(self, output, batch):
        init = rcnn_snake_utils.prepare_training(output, batch)
        output.update({'i_it_py': init['i_it_py']})
        output.update({'i_gt_py': init['i_gt_py']})
        return init

    def get_quadrangle(self, box):
        x_min, y_min, x_max, y_max = box[:,0], box[:,1], box[:,2], box[:,3] 
        a = torch.stack(((x_min + x_max) / 2., y_min), dim=-1)
        b = torch.stack((x_min, (y_min + y_max) / 2.), dim=-1)
        c = torch.stack(((x_min + x_max) / 2., y_max), dim=-1)
        d = torch.stack((x_max, (y_min + y_max) / 2.), dim=-1)
        qua = torch.stack((a,b,c,d), dim=1)

        return qua

    def prepare_testing_init(self, output):
        i_it_4py = snake_decode.get_init(output['cp_box'][None])
        ind = output['roi_ind'][output['cp_ind'].long()]
        init = {'ind': ind}
        output.update({'qua': i_it_4py[0]})

        return init

    def prepare_testing_evolve(self, output, h, w):
        ex = output['qua']
        ex[..., 0] = torch.clamp(ex[..., 0], min=0, max=w-1)
        ex[..., 1] = torch.clamp(ex[..., 1], min=0, max=h-1)
        evolve = snake_gcn_utils.prepare_testing_evolve(ex)

        output.update({'it_py': evolve['i_it_py']})
        return evolve

    def evolve_poly(self, snake, cnn_feature, i_it_poly, c_it_poly, ind):
        if len(i_it_poly) == 0:
            return torch.zeros_like(i_it_poly)
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        init_feature = snake_gcn_utils.get_gcn_feature(cnn_feature, i_it_poly, ind, h, w)
        init_input = torch.cat([init_feature, c_it_poly.permute(0, 2, 1)], dim=1)
        i_poly_fea = snake(init_input)
        return i_poly_fea

    def forward(self, output, cnn_feature, batch):
        ret = output
        if batch is not None and 'test' not in batch['meta']:
            with torch.no_grad():
                init = self.prepare_training(output, batch)
                
            py_pred = init['i_it_py'] * snake_config.ro
            i_poly_fea = self.evolve_poly(self.evolve_gcn, cnn_feature, init['i_it_py'], init['c_it_py'], init['py_ind'])
            net = torch.tanh(i_poly_fea)
            i_poly_fea = F.leaky_relu(i_poly_fea)
            py_preds = []
            for i in range(self.iter):
                net, offset = self.update_block(net, i_poly_fea)
                py_pred = py_pred + offset * snake_config.ro
                py_preds.append(py_pred)

                if i != (self.iter - 1):
                    py_pred_sm = py_pred / snake_config.ro
                    c_py_pred = snake_gcn_utils.img_poly_to_can_poly(py_pred_sm)
                    i_poly_fea = self.evolve_poly(self.evolve_gcn, cnn_feature, py_pred_sm, c_py_pred, init['py_ind'])
                    i_poly_fea = F.leaky_relu(i_poly_fea)
            ret.update({'py_pred': py_preds, 'i_gt_py': output['i_gt_py'] * snake_config.ro})
            
        if not self.training:
            with torch.no_grad():
                init = self.prepare_testing_init(output)

                evolve = self.prepare_testing_evolve(output, cnn_feature.size(2), cnn_feature.size(3))
                py_pred = evolve['i_it_py'] * snake_config.ro
                i_poly_fea = self.evolve_poly(self.evolve_gcn, cnn_feature, evolve['i_it_py'], evolve['c_it_py'], init['ind'])
                
                if len(evolve['i_it_py']) != 0:
                    net = torch.tanh(i_poly_fea)
                    i_poly_fea = F.leaky_relu(i_poly_fea)
                    for i in range(self.iter):
                        net, offset = self.update_block(net, i_poly_fea)
                        py_pred = py_pred + offset * snake_config.ro
                        py_pred_sm = py_pred / snake_config.ro
                        
                        if i != (self.iter - 1):  
                            c_py_pred = snake_gcn_utils.img_poly_to_can_poly(py_pred_sm)
                            i_poly_fea = self.evolve_poly(self.evolve_gcn, cnn_feature, py_pred_sm, c_py_pred, init['ind'])
                            i_poly_fea = F.leaky_relu(i_poly_fea)
                    py_preds = [py_pred_sm]
                else:
                    py_preds = [i_poly_fea]
                ret.update({'py': py_preds})
        return output

