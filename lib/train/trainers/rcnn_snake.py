import torch.nn as nn
from lib.utils import net_utils
import torch


class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net

        self.act_crit = net_utils.FocalLoss()
        self.awh_crit = net_utils.IndL1Loss1d('smooth_l1')
        self.cp_crit = net_utils.FocalLoss()
        self.cp_wh_crit = net_utils.IndL1Loss1d('smooth_l1')
        self.ex_crit = torch.nn.functional.smooth_l1_loss
        self.py_crit = torch.nn.functional.smooth_l1_loss
        
    def shape_loss(self, pred, targ_shape):
        pre_dis = torch.cat((pred[:,1:], pred[:,0].unsqueeze(1)), dim=1)
        pred_shape = pre_dis - pred
        loss = self.py_crit(pred_shape, targ_shape)
        return loss
        
    def forward(self, batch):
        output = self.net(batch['inp'], batch)

        scalar_stats = {}
        loss = 0

        act_loss = self.act_crit(net_utils.sigmoid(output['act_hm']), batch['act_hm'])
        scalar_stats.update({'act_loss': act_loss})
        loss += act_loss

        awh_loss = self.awh_crit(output['awh'], batch['awh'], batch['act_ind'], batch['act_01'])
        scalar_stats.update({'awh_loss': awh_loss})
        loss += 0.1 * awh_loss

        act_01 = batch['act_01'].byte()

        cp_loss = self.cp_crit(net_utils.sigmoid(output['cp_hm']), batch['cp_hm'][act_01])
        scalar_stats.update({'cp_loss': cp_loss})
        loss += cp_loss
        
        cp_wh, cp_ind, cp_01 = [batch[k][act_01] for k in ['cp_wh', 'cp_ind', 'cp_01']]
        cp_wh_loss = self.cp_wh_crit(output['cp_wh'], cp_wh, cp_ind, cp_01)
        scalar_stats.update({'cp_wh_loss': cp_wh_loss})
        loss += 0.1 * cp_wh_loss

        n_predictions = len(output['py_pred'])
        py_loss = 0.0
        shape_loss = 0.0
        py_dis = torch.cat((output['i_gt_py'][:,1:], output['i_gt_py'][:,0].unsqueeze(1)), dim=1)
        tar_shape = py_dis - output['i_gt_py']
        for i in range(n_predictions):
            i_weight = 0.85**(n_predictions - i - 1)
            py_loss += i_weight * self.py_crit(output['py_pred'][i], output['i_gt_py'])
            shape_loss += i_weight * self.shape_loss(output['py_pred'][i], tar_shape)
            
        py_loss = py_loss / n_predictions
        shape_loss = shape_loss / n_predictions
        scalar_stats.update({'py_loss': py_loss})
        scalar_stats.update({'shape_loss': shape_loss})
        loss += py_loss
        loss += shape_loss

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return output, loss, scalar_stats, image_stats
