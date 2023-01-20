import torch.nn as nn
from lib.utils import net_utils
import torch


class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net

        self.m_crit = net_utils.FocalLoss()
        self.ct_crit = net_utils.FocalLoss()
        self.py_crit = torch.nn.functional.smooth_l1_loss
    
    def shape_loss(self, pred, targ_shape):
        pre_dis = torch.cat((pred[:,1:], pred[:,0].unsqueeze(1)), dim=1)
        pred_shape = pre_dis-pred
        # targ_shape = targ(:,:,1:)-targ(:,:,:-1)
        loss = self.py_crit(pred_shape, targ_shape)
        return loss

    def forward(self, batch):
        output = self.net(batch['inp'], batch)

        scalar_stats = {}
        loss = 0

        mask_loss = self.m_crit(net_utils.sigmoid(output['mask']), batch['cmask'])
        scalar_stats.update({'mask_loss': mask_loss})
        loss += mask_loss
        
        ct_loss = self.ct_crit(net_utils.sigmoid(output['ct_hm']), batch['ct_hm'])
        scalar_stats.update({'ct_loss': ct_loss})
        loss += ct_loss

        wh_loss = self.py_crit(output['poly_init'], output['i_gt_py'])
        scalar_stats.update({'wh_loss': 0.1 * wh_loss})
        loss += 0.1 * wh_loss

        n_predictions = len(output['py_pred'])
        py_loss = 0.0
        shape_loss = 0.0
        py_dis = torch.cat((output['i_gt_py'][:,1:], output['i_gt_py'][:,0].unsqueeze(1)), dim=1)
        tar_shape = py_dis - output['i_gt_py']
        for i in range(n_predictions):
            i_weight = 0.8**(n_predictions - i - 1)
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
