import torch
from lib.utils.snake import snake_config
from lib.utils.snake.snake_decode import nms, topk, transpose_and_gather_feat, get_octagon, get_init
from lib.utils.rcnn_snake import rcnn_snake_config
from lib.csrc.extreme_utils import _ext as extreme_utils


def box_to_roi(box, box_01):
    """ box: [b, n, 4] """
    box = box[box_01]
    ind = torch.cat([torch.full([box_01[i].sum()], i) for i in range(len(box_01))], dim=0)
    ind = ind.to(box.device).float()
    roi = torch.cat([ind[:, None], box], dim=1)
    return roi


def decode_cp_detection(cp_hm, cp_wh, abox, adet):
    batch, cat, height, width = cp_hm.size()
    if rcnn_snake_config.cp_hm_nms:
        cp_hm = nms(cp_hm)

    abox_w, abox_h = abox[..., 2] - abox[..., 0], abox[..., 3] - abox[..., 1]

    scores, inds, clses, ys, xs = topk(cp_hm, rcnn_snake_config.max_cp_det)
    cp_wh = transpose_and_gather_feat(cp_wh, inds)
    cp_wh = cp_wh.view(batch, rcnn_snake_config.max_cp_det, 2)

    cp_hm_h, cp_hm_w = cp_hm.size(2), cp_hm.size(3)

    xs = xs / cp_hm_w * abox_w[..., None] + abox[:, 0:1]
    ys = ys / cp_hm_h * abox_h[..., None] + abox[:, 1:2]
    boxes = torch.stack([xs - cp_wh[..., 0] / 2,
                         ys - cp_wh[..., 1] / 2,
                         xs + cp_wh[..., 0] / 2,
                         ys + cp_wh[..., 1] / 2], dim=2)

    ascore = adet[..., 4]
    acls = adet[..., 5]
    excluded_clses = [1, 2]
    for cls_ in excluded_clses:
        boxes[acls == cls_, 0] = abox[acls == cls_]
        scores[acls == cls_, 0] = 1
        scores[acls == cls_, 1:] = 0

    ct_num = len(abox)
    boxes_ = []
    for i in range(ct_num):
        cp_ind = extreme_utils.nms(boxes[i], scores[i], rcnn_snake_config.max_cp_overlap)
        cp_01 = scores[i][cp_ind] > rcnn_snake_config.cp_score
        boxes_.append(boxes[i][cp_ind][cp_01])

    cp_ind = torch.cat([torch.full([len(boxes_[i])], i) for i in range(len(boxes_))], dim=0)
    cp_ind = cp_ind.to(boxes.device)
    boxes = torch.cat(boxes_, dim=0)

    return boxes, cp_ind


def decode_ct_hm(ct_hm, wh, reg=None, K=100):
    batch, cat, height, width = ct_hm.size()
    ct_hm = nms(ct_hm)

    scores, inds, clses, ys, xs = topk(ct_hm, K=K)
    wh = transpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, 2)

    if reg is not None:
        reg = transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1)
        ys = ys.view(batch, K, 1)

    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    ct = torch.cat([xs, ys], dim=2)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2], dim=2)
    detection = torch.cat([bboxes, scores, clses], dim=2)

    return ct, detection


def collect_training(poly, ct_01):
    batch_size = ct_01.size(0)
    poly = torch.cat([poly[i][ct_01[i]] for i in range(batch_size)], dim=0)
    return poly


def uniform_upsample(poly, p_num):
    # 1. assign point number for each edge
    # 2. calculate the coefficient for linear interpolation
    next_poly = torch.roll(poly, -1, 2)
    edge_len = (next_poly - poly).pow(2).sum(3).sqrt()
    edge_num = torch.round(edge_len * p_num / torch.sum(edge_len, dim=2)[..., None]).long()
    edge_num = torch.clamp(edge_num, min=1)
    edge_num_sum = torch.sum(edge_num, dim=2)
    edge_idx_sort = torch.argsort(edge_num, dim=2, descending=True)
    extreme_utils.calculate_edge_num(edge_num, edge_num_sum, edge_idx_sort, p_num)
    edge_num_sum = torch.sum(edge_num, dim=2)
    assert torch.all(edge_num_sum == p_num)

    edge_start_idx = torch.cumsum(edge_num, dim=2) - edge_num
    weight, ind = extreme_utils.calculate_wnp(edge_num, edge_start_idx, p_num)
    poly1 = poly.gather(2, ind[..., 0:1].expand(ind.size(0), ind.size(1), ind.size(2), 2))
    poly2 = poly.gather(2, ind[..., 1:2].expand(ind.size(0), ind.size(1), ind.size(2), 2))
    poly = poly1 * (1 - weight) + poly2 * weight

    return poly
    

def prepare_training_box(ret, batch, init):
    box = ret['detection'][..., :4]
    score = ret['detection'][..., 4]
    batch_size = box.size(0)
    i_gt_4py = batch['i_gt_4py']
    ct_01 = batch['ct_01'].byte()
    ind = [get_box_match_ind(box[i], score[i], i_gt_4py[i][ct_01[i]]) for i in range(batch_size)]
    box_ind = [ind_[0] for ind_ in ind]
    gt_ind = [ind_[1] for ind_ in ind]

    i_it_4py = torch.cat([get_init(box[i][box_ind[i]][None]) for i in range(batch_size)], dim=1)
    if i_it_4py.size(1) == 0:
        return

    i_it_4py = uniform_upsample(i_it_4py, snake_config.init_poly_num)[0]
    c_it_4py = img_poly_to_can_poly(i_it_4py)
    i_gt_4py = torch.cat([batch['i_gt_4py'][i][gt_ind[i]] for i in range(batch_size)], dim=0)
    c_gt_4py = torch.cat([batch['c_gt_4py'][i][gt_ind[i]] for i in range(batch_size)], dim=0)
    init_4py = {'i_it_4py': i_it_4py, 'c_it_4py': c_it_4py, 'i_gt_4py': i_gt_4py, 'c_gt_4py': c_gt_4py}

    i_it_py = get_octagon(i_gt_4py[None])
    i_it_py = uniform_upsample(i_it_py, snake_config.poly_num)[0]
    c_it_py = img_poly_to_can_poly(i_it_py)
    i_gt_py = torch.cat([batch['i_gt_py'][i][gt_ind[i]] for i in range(batch_size)], dim=0)
    init_py = {'i_it_py': i_it_py, 'c_it_py': c_it_py, 'i_gt_py': i_gt_py}

    ind = torch.cat([torch.full([len(gt_ind[i])], i) for i in range(batch_size)], dim=0)

    if snake_config.train_pred_box_only:
        for k, v in init_4py.items():
            init[k] = v
        for k, v in init_py.items():
            init[k] = v
        init['4py_ind'] = ind
        init['py_ind'] = ind
    else:
        init.update({k: torch.cat([init[k], v], dim=0) for k, v in init_4py.items()})
        init.update({'4py_ind': torch.cat([init['4py_ind'], ind], dim=0)})
        init.update({k: torch.cat([init[k], v], dim=0) for k, v in init_py.items()})
        init.update({'py_ind': torch.cat([init['py_ind'], ind], dim=0)})
        

def prepare_training(ret, batch):
    ct_01 = batch['ct_01'].byte()
    init = {}

    init.update({'i_it_py': collect_training(batch['i_it_py'], ct_01)})
    init.update({'c_it_py': collect_training(batch['c_it_py'], ct_01)})
    init.update({'i_gt_py': collect_training(batch['i_gt_py'], ct_01)})
    init.update({'c_gt_py': collect_training(batch['c_gt_py'], ct_01)})

    ct_num = batch['meta']['ct_num']
    init.update({'4py_ind': torch.cat([torch.full([ct_num[i]], i) for i in range(ct_01.size(0))], dim=0)})
    init.update({'py_ind': init['4py_ind']})

    if snake_config.train_pred_box:
        prepare_training_box(ret, batch, init)

    init['4py_ind'] = init['4py_ind'].to(ct_01.device)
    init['py_ind'] = init['py_ind'].to(ct_01.device)

    return init
