# import torch
# import torch.nn.functional as F
# from IPython import embed
# import torch.nn as nn
#
# def combined_loss(scores, masks, sentence_masks, targets, similarity_scores, cfg):
#     """
#     计算结合了原始损失和clip损失的总损失。
#
#     参数:
#     scores: 模型的输出分数
#     masks: 掩码
#     sentence_masks: 句子掩码
#     targets: 目标值
#     similarity_scores: 特征之间的相似度分数
#     cfg: 配置参数
#     clip_loss_weight: clip损失的权重
#
#     返回:
#     total_loss: 总损失
#     loss_overlap: 重叠损失
#     loss_order: 顺序损失
#     joint_prob: 联合概率
#     clip_loss_value: clip损失
#     """
#     clip_loss_weight = 1.0
#     # 原始损失函数
#     bce_loss_value, loss_overlap, loss_order, joint_prob = bce_rescale_loss(scores, masks, sentence_masks, targets, similarity_scores,cfg)
#
#     # 计算clip损失
#     clip_loss_value = clip_loss(similarity_scores)
#
#     # 将两个损失结合
#     total_loss = bce_loss_value + clip_loss_weight * clip_loss_value
#     print(bce_loss_value)
#     print(clip_loss_value)
#     return total_loss, loss_overlap, loss_order, joint_prob
#
#
#
#
#
#
#
# def bce_rescale_loss(scores, masks, sentence_masks, targets,similarity_scores, cfg):
#     # sentence_masks [b,set,1]
#     sentence_masks = sentence_masks[:,:,0] #[b,set]
#     min_iou, max_iou, bias = cfg.MIN_IOU, cfg.MAX_IOU, cfg.BIAS
#     beta, gamma = cfg.BETA, cfg.GAMMA
#     # joint_prob,scores,masks [b, sent, 1, 32, 32]
#     joint_prob = torch.sigmoid(scores) * masks
#     # joint_prob[0, 0, 0, :5,:5]
#     # joint_prob[0, -1, 0, :5,:5]
#     # print(joint_prob.shape)torch.Size([4, 8, 1, 32, 32])
#     start_prob = joint_prob.max(-1).values
#     start_prob = F.softmax(start_prob*beta, dim=-1)
#     # [b,sent,1,32]
#     #start_prob[0, 0, 0]
#
#     N_clip = joint_prob.size(-1)
#     start_time = torch.arange(0, N_clip).float()/float(N_clip) # [b,]
#     start_time = start_time.repeat((start_prob.size(0), start_prob.size(1), 1, 1)).cuda()
#     # [b,sent,1,32]
#     #start_time[0, 0, 0]
#
#     expect_start = start_prob * start_time # [b,sent,1,32]
#     expect_start = expect_start.sum(-1) # [b,sent,1]
#     # epect_start[0, :, 0], epect_start[1, :, 0]
#
#     loss_order = 0.0
#     tot_sent = 0
#     for i in range(sentence_masks.size(0)):
#         current_sentence_mask = sentence_masks[i]
#         num_sent =  current_sentence_mask.sum().item()
#         tot_sent += num_sent
#
#         current_start = expect_start[i,:,0]-1
#
#         diff = current_start[1:] - current_start[:-1]
#         diff_mask = current_sentence_mask[1:]
#         current_loss_instance = F.relu(-diff) * diff_mask
#
#         loss_order += current_loss_instance.sum()
#     loss_order = loss_order/tot_sent
#
#     target_prob = (targets-min_iou)*(1-bias)/(max_iou-min_iou)
#     target_prob[target_prob > 0] += bias
#     target_prob[target_prob > 1] = 1
#     target_prob[target_prob < 0] = 0
#     loss = F.binary_cross_entropy(joint_prob, target_prob, reduction='none') * masks
#     loss_overlap = torch.sum(loss) / torch.sum(masks)
#
#     loss_value = loss_overlap + gamma * loss_order
#     return loss_value, loss_overlap, loss_order, joint_prob,
#
#
# def clip_loss(similarity_scores):
#     """
#     计算归一化后的clip损失。
#
#     参数:
#     similarity_scores: Tensor, 形状为 [bz, bz]，包含所有特征对之间的原始相似度分数
#
#     返回:
#     loss: 计算得到的归一化后的clip损失
#     """
#     # 将相似度分数归一化到0-1之间
#     normalized_scores = torch.sigmoid(similarity_scores)
#
#     # 获取批次大小
#     bz = similarity_scores.size(0)
#
#     # 创建一个对角矩阵，对角线上的元素为1，其他元素为0
#     identity = torch.eye(bz).to(similarity_scores.device)
#
#     # 计算对角线上的损失（最大化对角线上的值）
#     diagonal_loss = F.binary_cross_entropy(normalized_scores, identity)
#
#     return diagonal_loss

import torch
import torch.nn.functional as F
from IPython import embed

def bce_rescale_loss(scores, masks, sentence_masks, targets, cfg):
    # sentence_masks [b,set,1]
    sentence_masks = sentence_masks[:,:,0] #[b,set]
    min_iou, max_iou, bias = cfg.MIN_IOU, cfg.MAX_IOU, cfg.BIAS
    beta, gamma = cfg.BETA, cfg.GAMMA
    # joint_prob,scores,masks [b, sent, 1, 32, 32]
    joint_prob = torch.sigmoid(scores) * masks
    # joint_prob[0, 0, 0, :5,:5]
    # joint_prob[0, -1, 0, :5,:5]

    start_prob = joint_prob.max(-1).values
    start_prob = F.softmax(start_prob*beta, dim=-1)
    # [b,sent,1,32]
    #start_prob[0, 0, 0]

    N_clip = joint_prob.size(-1)
    start_time = torch.arange(0, N_clip).float()/float(N_clip) # [b,]
    start_time = start_time.repeat((start_prob.size(0), start_prob.size(1), 1, 1)).cuda()
    # [b,sent,1,32]
    #start_time[0, 0, 0]

    expect_start = start_prob * start_time # [b,sent,1,32]
    expect_start = expect_start.sum(-1) # [b,sent,1]
    # epect_start[0, :, 0], epect_start[1, :, 0]

    loss_order = 0.0
    tot_sent = 0
    for i in range(sentence_masks.size(0)):
        current_sentence_mask = sentence_masks[i]
        num_sent =  current_sentence_mask.sum().item()
        tot_sent += num_sent

        current_start = expect_start[i,:,0]-1

        diff = current_start[1:] - current_start[:-1]
        diff_mask = current_sentence_mask[1:]
        current_loss_instance = F.relu(-diff) * diff_mask

        loss_order += current_loss_instance.sum()
    loss_order = loss_order/tot_sent

    target_prob = (targets-min_iou)*(1-bias)/(max_iou-min_iou)
    target_prob[target_prob > 0] += bias
    target_prob[target_prob > 1] = 1
    target_prob[target_prob < 0] = 0
    # print(joint_prob.shape)
    # print(target_prob.shape)
    # torch.Size([4, 8, 1, 64, 64])
    # torch.Size([4, 8, 1, 32, 32])
    loss = F.binary_cross_entropy(joint_prob.float().cuda(), target_prob.float().cuda(), reduction='none') * masks
    loss_overlap = torch.sum(loss) / torch.sum(masks)

    loss_value = loss_overlap + gamma * loss_order
    return loss_value



# def clip_loss(similarity_scores):
#     """
#     计算归一化后的clip损失。
#
#     参数:
#     similarity_scores: Tensor, 形状为 [bz, bz]，包含所有特征对之间的原始相似度分数
#
#     返回:
#     loss: 计算得到的归一化后的clip损失
#     """
#     # 将相似度分数归一化到0-1之间
#     normalized_scores = torch.sigmoid(similarity_scores)
#
#     # 获取批次大小
#     bz = similarity_scores.size(0)
#
#     # 创建一个对角矩阵，对角线上的元素为1，其他元素为0
#     identity = torch.eye(bz).to(similarity_scores.device)
#
#     # 计算对角线上的损失（最大化对角线上的值）
#     diagonal_loss = F.binary_cross_entropy(normalized_scores, identity)
#
#     return diagonal_loss

def clip_loss(sims, logit_scale):
    """
    Inputs: cosine similarities
        sims: n x n (text is dim-0)
        logit_scale: 1 x 1
    """
    logit_scale = logit_scale.exp()
    logits = sims * logit_scale

    t2v_log_sm = F.log_softmax(logits, dim=1)
    t2v_neg_ce = torch.diag(t2v_log_sm)
    t2v_loss = -t2v_neg_ce.mean()

    v2t_log_sm = F.log_softmax(logits, dim=0)
    v2t_neg_ce = torch.diag(v2t_log_sm)
    v2t_loss = -v2t_neg_ce.mean()
    tt = (t2v_loss + v2t_loss) / 2.0
    return tt