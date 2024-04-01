from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import pprint
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, '/home/l/data_2/wmz/3-b/DepNet_ANet_Release')
from  lib.models.loss_w import weakly_supervised_loss
from  lib.models.text_rec import weakly_supervised_loss_text
import torch.optim as optim
from tqdm import tqdm
from lib.core.eval import  eval_predictions, display_results
from lib import datasets
from lib import models
from lib.core.config import config, update_config
from lib.core.engine import Engine
from lib.core.utils import AverageMeter
from lib.core import eval
from lib.core.utils import create_logger
import lib.models.loss as loss
import math
import matplotlib.pyplot as plt
import pickle
from IPython import embed
from  lib.models.loss import bce_rescale_loss
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.autograd.set_detect_anomaly(True)
def parse_args():
    parser = argparse.ArgumentParser(description='Train localization network')

    # general
    parser.add_argument('--cfg', help='experiment configure file name', default='/home/l/data_2/wmz/3-b/DepNet_ANet_Release/experiments/dense_tacos/tacos.yaml',required=False, type=str)
    args, rest = parser.parse_known_args()

    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--gpus', help='gpus', type=str)
    parser.add_argument('--workers', help='num of dataloader workers', type=int)
    parser.add_argument('--dataDir', help='data path', type=str)
    parser.add_argument('--modelDir', help='model path', type=str)
    parser.add_argument('--logDir', help='log path', type=str)
    parser.add_argument('--verbose', default=False, action="store_true", help='print progress bar')
    parser.add_argument('--tag', help='tags shown in log', type=str)
    args = parser.parse_args()

    return args

def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers
    if args.dataDir:
        config.DATA_DIR = args.dataDir
    if args.modelDir:
        config.MODEL_DIR = args.modelDir
    if args.logDir:
        config.LOG_DIR = args.logDir
    if args.verbose:
        config.VERBOSE = args.verbose
    if args.tag:
        config.TAG = args.tag





if __name__ == '__main__':

    args = parse_args()
    reset_config(config, args)

    logger, final_output_dir = create_logger(config, args.cfg, config.TAG)
    logger.info('\n'+pprint.pformat(args))
    logger.info('\n'+pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    dataset_name = config.DATASET.NAME
    model_name = config.MODEL.NAME

    test_dataset = getattr(datasets, dataset_name)('test')

    model = getattr(models, model_name)()

    model_checkpoint = torch.load(config.MODEL.CHECKPOINT)
    model.load_state_dict(model_checkpoint,strict=True)


    device = ("cuda" if torch.cuda.is_available() else "cpu" )
    model = model.to(device)
    model.eval()

    dataloader = DataLoader(test_dataset,
                            batch_size=config.TEST.BATCH_SIZE,
                            shuffle=False,
                            num_workers=config.WORKERS,
                            pin_memory=False,
                            collate_fn=datasets.dense_collate_fn)


    def network(sample):
        # identical as single
        # anno_idxs:(b,) list
        # visual_input: (b,256,500) tensor

        # different due to dense
        # textual_input: (b,K,seq,300) tensor
        # textual_mask: (b,K,seq,1) tensor
        # sentence_mask: (b,K,1) tensor
        # map_gt: (b,K,1,64,64) tensor

        anno_idxs = sample['batch_anno_idxs']
        textual_input = sample['batch_word_vectors'].cuda()
        textual_mask = sample['batch_txt_mask'].cuda()
        sentence_mask = sample['batch_sentence_mask'].cuda()  # new
        visual_input = sample['batch_vis_input'].cuda()
        map_gt = sample['batch_map_gt'].cuda()
        # torch.Size([4, 8, 1, 32, 32])
        duration = sample['batch_duration']
        weights_list = sample['batch_weights_list']
        ids_list = sample['batch_ids_list']
        ids_list = ids_list.squeeze().int()
        # print(type(ids_list))
        # print(weights_list.shape)torch.Size([4, 8, 24, 1])
        # print(textual_mask.shape)torch.Size([4, 8, 24, 1])

        prediction, map_mask, sims, logit_scale, jj, weight_3, words_logit, ids_list, weights, words_mask1, overlaps_tensor_f, p_values_tensor_f = model(textual_input, textual_mask, sentence_mask, visual_input, duration, weights_list, ids_list)
        # print(map_gt.shape)

        # end - ----------------------------------
        # torch.Size([16, 3, 2])
        # torch.Size([16, 3])
        # torch.Size([48, 24, 40000])
        # torch.Size([48, 24])
        # torch.Size([48, 24])
        rewards = torch.from_numpy(np.asarray([0, 0.5, 1.0])).cuda()
        # loss_value1 =bce_rescale_loss(prediction, map_mask, sentence_mask,overlaps_tensor_z.unsqueeze(2),config.LOSS.PARAMS)

        loss_value2 = bce_rescale_loss(prediction, map_mask, sentence_mask, overlaps_tensor_f.unsqueeze(2),
                                       config.LOSS.PARAMS)

        # loss_value1, loss_overlap, loss_order, joint_prob = getattr(loss, config.LOSS.NAME)(prediction, map_mask, sentence_mask, overlaps_tensor.unsqueeze(2),config.LOSS.PARAMS)
        joint_prob = torch.sigmoid(prediction) * map_mask


        map_gt_plt = map_gt.cpu().detach()
        # 绘制前三个32x32分数图
        fig, axs = plt.subplots(1, 8, figsize=(15, 5))

        for i in range(8):
            # 选择相应的分数图：第i个32x32的分数图
            score_map =map_gt_plt [0, i, 0, :, :]

            ax = axs[i]
            im = ax.imshow(score_map, cmap='viridis')
            ax.axis('off')  # 不显示坐标轴
            ax.set_title(f'Score Map {i + 1}')

        # 为子图添加颜色条
        fig.colorbar(im, ax=axs, orientation='horizontal', fraction=0.02, pad=0.04)

        plt.tight_layout()
        plt.show()






        # loss_w = weakly_supervised_loss(**output, rewards=rewards)
        loss_w = weakly_supervised_loss(weight_3, words_logit, ids_list, words_mask1, rewards, sentence_mask)

        loss_t = weakly_supervised_loss_text(words_logit, ids_list, words_mask1)

        # loss_clip = loss.clip_loss(sims,logit_scale)
        # print(loss_value1)
        # print(loss_value2)
        # loss_value = 0.1 * (loss_w + loss_t)

        loss_clip = loss.clip_loss(sims, logit_scale)
        # print(loss_value1)
        # print(loss_value2)
        loss_value = loss_value2 + 0.1 * (loss_w + loss_t) + loss_clip

        sorted_times = None if model.training else get_proposal_results(joint_prob, duration)
        # sorted_times = get_proposal_results(joint_prob, duration)
        # 4 3 4 2
        # print(sorted_times)
        # if 'clip_loss' in config.LOSS.NAME:
        # loss_value += loss_clip
        # loss_value += loss_w

        # print(loss_value)

        return loss_value, sorted_times


    def get_proposal_results(scores, durations):
        # assume all valid scores are larger than one
        out_sorted_times = []
        for score_sent, duration in zip(scores, durations):
            sent_times = []
            for score in score_sent:
                if score.sum() < 1e-3:
                    break
                T = score.shape[-1]
                sorted_indexs = np.dstack(
                    np.unravel_index(np.argsort(score.cpu().detach().numpy().ravel())[::-1], (T, T))).tolist()
                sorted_indexs = np.array([item for item in sorted_indexs[0] if item[0] <= item[1]]).astype(float)

                sorted_indexs[:, 1] = sorted_indexs[:, 1] + 1
                sorted_indexs = torch.from_numpy(sorted_indexs).cuda()
                target_size = config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE
                sent_times.append((sorted_indexs.float() / target_size * duration).tolist())
            out_sorted_times.append(sent_times)
        return out_sorted_times



    def on_test_start(state):
        state['loss_meter'] = AverageMeter()
        state['sorted_segments_list'] = []
        if config.VERBOSE:
            if state['split'] == 'test':
                state['progress_bar'] = tqdm(total=math.ceil(len(test_dataset)/config.TEST.BATCH_SIZE))
            else:
                raise NotImplementedError

    def on_test_forward(state):
        if config.VERBOSE:
            state['progress_bar'].update(1)
        state['loss_meter'].update(state['loss'].item(), 1)

        min_idx = min(state['sample']['batch_anno_idxs'])
        batch_indexs = [idx - min_idx for idx in state['sample']['batch_anno_idxs']]
        sorted_segments = [state['output'][i] for i in batch_indexs]
        state['sorted_segments_list'].extend(sorted_segments)

    def on_test_end(state):
        annotations = state['iterator'].dataset.annotations
        state['Rank@N,mIoU@M'], state['miou'] = eval.eval_predictions(state['sorted_segments_list'], annotations, verbose=False)
        test_table=display_results(state['Rank@N,mIoU@M'], state['miou'],'performance on test set with \n'+config.MODEL.CHECKPOINT)
        test_message='\n'+test_table
        print(test_message)
        logger.info(test_message)
        if config.VERBOSE:
            state['progress_bar'].close()

    engine = Engine()
    engine.hooks['on_test_start'] = on_test_start
    engine.hooks['on_test_forward'] = on_test_forward
    engine.hooks['on_test_end'] = on_test_end
    engine.test(network, dataloader,split='test')