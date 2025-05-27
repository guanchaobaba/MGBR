#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import random
from torch.utils.data import DataLoader
import setproctitle
import dataset
from model import MGBR, MGBR_Info
from utils import check_overfitting, early_stop, logger, create_homogeneous_graph, structure
from train import train
from metric import Recall, NDCG
from config import CONFIG
from test import test
import loss
from itertools import product
import time
from tensorboardX import SummaryWriter



def main():
    #  set env
    setproctitle.setproctitle(f"train{CONFIG['name']}")
    #os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG['gpu_id']
    device = torch.device('cuda:1')

    #  fix seed
    seed = 123
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True



    #  load data
    bundle_train_data, bundle_test_data, item_data, assist_data = \
            dataset.get_dataset(CONFIG['path'], CONFIG['dataset_name'], task=CONFIG['task'])

    train_loader = DataLoader(bundle_train_data, 2048, True,
                              num_workers=8, pin_memory=True)

    test_loader = DataLoader(bundle_test_data, 2048, False,
                             num_workers=8, pin_memory=True)

    #  pretrain
    if 'pretrain' in CONFIG:
        pretrain = torch.load(CONFIG['pretrain'], map_location='cpu')
        print('load pretrain')

    #  graph
    ub_graph = bundle_train_data.ground_truth_u_b
    ui_graph = item_data.ground_truth_u_i
    bi_graph = assist_data.ground_truth_b_i

    #  metric
    metrics = [Recall(20), NDCG(20), Recall(40), NDCG(40), Recall(80), NDCG(80)]
    TARGET = 'Recall@20'

    #  loss
    loss_func = loss.BPRLoss('mean')

    #  log
    log = logger.Logger(os.path.join(
        CONFIG['log'], CONFIG['dataset_name'], 
        f"{CONFIG['model']}_{CONFIG['task']}", ''), 'best', checkpoint_target=TARGET)

    theta = 0.6

    time_path = time.strftime("%y%m%d-%H%M%S", time.localtime(time.time()))

    num_users = assist_data.num_users
    num_bundles = assist_data.num_bundles
    num_items = assist_data.num_items

    ui_hom_graph = create_homogeneous_graph(ui_graph, num_users, num_items, device)
    ub_hom_graph = create_homogeneous_graph(ub_graph, num_users, num_bundles, device)
    bi_hom_graph = create_homogeneous_graph(bi_graph, num_bundles, num_items, device)
    
    gprompt = structure(ub_hom_graph,ui_hom_graph,bi_hom_graph, device)

    for lr, decay, message_dropout, node_dropout \
            in product(CONFIG['lrs'], CONFIG['decays'], CONFIG['message_dropouts'], CONFIG['node_dropouts']):

        visual_path = os.path.join(CONFIG['visual'],
                                    CONFIG['dataset_name'],  
                                    f"{CONFIG['model']}_{CONFIG['task']}", 
                                    f"{time_path}@{CONFIG['note']}", 
                                    f"lr{lr}_decay{decay}_medr{message_dropout}_nodr{node_dropout}")

        # model
        if CONFIG['model'] == 'MGBR':
            graph = [ub_graph, ui_graph, bi_graph]
            info = MGBR_Info(64, decay, message_dropout, node_dropout, 2)
            model = MGBR(info, assist_data, graph, device,gprompt, pretrain=None).to(device)

        assert model.__class__.__name__ == CONFIG['model']

        # op
        op = optim.RMSprop(model.parameters(), lr=lr)

        # 初始化调度器，监控验证表现中的指标
        scheduler = ReduceLROnPlateau(op, mode='max', factor=0.95, patience=5, verbose=True)

        # env
        env = {'lr': lr,
               'op': str(op).split(' ')[0],   
               'dataset': CONFIG['dataset_name'],
               'model': CONFIG['model'], 
               'sample': CONFIG['sample'],
               }

        retry = CONFIG['retry']  # =1
        while retry >= 0:
            # log
            log.update_modelinfo(info, env, metrics)
            # train & test
            early = CONFIG['early']
            train_writer = SummaryWriter(log_dir=visual_path, comment='train')
            test_writer = SummaryWriter(log_dir=visual_path, comment='test')
            for epoch in range(CONFIG['epochs']):
                # train
                trainloss = train(model, epoch+1, train_loader, op, device, CONFIG, loss_func)
                train_writer.add_scalars('loss/single', {"loss": trainloss}, epoch)


                # test
                if epoch % CONFIG['test_interval'] == 0:
                    output_metrics = test(model, test_loader, device, CONFIG, metrics)

                    # 选择Target指标进行记录
                    ndcg_20 = next((m.metric for m in output_metrics if m.get_title() == "NDCG@20"), None)
                    #testloss = None

                    for metric in output_metrics:
                        test_writer.add_scalars('metric/all', {metric.get_title(): metric.metric}, epoch)
                        if metric==output_metrics[0]:
                            test_writer.add_scalars('metric/single', {metric.get_title(): metric.metric}, epoch)

                    # log
                    log.update_log(metrics, model)

                    # 更新学习率调度器
                    if ndcg_20 is not None:
                        scheduler.step(ndcg_20)
                    #if testloss is not None:
                    #    scheduler.step(testloss)

                    # check overfitting
                    if epoch > 10:
                        if check_overfitting(log.metrics_log, TARGET, 1, show=False):
                            break
                    # early stop
                    early = early_stop(
                        log.metrics_log[TARGET], early, threshold=0)
                    if early <= 0:
                        break
            train_writer.close()
            test_writer.close()

            log.close_log(TARGET)
            retry = -1

    log.close()


if __name__ == "__main__":
    main()
