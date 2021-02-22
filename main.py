import argparse
import pdb, os, sys, time

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

import config as cfg
from dataloader import get_loader
from log.logger import logger
from model import EfficientDet, EfficientDetD3
from train import train
from utils import (CosineLRScheduler, DetectionLoss, ExponentialMovingAverage)
from utils.utils import count_parameters, init_seed
from validation import validate


def parse_args():
    parser = argparse.ArgumentParser(description='Main')

    parser.add_argument('-mode', choices=['trainval', 'eval'],
                        default='trainval', type=str)
    parser.add_argument('-model', default='efficientdet-d0', type=str)
    parser.add_argument('--experiment', type=str, default='experiment')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true')
    parser.add_argument('--cuda', dest='cuda', action='store_true')
    parser.add_argument('--cpu', dest='cuda', action='store_false')
    parser.add_argument('--device', type=int, default=0)
    parser.set_defaults(cuda=False)

    arguments = parser.parse_args()
    return arguments


def build_tools(model):
    optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg.WARMUP_LR,
        weight_decay=cfg.WEIGHT_DECAY, momentum=cfg.MOMENTUM)

    schedule_helper = CosineLRScheduler(
        lr_warmup_init=cfg.WARMUP_LR, base_lr=cfg.BASE_LR,
        lr_warmup_step=cfg.STEPS_PER_EPOCH, total_steps=cfg.TOTAL_STEPS)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: schedule_helper.get_lr_coeff(step))

    criterion = DetectionLoss(
        cfg.ALPHA, cfg.GAMMA, cfg.DELTA, cfg.BOX_LOSS_WEIGHT, cfg.NUM_CLASSES)

    ema_decay = ExponentialMovingAverage(model, cfg.MOVING_AVERAGE_DECAY)

    return optimizer, scheduler, criterion, ema_decay


def setup_writer(tb_tag, args):
    writer = SummaryWriter(log_dir=cfg.LOG_PATH / tb_tag)
    writer.add_text("Hyperparams", '<br />'.join(
        [f"{k}: {v}" for k, v in args.__dict__.items()]))
    writer.train_step, writer.eval_step = 0, 0
    return writer


def main(args):
    device = torch.device('cuda:{}'.format(args.device)) \
        if args.cuda else torch.device('cpu')

    model = EfficientDet.from_pretrained(args.model).to(device) \
        if args.pretrained else EfficientDet.from_name(args.model).to(device)
    model_pnames = '\n'.join(sorted([n+': ('+', '.join([str(p.size(i))
                                                       for i in range(p.ndim)])+')'
                                    for n,p in model.named_parameters()]))
    model = EfficientDetD3.from_pretrained().to(device) \
        if args.pretrained else EfficientDetD3.from_name().to(device)
    model2_pnames = '\n'.join(sorted([n+': ('+', '.join([str(p.size(i))
                                                         for i in range(p.ndim)])+')'
                                      for n,p in model.named_parameters()]))
    print(model_pnames == model2_pnames)
    #pdb.set_trace()

    if args.mode == 'trainval':
        logger("Model's trainable parameters: {}".format(count_parameters(model)))

        loader = get_loader(path=cfg.TRAIN_SET,
                            annotations=cfg.TRAIN_ANNOTATIONS,
                            batch_size=cfg.BATCH_SIZE)

        optimizer, scheduler, criterion, ema_decay = build_tools(model)
        writer = setup_writer(args.experiment, args)
        best_score = -1

        for epoch in range(cfg.NUM_EPOCHS):
            model, optimizer, scheduler, writer = \
                train(model, optimizer, loader, scheduler,
                      criterion, ema_decay, device, writer)

            if epoch > cfg.VAL_DELAY and \
                    (epoch + 1) % cfg.VAL_INTERVAL == 0:
                ema_decay.assign(model)
                model, writer, best_score = \
                    validate(model, device, writer,
                             cfg.MODEL.SAVE_PATH, best_score=best_score)
                ema_decay.resume(model)

    elif args.mode == 'eval':
        #a,b = model(torch.randn(1,3,896,896))
        #for i in range(5):
        #    print(a[i].size(), b[i].size())
        
        import numpy as np
        import tvm
        import tvm.target
        import tvm.contrib.ndk
        from tvm import relay
        
        model = model.eval()
        input_shape = [1, 3, 896, 896]
        input_data = torch.randn(input_shape)
        scripted_model = torch.jit.trace(model, input_data).eval()
        
        input_name = "input0"
        shape_list = [(input_name, input_shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
        
        target = tvm.target.arm_cpu("aarch64")
        target_host = "llvm -mtriple=aarch64-linux-android27"
        pdb.set_trace()
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, target_host=target_host, params=params)
        
        pdb.set_trace()
        ret = lib.export_library("deploy_lib.so", tvm.contrib.ndk.create_shared)
        #with open("deploy_graph.json", "w") as fo:
        #    fo.write(graph.json())
        with open("deploy_param.params", "wb") as fo:
            fo.write(relay.save_param_dict(params))
        
        pdb.set_trace()
        validate(model, device)


if __name__ == '__main__':
    init_seed(cfg.SEED)
    main(parse_args())
