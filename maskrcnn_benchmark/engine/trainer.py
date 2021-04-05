# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import os
import time

import torch
import torch.distributed as dist
from tqdm import tqdm

from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.utils.comm import get_world_size, synchronize
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.engine.inference import inference

from apex import amp
import numpy as np
from torch.autograd import Variable

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    cfg,
    model,
    source_data_loader,
    target_data_loader,
    da_enable,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(source_data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()

    data_iter = iter(source_data_loader)
    tgt_data_iter = iter(target_data_loader)


    for iteration in range(max_iter):
        images, targets, semgt_raw = next(data_iter)
        t_images, _, _ = next(tgt_data_iter)
        epoch_period = cfg.SOLVER.EPOCH_PERIOD
        if any(len(target) < 1 for target in targets):
            logger.error(f"Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}" )
            continue
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        s_images = images.to(device)
        s_targets = [target.to(device) for target in targets]
        s_semgt = Variable(semgt_raw[0].long().cuda(0))

        if da_enable:
            epoch_cur = int((iteration) // epoch_period) + 1
            max_epoch = max_iter / epoch_period
            train_ratio = float(epoch_cur) / max_epoch
            grl_alpha = 2. / (1. + np.exp(-10 * train_ratio)) - 1
            t_images = t_images.to(device)
        else:
            t_images = None
            grl_alpha = None

        loss_dict = model(s_images, t_images, s_targets, grl_alpha, s_semgt)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        with amp.scale_loss(losses, optimizer) as scaled_losses:
            scaled_losses.backward()
        optimizer.step()
        scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )

        if (iteration + 1) % epoch_period == 0 and iteration > 0:
            epoch = int((iteration + 1) / epoch_period)
            checkpointer.save("model_epoch_{:03d}".format(epoch), **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
