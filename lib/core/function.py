# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import torch

from core.evaluate import accuracy


# reduce loss like in maskrcnn, however i am not sure about this function. It is not utilize yet.
def reduce_loss(loss):
  world_size = torch.distributed.get_world_size()
  if world_size < 2:
    return loss
  with torch.no_grad():
    reduce_loss = loss
    torch.distributed.reduce(reduce_loss, dst = 0)
    if torch.distributed.get_rank() == 0:
      reduce_loss /= world_size
  return reduce_loss


def analysis_grad(model):
    res = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            res[name]={}
            res[name]['shape']=param.grad.shape
            res[name]['mean'], res[name]['std'], res[name]['mean_abs'], res[name]['max_abs'], res[name][' f_norm'], res[name]['1_norm'] = \
                      param.grad.mean().item(), param.grad.std().item(), param.grad.abs().std().item(), param.grad.abs().max().item(), \
                      torch.norm(param.grad).item(), torch.norm(param.grad, 1).item()
            '''print(' --- mean --- ', param.grad.mean())
            print(' --- std  --- ', param.grad.std())
            print(' --- mean_abs  --- ', param.grad.abs().std())
            print(' --- max_abs   --- ', param.grad.abs().max())
            print(' --- f norm  --- ', torch.norm(param.grad))
            print(' --- 1 norm   --- ', torch.norm(param.grad, 1))'''
    return res


def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict=None, to_output=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if to_output:
        logger = logging.getLogger(__name__)
        if torch.distributed.is_initialized():
            print("Output at this rank", torch.distributed.get_rank())

    # switch to train mode
    model.train()
    
    # tmp: to analysis grad
    grad_dict = {}

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        #target = target - 1 # Specific for imagenet

        # compute output
        input = input.cuda()
        output = model(input)
        target = target.cuda(non_blocking=True)

        loss = criterion(output, target)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        if i % 1 == 0:
            grad_dict[i] = analysis_grad(model.module)
            if i % 200 == 0:
                torch.save(grad_dict, "grad_dict.pth")
                print('Saved at', i)
        optimizer.step()
        prec1, prec5 = accuracy(output, target, (1, 5))

        # measure accuracy and record loss
        if torch.distributed.is_initialized():
            loss = reduce_loss(loss)
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0 and to_output:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy@1 {top1.val:.3f} ({top1.avg:.3f})\t' \
                  'Accuracy@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, top1=top1, top5=top5)
            print(msg)
            logger.info(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer.add_scalar('train_top1', top1.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1


def validate(config, val_loader, model, criterion, output_dir, tb_log_dir,
             writer_dict=None, to_output=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if to_output:
        logger = logging.getLogger(__name__)
        
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            # compute output
            output = model(input)

            target = target.cuda(non_blocking=True)

            loss = criterion(output, target)

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            prec1, prec5 = accuracy(output, target, (1, 5))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        
        if torch.distributed.is_initialized():
            results = torch.cuda.FloatTensor([top1.sum, top5.sum, top1.count])
            
            torch.distributed.reduce(results, dst = 0)
            top1_avg, top5_avg = results[0] / results[2], results[1] / results[2]
        else:
            top1_avg, top5_avg = top1.avg, top5.avg

        if to_output:
            msg = 'Test: Time {batch_time.avg:.3f}\t' \
                  'Loss {loss.avg:.4f}\t' \
                  'Error@1 {error1:.3f}\t' \
                  'Error@5 {error5:.3f}\t' \
                  'Accuracy@1 {top1_avg:.3f}\t' \
                  'Accuracy@5 {top5_avg:.3f}\t'.format(
                      batch_time=batch_time, loss=losses, top1_avg=top1_avg, top5_avg=top5_avg,
                      error1=100-top1_avg, error5=100-top5_avg)
            logger.info(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['valid_global_steps']
                writer.add_scalar('valid_loss', losses.avg, global_steps)
                writer.add_scalar('valid_top1', top1_avg, global_steps)
                writer_dict['valid_global_steps'] = global_steps + 1

    return top1_avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
