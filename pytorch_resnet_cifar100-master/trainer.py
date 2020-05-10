import argparse
import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
#import torchvision.datasets as datasets
import resnet
import pickle as pkl
import datasets

model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))

#print(model_names)
# resnet152:
# -1: 72.910, 0: 71.14,
#
select_model = "resnet34"
mode = "test"


#CUDA_VISIBLE_DEVICES=0 python trainer.py

evaluate = False if mode=='train' else True

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('-e', '--evaluate', default=evaluate, help='evaluate model on validation set')
parser.add_argument('--cfg_sub_set', default=-1, type=int, help='use pre-trained model')
parser.add_argument('--pretrained', default='./checkpoints/' + select_model + '/', type=str, help='use pre-trained model')

parser.add_argument('--arch', '-a', metavar='ARCH', default=select_model,
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet32)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', default="./checkpoints/" + select_model,
                    help='The directory used to save the trained models', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=5)
best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    cfg_sub_set = args.cfg_sub_set


    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
    model.cuda()

    # load a checkpoint to resume training or validate
    if args.resume is not None or (args.evaluate and args.pretrained is not None):
        if args.pretrained is not None:
            all_pretrained_checkpoints = os.listdir(args.pretrained)
            #checkpoint_path = [pretrained_checkpoint for pretrained_checkpoint in all_pretrained_checkpoints if pretrained_checkpoint.startswith(args.arch)][0]
            #checkpoint_path = max(all_pretrained_checkpoints)
            checkpoint_path = select_model + "_val_"+ str(cfg_sub_set) + "_epoch_195.th"
            checkpoint_path = args.pretrained + checkpoint_path
        else:
            checkpoint_path = args.resume
        if os.path.isfile(checkpoint_path):
            print("=> loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)
            if args.resume is not None:
                args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}'".format(checkpoint_path))
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint_path))
    cudnn.benchmark = True
    CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    normalize = transforms.Normalize(mean=CIFAR100_TRAIN_MEAN, std=CIFAR100_TRAIN_STD)

    # 45000 images, 313 batches, 100 classes
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=False, sub_set=cfg_sub_set, val=False,),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)


    # 5000/10000 images, 79 batches, 100 classes.
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]), sub_set=cfg_sub_set, val=True,),
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    # 10000 images, 79 batches, 100 classes.
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.CIFAR100(root='./data', train=False, transform=transforms.Compose([
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=128, shuffle=False,
    #     num_workers=args.workers, pin_memory=True)


    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.half:
        model.half()
        criterion.half()

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2, last_epoch=args.start_epoch - 1)

    if args.arch in ['resnet1202', 'resnet110']:
        # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this setup it will correspond for first epoch.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr*0.1


    if args.evaluate:
        validate(val_loader, model, criterion, enable_save_pred=True)
        return

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        # print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        if epoch > 0 and (epoch + 1) % args.save_every == 0:
            prec1 = validate(val_loader, model, criterion, enable_save_pred=(epoch + 1) == args.epochs)
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, select_model + '_val_' + str(val_loader.dataset.sub_set) + '_epoch_'+ str(epoch + 1) +'.th'))

            if epoch + 1 == args.epochs:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                }, is_best, filename=os.path.join(args.save_dir, select_model + '_val_' + str(val_loader.dataset.sub_set) + '_epoch_'+ str(epoch + 1) +'.th'))
                print("TRAINING FINISHED")
                return


def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion, enable_save_pred=False):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    saved_pred = torch.Tensor(size=(len(val_loader.dataset), 103)).cuda()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            if enable_save_pred:
                save_predictions(output.data, target, saved_pred, batch_num=i, total_batch=len(val_loader)-1, sub_set=val_loader.dataset.sub_set)
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    print('Test: [{0}/{1}]\t'
          'Time {batch_time.avg:.3f}\t'
          'Loss {loss.avg:.4f}\t'
          'Prec@1 {top1.avg:.3f}'.format(len(val_loader), len(val_loader), batch_time=batch_time, loss=losses, top1=top1))

    #print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

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


def save_predictions(output, target, saved_pred, batch_num, total_batch, sub_set):
    '''
        output: [batch_size, num_classes]
        target: [batch_size]
        saved_pred: [10000/50000, batch_size, 103], 103 -> [1 (correct pred), 1 (seq_num), 1 (target), 10 (pred)]
    '''
    index_range = [batch_num * 128, (batch_num + 1) * 128]
    start_seq_num=0
    if sub_set != -1:
        start_seq_num = sub_set * saved_pred.shape[0]
    seq_num_range = [batch_num * 128 + start_seq_num, (batch_num+1) * 128 + start_seq_num]
    if seq_num_range[1] > saved_pred.shape[0] + start_seq_num:
        seq_num_range[1] = saved_pred.shape[0] + start_seq_num
        index_range[1] = saved_pred.shape[0]

    index_range = torch.arange(*index_range).view(-1, 1).cuda()

    sample_seq_num = torch.arange(*seq_num_range).view(-1, 1).cuda()
    pred_target = output.max(dim=1)[1]
    correctness = (target == pred_target).float().view(-1, 1)


    saved_pred[index_range.squeeze()] = torch.cat([sample_seq_num.float(), correctness, target.view(-1, 1).float(), output], dim=1)  # [128, 103]

    if batch_num == total_batch:
        pred_save_path = "./pred_pkl/" + select_model + "_val_" + str(sub_set) +'_pred.pkl'
        with open(pred_save_path, 'wb') as f:
            pkl.dump(saved_pred, f)
        print("{} saved.".format(pred_save_path))

def concat_predictions(pkl_dir="./pred_pkl/"):
    file_names = os.listdir(pkl_dir)
    cat_data = {}

    for file_name in file_names:
        file_path = os.path.join(pkl_dir, file_name)
        with open(file_path, 'rb') as f:
            print(file_path)
            pred_data = pkl.load(f).cpu().numpy()
            f.close()

        model_name = file_name.split('_')[0]
        if file_name.split('_')[2] != '-1':
            if model_name in cat_data.keys():
                cat_data[model_name] = np.concatenate([cat_data[model_name], pred_data], axis=0)
            else:
                cat_data[model_name] = pred_data

    for key in cat_data.keys():
        pred_data = cat_data[key]
        cat_data_path = './cat_pred_pkl/' + key + '_cat_pred_data.pkl'
        with open(cat_data_path, 'wb') as f:
            pkl.dump(pred_data, f)
            print(cat_data_path+" is dumped!")
            f.close()


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
    #concat_predictions()
