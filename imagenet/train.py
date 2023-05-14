import torch
from util.torch_dist_sum import *
from data.imagenet import *
from util.meter import *
import builtins
import time
from util.accuracy import accuracy
from data.augmentation import build_transform
from util.dist_init import dist_init
from util.set_weight_decay import set_weight_decay
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import argparse
import math
import torch.nn as nn
from util.ema import ExponentialMovingAverage
import json
from network.genius import GENIUS
from timm.models.layers.activations import HardSwish
from functools import partial
from thop import profile


parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=23457)
parser.add_argument('--epochs', type=int, default=500)

# arch
parser.add_argument('--arch', type=str, nargs='+')

# Optimizer parameters
parser.add_argument("--lr", default=0.5, type=float, help="initial learning rate")
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=2e-5,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument(
    "--norm-weight-decay",
    default=0.0,
    type=float,
    help="weight decay for Normalization layers (default: None, same value as --wd)",
)

# Augmentation parameters
parser.add_argument('--train_size', type=int, default=224)
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5', metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". " + \
                            "(default: rand-m9-mstd0.5)'),
parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

# Random Erase params
parser.add_argument('--reprob', type=float, default=0.2, metavar='PCT',
                    help='Random erase prob (default: 0.2)')
parser.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "pixel")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')

# * Mixup params
parser.add_argument('--mixup', type=float, default=0.0,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
parser.add_argument('--cutmix', type=float, default=0.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

# ema params
parser.add_argument(
    "--model-ema-decay",
    type=float,
    default=0.99998,
    help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
)

parser.add_argument(
    "--model-ema-steps",
    type=int,
    default=32,
    help="the number of iterations that controls how often to update the EMA model (default: 32)",
)

parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--drop-path', type=float, default=0.0, help='Drop path rate (default: None)')

parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--warm_up', type=int, default=5)
parser.add_argument('--eval', default=False, action='store_true')
parser.add_argument('--sync_bn', default=False, action='store_true')
parser.add_argument('--checkpoint', type=str, default='')
args = parser.parse_args()

epochs = args.epochs
warm_up = args.warm_up


def adjust_learning_rate(optimizer, epoch, base_lr, i, iteration_per_epoch):
    if epoch < warm_up:
        T = epoch * iteration_per_epoch + i
        warmup_iters = warm_up * iteration_per_epoch
        lr = base_lr  * T / warmup_iters
    else:
        min_lr = 0
        T = epoch - warm_up
        total_iters = epochs - warm_up
        lr = 0.5 * (1 + math.cos(1.0 * T / total_iters * math.pi)) * (base_lr - min_lr) + min_lr
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(train_loader, model, model_ema, local_rank, optimizer, lr, epoch, criterion, mixup_fn):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    curr_lr = InstantMeter('LR', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, top1, top5, losses, curr_lr],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (samples, targets) in enumerate(train_loader):
        adjust_learning_rate(optimizer, epoch, lr, i, len(train_loader))
        # measure data loading time
        data_time.update(time.time() - end)

        samples = samples.cuda(local_rank, non_blocking=True)
        targets = targets.cuda(local_rank, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        logits = model(samples)
        loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

        acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
        top1.update(acc1[0], samples.size(0))
        top5.update(acc5[0], samples.size(0))
        losses.update(loss.item(), samples.size(0))
        curr_lr.update(optimizer.param_groups[0]['lr'])

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < warm_up:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 20 == 0:
            progress.display(i)



@torch.no_grad()
def test(test_loader, model, local_rank):
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    model.eval()
    
    for i, (img, target) in enumerate(test_loader):
        img = img.cuda(local_rank, non_blocking=True)
        target = target.cuda(local_rank, non_blocking=True)

        logits = model(img)
    
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        top1.update(acc1[0], img.size(0))
        top5.update(acc5[0], img.size(0))

    sum1, cnt1, sum5, cnt5 = torch_dist_sum(local_rank, top1.sum, top1.count, top5.sum, top5.count)
    top1_acc = sum(sum1.float()) / sum(cnt1.float())
    top5_acc = sum(sum5.float()) / sum(cnt5.float())
    return top1_acc, top5_acc


def main():
    rank, local_rank, world_size = dist_init(args.port)
    args.rank = rank
    args.local_rank = local_rank
    args.world_size = world_size
    num_workers = 6

    if rank != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    
    print(args)
    activation = partial(HardSwish, inplace=True)
    if args.arch is None:
        model = GENIUS(dropout=args.dropout, drop_path_rate=args.drop_path, act_layer=activation, num_classes=1000)
    else:
        arch = json.loads(''.join(args.arch))
        stage_list = [arch[0:6]  ,arch[6:12] ,arch[12:18],arch[18:24],arch[24:30]]
        model = GENIUS(arch=stage_list, dropout=args.dropout, drop_path_rate=args.drop_path, act_layer=activation, num_classes=1000)
    print('============ Generate Model ============')
    print(model)
    if not args.arch is None:
        print('============ Stage List ============')
        for stage_op in stage_list:
            print(stage_op)
    input = torch.randn(1, 3, args.train_size, args.train_size)
    macs, params = profile(model, inputs=(input, ))
    print('Flops: {}, Params: {}'.format(macs / (1000 ** 2), params / (1000 ** 2)))

    if args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda()

    parameters = set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=None,
    )
    
    optimizer = torch.optim.SGD(
        parameters,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    torch.backends.cudnn.benchmark = True
    train_dataset = Imagenet(mode='train', aug=build_transform(is_train=True, args=args))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=train_sampler, drop_last=True, persistent_workers=True)

    test_dataset = Imagenet(mode='val', aug=build_transform(is_train=False, args=args))
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=(test_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=test_sampler, persistent_workers=True)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
    alpha = 1.0 - args.model_ema_decay
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(model.module, device='cuda', decay=1.0 - alpha)

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    if mixup_active:
        criterion = SoftTargetCrossEntropy().cuda()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing).cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss().cuda()


    start_epoch = 0
    best_online_top1 = 0
    best_online_top5 = 0
    best_ema_top1 = 0
    best_ema_top5 = 0

    if not os.path.exists('checkpoint') and rank == 0:
        os.makedirs('checkpoint')

    checkpoint_path = 'checkpoint/{}.pth.tar'.format(args.checkpoint)
    best_checkpoint_path = 'checkpoint/{}_best.pth.tar'.format(args.checkpoint)
    print('checkpoint_path:', checkpoint_path)
    if os.path.exists(checkpoint_path):
        checkpoint =  torch.load(checkpoint_path, map_location='cpu')

        if args.eval:
            model.module.load_state_dict(checkpoint, strict=False)
        else:
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            model_ema.load_state_dict(checkpoint['model_ema'])
            try:
                best_online_top1 = checkpoint['best_online_top1'].item()
                best_online_top5 = checkpoint['best_online_top5'].item()
                best_ema_top1 = checkpoint['best_ema_top1'].item()
                best_ema_top5 = checkpoint['best_ema_top5'].item()
            except:
                best_online_top1 = checkpoint['best_online_top1']
                best_online_top5 = checkpoint['best_online_top5']
                best_ema_top1 = checkpoint['best_ema_top1']
                best_ema_top5 = checkpoint['best_ema_top5']
            start_epoch = checkpoint['epoch']
    
    if args.eval:
        top1, top5 = test(test_loader, model, local_rank)
        print('* Acc@1 {:.3f} Acc@5 {:.3f}'.format(top1, top5))
    else:
        for epoch in range(start_epoch, epochs):
            train_loader.sampler.set_epoch(epoch)
            train(train_loader, model, model_ema, local_rank, optimizer, args.lr, epoch, criterion, mixup_fn)
            online_top1, online_top5 = test(test_loader, model, local_rank)
            ema_top1, ema_top5 = test(test_loader, model_ema, local_rank)
            best_online_top1 = max(best_online_top1, online_top1)
            best_online_top5 = max(best_online_top5, online_top5)
            best_ema_top1 = max(best_ema_top1, ema_top1)
            best_ema_top5 = max(best_ema_top5, ema_top5)
            print('Online - Epoch:{} * Acc@1 {:.3f} Acc@5 {:.3f} Best_Acc@1 {:.3f} Best_Acc@5 {:.3f}'.format(epoch, online_top1, online_top5, best_online_top1, best_online_top5))
            print('EMA    - Epoch:{} * Acc@1 {:.3f} Acc@5 {:.3f} Best_Acc@1 {:.3f} Best_Acc@5 {:.3f}'.format(epoch, ema_top1, ema_top5, best_ema_top1, best_ema_top5))

            if rank == 0:
                state_dict =  {
                    'model': model.state_dict(),
                    'model_ema': model_ema.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_online_top1': best_online_top1,
                    'best_online_top5': best_online_top5,
                    'best_ema_top1': best_ema_top1,
                    'best_ema_top5': best_ema_top5,
                    'epoch': epoch + 1
                }
                torch.save(state_dict, checkpoint_path)

                if best_ema_top1 == ema_top1:
                    state_dict =  {
                        'model': model.state_dict(),
                        'model_ema': model_ema.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_online_top1': best_online_top1,
                        'best_online_top5': best_online_top5,
                        'best_ema_top1': best_ema_top1,
                        'best_ema_top5': best_ema_top5,
                        'epoch': epoch + 1
                    }
                    torch.save(state_dict, best_checkpoint_path)


if __name__ == "__main__":
    main()


