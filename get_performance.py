import argparse
import json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--benchmark', type=str, required=True, choices=[
                    'nas-macro', 'channel-res', 'channel-mob', '201-cifar10', '201-cifar100', '201-imagenet'])
parser.add_argument('--arch', type=str, required=True)
args = parser.parse_args()

if args.benchmark == 'nas-macro':
    benchmark_file = open('benchmark/nas-bench-macro_cifar10.json')
    data = json.load(benchmark_file)
    keys = list(data.keys())
    rank = np.array([data[k]['mean_acc'] for k in keys]).argsort().argsort()
    for k, r in zip(keys, rank):
        data[k]['rank'] = (3 ** 8) - r
    performance = {
        'rank': data[args.arch]['rank'],
        'val_acc': data[args.arch]['mean_acc'],
    }

elif args.benchmark == 'channel-res':
    benchmark_file = open('benchmark/Results_ResNet.json')
    data = json.load(benchmark_file)
    keys = list(data.keys())
    rank = np.array([data[k]['mean'] for k in keys]).argsort().argsort()
    for k, r in zip(keys, rank):
        data[k]['rank'] = (4 ** 7) - r
    base_channels = [64, 64, 64, 128, 128, 128, 128]
    channels = [int(c) for c in args.arch.split(', ')]
    operation_id_list_str = ''.join(str(int(c / bc)) for bc, c in zip(base_channels, channels))
    performance = {
        'rank': data[operation_id_list_str]['rank'],
        'val_acc': data[operation_id_list_str]['mean'],
    }

elif args.benchmark == 'channel-mob':
    benchmark_file = open('benchmark/Results_MobileNet.json')
    data = json.load(benchmark_file)
    keys = list(data.keys())
    rank = np.array([data[k]['mean'] for k in keys]).argsort().argsort()
    for k, r in zip(keys, rank):
        data[k]['rank'] = (4 ** 7) - r
    base_channels = [32, 192, 192, 192, 64, 384, 256]
    channels = [int(c) for c in args.arch.split(', ')]
    operation_id_list_str = ''.join(str(int(c / bc)) for bc, c in zip(base_channels, channels))
    performance = {
        'rank': data[operation_id_list_str]['rank'],
        'val_acc': data[operation_id_list_str]['mean'],
    }

elif args.benchmark == '201-cifar10':
    benchmark_file = open('benchmark/nasbench201_cifar10.json')  
    data = json.load(benchmark_file)
    operation_id_list = [int(opid) for opid in list(args.arch)]
    struct_dict = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
    operation_id_list_str = '|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|'.format(*[struct_dict[operation_id] for operation_id in operation_id_list])
    performance = {
        'rank' : data[operation_id_list_str]['rank'],
        'val_acc'  : data[operation_id_list_str]['val_acc_200'],
        'test_acc' : data[operation_id_list_str]['test_acc_200'],
    }

elif args.benchmark == '201-cifar100':
    benchmark_file = open('benchmark/nasbench201_cifar100.json')  
    data = json.load(benchmark_file)
    operation_id_list = [int(opid) for opid in list(args.arch)]
    struct_dict = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
    operation_id_list_str = '|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|'.format(*[struct_dict[operation_id] for operation_id in operation_id_list])
    performance = {
        'rank' : data[operation_id_list_str]['rank'],
        'val_acc'  : data[operation_id_list_str]['val_acc_200'],
        'test_acc' : data[operation_id_list_str]['test_acc_200'],
    }
elif args.benchmark == '201-imagenet':
    benchmark_file = open('benchmark/nasbench201_imagenet.json')  
    data = json.load(benchmark_file)
    operation_id_list = [int(opid) for opid in list(args.arch)]
    struct_dict = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
    operation_id_list_str = '|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|'.format(*[struct_dict[operation_id] for operation_id in operation_id_list])
    performance = {
        'rank' : data[operation_id_list_str]['rank'],
        'val_acc'  : data[operation_id_list_str]['val_acc_200'],
        'test_acc' : data[operation_id_list_str]['test_acc_200'],
    }
else:
    raise RuntimeError("Please chooese the benchmark from 'nas-macro', 'channel-res', 'channel-mob', '201-cifar10', '201-cifar100', '201-imagenet'")

print(performance)
