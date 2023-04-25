import os
import json
import openai
import numpy as np
from decimal import Decimal
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--openai_key', type=str, required=True)
parser.add_argument('--openai_organization', type=str, required=True)
args = parser.parse_args()
print(args)

openai.api_key = args.openai_key
openai.organization = args.openai_organization


benchmark_file = open('benchmark/nas-bench-macro_cifar10.json')
data = json.load(benchmark_file)
keys = list(data.keys())
rank = np.array([data[k]['mean_acc'] for k in keys]).argsort().argsort()
for k, r in zip(keys, rank):
    data[k]['rank'] = (3 ** 8) - r

system_content = "You are an expert in the field of neural architecture search."

user_input = '''Your task is to assist me in selecting the best operations for a given model architecture, which includes some undefined layers and available operations. The model will be trained and tested on CIFAR10, and your objective will be to maximize the model's performance on CIFAR10.

We define the 3 available operations as the following:
0: Identity(in_channels, out_channels, stride)
1: InvertedResidual(in_channels, out_channels, stride expansion=3, kernel_size=3)
2: InvertedResidual(in_channels, out_channels, stride expansion=6, kernel_size=5)

The implementation of the Identity is as follows:
class Identity(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Identity, self).__init__()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        return x

The implementation of the InvertedResidual is as follows:
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion, kernel_size):
        super(InvertedResidual, self).__init__()
        hidden_dim = in_channels * expansion
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.use_shortcut = in_channels == out_channels and stride == 1

    def forward(self, x):
        if self.use_shortcut:
            return self.conv(x) + x
        return self.conv(x)
        

The model architecture will be defined as the following.
{
    layer1:  {defined: True,  operation: nn.Conv2d(in_channels=3,  out_channels=32, kernel_size=3, padding=1, bias=False)},
    layer2:  {defined: False, downsample: True , in_channels: 32,  out_channels: 64 , stride: 2},
    layer3:  {defined: False, downsample: False, in_channels: 64,  out_channels: 64 , stride: 1},
    layer4:  {defined: False, downsample: True , in_channels: 64,  out_channels: 128, stride: 2},
    layer5:  {defined: False, downsample: False, in_channels: 128, out_channels: 128, stride: 1},
    layer6:  {defined: False, downsample: False, in_channels: 128, out_channels: 128, stride: 1},
    layer7:  {defined: False, downsample: True , in_channels: 128, out_channels: 256, stride: 2},
    layer8:  {defined: False, downsample: False, in_channels: 256, out_channels: 256, stride: 1},
    layer9:  {defined: False, downsample: False, in_channels: 256, out_channels: 256, stride: 1},
    layer10: {defined: True,  operation: nn.Conv2d(in_channels=256, out_channels=1280, kernel_size=1, bias=False, stride=1)},
    layer11: {defined: True,  operation: nn.AdaptiveAvgPool2d(output_size=1)},
    layer12: {defined: True,  operation: nn.Linear(in_features=1280, out_features=10)},
}

The currently undefined layers are layer2 - layer9, and the in_channels and out_channels have already been defined for each layer. To maximize the model's performance on CIFAR10, please provide me with your suggested operation for the undefined layers only. 

Your response should be an operation ID list for the undefined layers. For example:
[1, 2, ..., 0] means we use operation 1 for layer2, operation 2 for layer3, ..., operation 0 for layer9.
'''

experiments_prompt = lambda arch_list, acc_list : '''Here are some experimental results that you can use as a reference:
{}
Please suggest a better operation ID list that can improve the model's performance on CIFAR10 beyond the experimental results provided above.
'''.format(''.join(['{} gives an accuracy of {:.2f}%\n'.format(arch, acc) for arch, acc in zip(arch_list, acc_list)]))

suffix = '''Please do not include anything other than the operation ID list in your response.'''


arch_list = []
acc_list = []

messages = [
    {"role": "system", "content": system_content},
    {"role": "user", "content": user_input + suffix},
]



performance_history = []
messages_history = []

if not os.path.exists('history'):
    os.makedirs('history')

for iteration in range(10):
    res = openai.ChatCompletion.create(model='gpt-4', messages=messages, temperature=0, n=1)['choices'][0]['message']
    messages.append(res)
    messages_history.append(messages)

    print(res['content'])

    operation_id_list = json.loads(res['content'])
    operation_id_list_str = ''.join(str(opid) for opid in operation_id_list)
    accuracy = data[operation_id_list_str]['mean_acc']
    accuracy = float(Decimal(accuracy).quantize(Decimal("0.01"), rounding = "ROUND_HALF_UP"))

    arch_list.append(operation_id_list)
    acc_list.append(accuracy)
    
    performance = {
        'arch' : operation_id_list_str,
        'rank' : str(data[operation_id_list_str]['rank']),
        'acc'  : str(data[operation_id_list_str]['mean_acc']),
        'flops': str(data[operation_id_list_str]['flops']),
    }

    print(iteration+1, performance)

    performance_history.append(performance)

    with open('history/nas_bench_macro_messages.json', 'w') as f:
        json.dump(messages_history, f)
    
    with open('history/nas_bench_macro_performance.json', 'w') as f:
        json.dump(performance_history, f)
    
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_input + experiments_prompt(arch_list, acc_list) + suffix},
    ]
