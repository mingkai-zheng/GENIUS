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


benchmark_file = open('benchmark/Results_MobileNet.json')
data = json.load(benchmark_file)
keys = list(data.keys())
rank = np.array([data[k]['mean'] for k in keys]).argsort().argsort()
for k, r in zip(keys, rank):
    data[k]['rank'] = (4 ** 7) - r


system_content = "You are a expert in the field of neural architecture search."

user_input = '''Your task is to assist me in selecting the best channel numbers for a given model architecture. The model will be trained and tested on CIFAR10, and your objective will be to maximize the model's performance on CIFAR10. 

The model architecture will be defined as the following.
{
    layer1: nn.Conv2d(in_channels=3, out_channels=channels[0], kernel_size=3, padding=1, bias=False),
    layer2: InvertedResidual(in_channels=channels[0], bottleneck_channels=channels[1], out_channels=channels[0], stride=1),
    layer3: InvertedResidual(in_channels=channels[0], bottleneck_channels=channels[2], out_channels=channels[0], stride=1),
    layer4: InvertedResidual(in_channels=channels[0], bottleneck_channels=channels[3], out_channels=channels[4], stride=2),
    layer5: InvertedResidual(in_channels=channels[4], bottleneck_channels=channels[5], out_channels=channels[4], stride=1),
    layer6: nn.Conv2d(channels[4], channels[6], kernel_size=1, stride = 1, padding=0, bias=False),
    layer7: nn.AdaptiveAvgPool2d(output_size=1),
    layer8: nn.Linear(in_features=channels[6], out_features=10),
}

The implementation of the InvertedResidual is as follows:
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck_channels, stride):
        super(InvertedResidual, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride, padding=1, groups=bottleneck_channels, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.use_shortcut = in_channels == out_channels and stride == 1

    def forward(self, x):
        if self.use_shortcut:
            return self.conv(x) + x
        return self.conv(x)

For the `channels` variable, the available channel number for each index would be:
{
    channels[0]: [32,  64,  96,  128],
    channels[1]: [192, 384, 576, 768],
    channels[2]: [192, 384, 576, 768],
    channels[3]: [192, 384, 576, 768],
    channels[4]: [64,  128, 192, 256],
    channels[5]: [384, 768, 1152, 1536],
    channels[6]: [256, 512, 768, 1024],
}

Your objective is to define the optimal number of channels for each layer based on the given options above to maximize the model's performance on CIFAR10. 
Your response should be the a channel list consisting of 7 numbers (e.g. [64, 576, ..., 256]).
'''


experiments_prompt = lambda arch_list, acc_list : '''Here are some experimental results that you can use as a reference:
{}
Please suggest a better channel list that can improve the model's performance on CIFAR10 beyond the experimental results provided above.
'''.format(''.join(['{} gives an accuracy of {:.2f}%\n'.format(arch, acc) for arch, acc in zip(arch_list, acc_list)]))

suffix = '''Please do not include anything else other than the channel list in your response.'''

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

base_channels = [32, 192, 192, 192, 64, 384, 256]

for iteration in range(10):
    res = openai.ChatCompletion.create(model='gpt-4', messages=messages, temperature=0, n=1)['choices'][0]['message']
    messages.append(res)
    messages_history.append(messages)

    # print(messages)
    print(res['content'])

    channels = json.loads(res['content'])
    search_id = ''.join([str(int(c / base_c)) for base_c, c in zip(base_channels, channels)])
    accuracy = data[search_id]['mean']
    accuracy = float(Decimal(accuracy).quantize(Decimal("0.01"), rounding = "ROUND_HALF_UP"))

    arch_list.append(channels)
    acc_list.append(accuracy)
    
    performance = {
        'arch' : channels,
        'rank' : str(data[search_id]['rank']),
        'acc'  : str(data[search_id]['mean']),
        'flops': str(data[search_id]['flops']),
    }

    print(iteration+1, performance)

    performance_history.append(performance)

    with open('history/channel_bench_mob_messages.json', 'w') as f:
        json.dump(messages_history, f)
    
    with open('history/channel_bench_mob_performance.json', 'w') as f:
        json.dump(performance_history, f)
    
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_input + experiments_prompt(arch_list, acc_list) + suffix},
    ]
