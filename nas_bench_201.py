import os
import json
import openai
from decimal import Decimal
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--openai_key', type=str, required=True)
parser.add_argument('--openai_organization', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'cifar100', 'imagenet'])
args = parser.parse_args()
print(args)

openai.api_key = args.openai_key
openai.organization = args.openai_organization

if args.dataset == 'cifar10':
    benchmark_file = open('benchmark/nasbench201_cifar10.json')
elif args.dataset == 'cifar100':
    benchmark_file = open('benchmark/nasbench201_cifar100.json')
else:
    benchmark_file = open('benchmark/nasbench201_imagenet.json')

data = json.load(benchmark_file)

system_content = "You are Quoc V. Le, a computer scientist and artificial intelligence researcher who is widely regarded as one of the leading experts in deep learning and neural network architecture search. Your work in this area has focused on developing efficient algorithms for searching the space of possible neural network architectures, with the goal of finding architectures that perform well on a given task while minimizing the computational cost of training and inference."

user_input = '''You are an expert in the field of neural architecture search. Your task is to assist me in selecting the best operations to design a neural network block using the available operations. The objective is to maximize the model's performance.

The 5 available operations are as follows:
0: Zeroize()     # This operation simply outputs a tensor of zeros regardless of the input, which breaks the gradient flow between two nodes.
1: nn.Identity() # Skip Connection.
2: ReLUConvBN(channels, channels, kernal_size=1, stride=1, padding=0) # The input channels and output channels are the same.
3: ReLUConvBN(channels, channels, kernal_size=3, stride=1, padding=1) # The input channels and output channels are the same.
4: nn.AvgPool2d(kernel_size=3, stride=1, padding=1)                   # This operation does not change the spatial resolution.

The neural network block is defined by 6 operations (i.e., op_list = [op0, op1, op2, op3, op4, op5]), which represent the operations executed between various stages of the block. This block comprises 4 stages, labeled as s0, s1, s2, and s3, each corresponding to distinct feature maps in the neural network.

s0 serves as the input feature map for this block.
s1 will be calculated by s1 = op0(s0).
s2 will be calculated by s2 = op1(s0) + op2(s1).
s3 will be calculated by s3 = op3(s0) + op4(s1) + op5(s2). Note that s3 becomes the output for this block and serves as the input for the subsequent block.

Then the implementation of the block will be:
class Block(nn.Module):
    def __init__(self, channels):
        super(Block, self).__init__()
        self.op0 = op_id_list[0]
        self.op1 = op_id_list[1]
        self.op2 = op_id_list[2]
        self.op3 = op_id_list[3]
        self.op4 = op_id_list[4]
        self.op5 = op_id_list[5]

    def forward(self, s0):
        s1 = self.op0(s0)
        s2 = self.op1(s0) + self.op2(s1)
        s3 = self.op3(s0) + self.op4(s1) + self.op5(s2)
        return s3

Let's break this down step by step:

First, please analyze the 5 available operations.

Next, please consider the gradient flow based on the Block class implementation. For example, how the gradient from the later stage affects the earlier stage.

Now, answer the question - how we can design a high-performance block using the available operations?

Based the analysis, your task is to propose a block design with the given operations that prioritizes performance, without considering factors such as size and complexity.

After you suggest a design, I will test its actual performance and provide you with feedback. Based on the results of previous experiments, we can collaborate to iterate and improve the design. Please avoid suggesting the same design again during this iterative process.
'''

experiments_prompt = lambda x : '''By using this model, we achieved an accuracy of {}%. Please recommend a new model that outperforms prior architectures based on the abovementioned experiments. Also, Please provide a rationale explaining why the suggested model surpasses all previous architectures.'''.format(x)

test_acc_list = []
val_acc_list = []
rank_list = []

messages = [
    {"role": "system", "content": system_content},
    {"role": "user", "content": user_input},
]

performance_history = []

if not os.path.exists('history'):
    os.makedirs('history')

num_iters = 0
for iteration in range(num_iters, 10):
    res = openai.ChatCompletion.create(model='gpt-4', messages=messages, temperature=0, n=1)['choices'][0]['message']
    messages.append(res)
    print('Assistant:', res['content'])

    arch = input('\nUser: Please enter the GPT-4 suggested model (use 6 operation IDs to represent the model):')
    print()
    operation_id_list = [int(opid) for opid in list(arch)]
    struct_dict = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
    operation_id_list_str = '|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|'.format(*[struct_dict[operation_id] for operation_id in operation_id_list])

    rank = data[operation_id_list_str]['rank']
    val_acc = data[operation_id_list_str]['val_acc_200']
    test_acc = data[operation_id_list_str]['test_acc_200']

    val_acc = float(Decimal(val_acc).quantize(Decimal("0.001"), rounding = "ROUND_HALF_UP"))
    test_acc = float(Decimal(test_acc).quantize(Decimal("0.001"), rounding = "ROUND_HALF_UP"))

    test_acc_list.append(test_acc)
    val_acc_list.append(val_acc)
    rank_list.append(rank)

    performance = {
        'rank' : rank, 
        'val_acc'  : val_acc,
        'test_acc' : test_acc,
    }

    print(iteration+1, performance, '\n')

    performance_history.append(performance)

    with open('history/nas_bench_201_{}_messages.json'.format(args.dataset), 'w') as f:
        json.dump(messages, f)
    

    with open('history/nas_bench_201_{}_performance.json'.format(args.dataset), 'w') as f:
        json.dump(performance_history, f)

    messages.append({"role": "user", "content": experiments_prompt(val_acc)})
