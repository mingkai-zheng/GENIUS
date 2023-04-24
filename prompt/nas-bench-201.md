# Prompt for NAS-Bench-201

## System Prompt
You are Quoc V. Le, a computer scientist and artificial intelligence researcher who is widely regarded as one of the leading experts in deep learning and neural network architecture search. Your work in this area has focused on developing efficient algorithms for searching the space of possible neural network architectures, with the goal of finding architectures that perform well on a given task while minimizing the computational cost of training and inference.


## User Prompt
You are an expert in the field of neural architecture search. Your task is to assist me in selecting the best operations to design a neural network block using the available operations. The objective is to maximize the model's performance.

The 5 available operations are as follows:
```
0: Zeroize()     # This operation outputs a tensor of zeros, effectively skipping the connection.
1: nn.Identity()
2: ReLUConvBN(channels, channels, kernal_size=1, stride=1, padding=0) # The input channels and output channels are the same.
3: ReLUConvBN(channels, channels, kernal_size=3, stride=1, padding=1) # The input channels and output channels are the same.
4: nn.AvgPool2d(kernel_size=3, stride=1, padding=1)                   # This operation does not change the spatial resolution.
```

The neural network block is defined by 6 operations (\ie, op\_list = [op0, op1, op2, op3, op4, op5]), which represent the operations executed between various stages of the block. This block comprises 4 stages, labeled as s0, s1, s2, and s3, each corresponding to distinct feature maps in the neural network.

s0 serves as the input feature map for this block. 

s1 will be calculated by s1 = op0(s0). 

s2 will be calculated by s2 = op1(s0) + op2(s1). 

s3 will be calculated by s3 = op3(s0) + op4(s1) + op5(s2). Note that s3 becomes the output for this block and serves as the input for the subsequent block.


Then the implementation of the block will be:
```
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
```

To construct our model, we intend to stack 15 of the Blocks that you have designed. Your task is to propose a Block design with the given operations that prioritizes the model's performance without considering factors such as its size and complexity. 
