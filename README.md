# Can GPT-4 Perform Neural Architecture Search?

For details, see [Paper Link](https://arxiv.org/pdf/2304.10970.pdf) by Mingkai Zheng, Xiu Su, Shan You, Fei Wang, Chen Qian, Chang Xu, and Samuel Albanie.

## Reproduce
**The results presented in the paper for NAS-Bench-Macro, Channel-Bench-Macro, and NAS-Bench-201 were generated using the code provided below. Although we set the temperature to 0 in the code, it is important to acknowledge that some level of randomness may persist. Consequently, the results obtained from executing the code may not perfectly match a specific experimental run in the paper. Nevertheless, as the code is run multiple times, the average performance should align with the overall performance reported in the paper.**

### * NAS-Bench-Macro 
```
python nas_bench_macro.py --openai_key {YOUR_OPENAI_API_KEY} --openai_organization {YOUR_OPENAI_ORGANIZATION}
```

### * Channel-Bench-Macro 
```
# For ResNet
python channel_bench_res.py --openai_key {YOUR_OPENAI_API_KEY} --openai_organization {YOUR_OPENAI_ORGANIZATION}
```
```
# For MobileNet
python channel_bench_mob.py --openai_key {YOUR_OPENAI_API_KEY} --openai_organization {YOUR_OPENAI_ORGANIZATION}
```

### * NAS-Bench-201
```
python nas_bench_201.py --openai_key {YOUR_OPENAI_API_KEY} --openai_organization {YOUR_OPENAI_ORGANIZATION} --dataset {DATASET}
```
For {DATASET}, you can select from ['cifar10', 'cifar100', 'imagenet']. Please note that this is an interactive program, requiring manual interpretation of GPT-4's suggested model into six numbers to retrieve the performance and provide feedback to GPT-4.


## Retrieve Performance From Benchmark

### * NAS-Bench-Macro 
```
python get_performance.py --benchmark nas-macro --arch xxxxxxxx
```
xxxxxxxx is 8 numbers (e.g. 01201201) which representes the operation for each layer. There are three different choices for each layer, you can use [0, 1, 2] to represents the operations.

### * Channel-Bench-Macro 
```
python get_performance.py --benchmark channel-res --arch 'xx, xx, xx, xx, xx, xx, xx'
python get_performance.py --benchmark channel-mob --arch 'xx, xx, xx, xx, xx, xx, xx'
```
Use ``channel-res`` for ResNet base model and ``channel-mob`` for MobileNet base model. xx represents the channel numers of each layer.


### * NAS-Bench-201
```
python get_performance.py --benchmark 201-cifar10  --arch xxxxxx
python get_performance.py --benchmark 201-cifar100 --arch xxxxxx
python get_performance.py --benchmark 201-imagenet --arch xxxxxx
```
Use ``201-cifar10``, ``201-cifar100``, and ``201-imagenet`` for CIFA10, CIFAR100, and ImageNet16-120 respectively. xxxxxx is 6 numbers (e.g. 213401) which representes the operation for each edge. There are three different choices for each layer, you can use [0, 1, 2, 3, 4] to represents the operations.


## Reference
```
@misc{zheng2023gpt4,
    title={Can GPT-4 Perform Neural Architecture Search?}, 
    author={Mingkai Zheng and Xiu Su and Shan You and Fei Wang and Chen Qian and Chang Xu and Samuel Albanie},
    year={2023},
    eprint={2304.10970},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
