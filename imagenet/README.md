# ImageNet Experiments

## Requirement
```
torch==1.13.0
timm=0.6.12
thop
```


## Search / Chat History
We provide the complete chat history with GPT-4, including the conversations for GENIUS-329, GENIUS-401, as well as the ablation study experiments. Please refer to the [chat_history](./chat_history) folder


## Reproduce Retraining
To run the code, you need to change the Dataset setting (Imagenet function in [data/imagenet.py](./data/imagenet.py)), and Pytorch DDP setting (dist_init function in  [util/dist_init.py](./util/dist_init.py)) for your server environment.

The distributed training of this code is based on slurm environment, we have provided the training and evaluation scripts in [script/script.sh](./script/script.sh)

We also provide the pre-trained model.
|          |FLOPs(M) | Param(M) | Top-1 Accuracy | Download  |
|----------|:----:|:---:|:---:|:---:|
|  GENIUS | 329 | 7.0 | 77.8% | [GENIUS-329.pth.tar](https://drive.google.com/file/d/1DbV27hWMq0aRl-SJ4vuphFduBQwr1RUr/view?usp=sharing) |
|  GENIUS | 401 | 7.5 | 78.2% | [GENIUS-401.pth.tar](https://drive.google.com/file/d/1R-qp6XlebgQji3UtbJ5yrc4UT2Bg3fJw/view?usp=sharing) |

If you want to test the pre-trained model, please download the weights from the link above, and move them to the checkpoint folder.


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
