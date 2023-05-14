#!/bin/bash

# ================ train ====================
# ./script/command.sh genius-329M {YOUR_PARTITION} 8 1 "python3 -u train.py --batch_size 256 --lr 0.8 --wd 4e-5  --arch [1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 4, 0, 4, 3, 0, 0, 4, 3, 0, 4, 4, 0, 5, 0, 5, 5, 5, 0] --checkpoint genius-329M"
# ./script/command.sh genius-401M {YOUR_PARTITION} 8 1 "python3 -u train.py --batch_size 256 --lr 0.8 --wd 4e-5  --arch [1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 4, 4, 0, 4, 0, 4, 4, 5, 0, 5, 0, 5, 6, 5, 0, 6, 0, 5] --checkpoint genius-401M"


# ================ eval ====================
# ./script/command.sh eval-genius-329M {YOUR_PARTITION} 8 1 "python3 -u train.py --eval --arch [1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 4, 0, 4, 3, 0, 0, 4, 3, 0, 4, 4, 0, 5, 0, 5, 5, 5, 0] --checkpoint pretrained-329M"
# ./script/command.sh eval-genius-401M {YOUR_PARTITION} 8 1 "python3 -u train.py --eval --arch [1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 4, 4, 0, 4, 0, 4, 4, 5, 0, 5, 0, 5, 6, 5, 0, 6, 0, 5] --checkpoint pretrained-401M"
