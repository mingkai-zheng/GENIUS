# Can GPT-4 Perform Neural Architecture Search?

## Retrival Performance From Benchmark

### * NAS-Bench-Macro 
```
python get_performance.py --benchmark nas-macro --arch xxxxxxxx
```
xxxxxxxx is an 8 number (e.g. 01201201) which representes the operation for each layer. There are three different choices for each layer, you can use [0, 1, 2] to represents the operations.

### * Channel-Bench-Macro 
```
python get_performance.py --benchmark channel-res --arch xxxxxxxx
python get_performance.py --benchmark channel-mob --arch xxxxxxxx
```
Use ``channel-res`` for ResNet base model and ``channel-mob`` for MobileNet base model. xxxxxxx is an 7 digital number (e.g. 11223344) which representes the channel numers of each layer. There are four different choices for each layer, you can use [1, 2, 3, 4] to represents the operations.


