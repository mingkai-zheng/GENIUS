# Can GPT-4 Perform Neural Architecture Search?

## Retrival Performance From Benchmark

### * NAS-Bench-Macro 
```
python get_performance.py --benchmark nas-macro --arch xxxxxxxx
```
xxxxxxxx is 8 numbers (e.g. 01201201) which representes the operation for each layer. There are three different choices for each layer, you can use [0, 1, 2] to represents the operations. The details and avialable operations can be found in [prompt/nas-bench-macro.md](prompt/nas-bench-macro.md)

- **In this experiment, with Temperature = 1, we get following results.**

<table>
  <thead>
    <tr>
      <th></th> <th></th> 
      <th>T=0</th> <th>T=1</th> <th>T=2</th> <th>T=3</th> <th>T=4</th> <th>T=5</th> <th>T=6</th> <th>T=7</th> <th>T=8</th> <th>T=9</th> <th style="color:gray">Optimal</th> 
    </tr>
  </thead>
  <tbody>
    <tr>
    <td rowspan="2"> Trial 1 </td> 
    <td> Acc  </td>
    <td> 90.90 </td>
    <td> 92.40 </td>
    <td> 92.30 </td>
    <td> 92.53 </td>
    <td> 92.63 </td>
    <td> 92.66 </td>
    <td style="font-weight: bold"> 92.97 </td>
    <td> 92.56 </td>
    <td> 92.50 </td>
    <td> 92.56 </td>
    <td style="color:gray"> 93.13</td>
    </tr>
    <tr>
    <td>  Ranking </td>
    <td>  3440 </td>
    <td>  590 </td>
    <td>  766 </td>
    <td>  353 </td>
    <td>  203 </td>
    <td>  180 </td>
    <td style="font-weight: bold">  19 </td>
    <td>  311 </td>
    <td>  394 </td>
    <td>  314 </td>
    <td style="color:gray">  1 </td>
    </tr>
    <tr>
    <td rowspan="2"> Trial 2 </td> 
    <td>  Acc </td>
    <td>  90.42 </td>
    <td>  92.49 </td>
    <td>  92.53 </td>
    <td style="font-weight: bold">  92.85 </td>
    <td>  92.54 </td>
    <td>  92.56 </td>
    <td>  92.58 </td>
    <td>  92.73 </td>
    <td>  92.48 </td>
    <td>  92.78 </td>
    <td style="color:gray">  93.13 </td>
    </tr>
    <tr>
    <td>  Ranking </td>
    <td>  4042 </td>
    <td>  442 </td>
    <td>  384 </td>
    <td style="font-weight: bold">  50 </td>
    <td>  332 </td>
    <td>  331 </td>
    <td>  272 </td>
    <td>  119 </td>
    <td>  446 </td>
    <td>  82 </td>
    <td style="color:gray">  1 </td>
    </tr>
    <td rowspan="2"> Trial 3 </td> 
    <td>  Acc  </td>
    <td>  91.35 </td>
    <td>  92.78 </td>
    <td style="font-weight: bold">  92.82 </td>
    <td>  92.74 </td>
    <td>  92.34 </td>
    <td>  92.35 </td>
    <td>  92.45 </td>
    <td>  92.56 </td>
    <td>  92.54 </td>
    <td>  92.66 </td>
    <td style="color:gray">  93.13 </td>
    <tr>    
    <td>  Ranking </td>
    <td>  2609 </td>
    <td>  83 </td>
    <td style="font-weight: bold">  65  </td>
    <td>  117 </td>
    <td>  683 </td>
    <td>  664 </td>
    <td>  483 </td>
    <td>  311 </td>
    <td>  341 </td>
    <td>  180 </td>
    <td style="color:gray">  1 </td>
    </tr>
  </tbody>
</table>



- **In this experiment, with Temperature = 0, we get following results. - means GPT-4 thinks there is no chance to improve the performance further.**
<table>
  <thead>
    <tr>
      <th></th> <th></th> 
      <th>T=0</th> <th>T=1</th> <th>T=2</th> <th>T=3</th> <th>T=4</th> <th>T=5</th> <th>T=6</th> <th>T=7</th> <th>T=8</th> <th>T=9</th> <th style="color:gray">Optimal</th> 
    </tr>
  </thead>
  <tbody>
    <tr>
        <td rowspan="2"> Trial 1 </td> 
        <td> Acc  </td>
        <td> 85.70 </td>
        <td> 92.62 </td>
        <td> 92.82 </td>
        <td  style="font-weight: bold"> 93.05 </td>
        <td> 92.95 </td>
        <td> 92.46 </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td style="color:gray"> 93.13 </td>
    </tr>
    <tr>     
        <td> Ranking </td>
        <td> 6221 </td>
        <td> 212 </td>
        <td> 64 </td>
        <td  style="font-weight: bold"> 8 </td>
        <td> 21 </td>
        <td> 479 </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td style="color:gray"> 1 </td>
    </tr>
    <tr>
        <td rowspan="2"> Trial 2 </td> 
        <td> Acc  </td>
        <td> 92.45  </td>
        <td> 92.66 </td>
        <td  style="font-weight: bold"> 92.92 </td>
        <td> 92.64 </td>
        <td> 92.33 </td>
        <td> 92.72 </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td style="color:gray"> 93.13 </td>
    </tr>
    <tr>       
        <td> Ranking </td>
        <td> 496 </td>
        <td> 189 </td>
        <td  style="font-weight: bold"> 27 </td>
        <td> 198 </td>
        <td> 695 </td>
        <td> 128 </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td style="color:gray"> 1 </td>
    </tr>
    <tr>
        <td rowspan="2"> Trial 3 </td> 
        <td> Acc  </td>
        <td> 92.41 </td>
        <td> 92.74 </td>
        <td  style="font-weight: bold"> 92.83 </td>
        <td> 92.74 </td>
        <td> 92.33 </td>
        <td> 92.53 </td>
        <td> 92.69 </td>
        <td> 92.34 </td>
        <td> 92.56 </td>
        <td> 92.72 </td>
        <td style="color:gray"> 93.13 </td>
    </tr>
    <tr>     
        <td> Ranking </td>
        <td> 564 </td>
        <td> 113 </td>
        <td  style="font-weight: bold"> 61 </td>
        <td> 112 </td>
        <td> 689 </td>
        <td> 352 </td>
        <td> 152 </td>
        <td> 683 </td>
        <td> 314 </td>
        <td> 128 </td>
        <td style="color:gray"> 1 </td>
    </tr>
  </tbody>
</table>


### * Channel-Bench-Macro 
```
python get_performance.py --benchmark channel-res --arch 'xx, xx, xx, xx, xx, xx, xx'
python get_performance.py --benchmark channel-mob --arch 'xx, xx, xx, xx, xx, xx, xx'
```
Use ``channel-res`` for ResNet base model and ``channel-mob`` for MobileNet base model. xx represents the channel numers of each layer. You can find the details for the avialable channel numbers in [prompt/channel-bench-resnet.md](prompt/channel-bench-resnet.md) and [prompt/channel-bench-mobilenet.md](prompt/channel-bench-mobilenet.md)

### * NAS-Bench-201
```
python get_performance.py --benchmark 201-cifar10  --arch xxxxxx
python get_performance.py --benchmark 201-cifar100 --arch xxxxxx
python get_performance.py --benchmark 201-imagenet --arch xxxxxx
```
Use ``201-cifar10``, ``201-cifar100``, and ``201-imagenet`` for CIFA10, CIFAR100, and ImageNet16-120 respectively. xxxxxx is 6 numbers (e.g. 213401) which representes the operation for each edge. There are three different choices for each layer, you can use [0, 1, 2, 3, 4] to represents the operations. The details and avialable operations can be found in [prompt/nas-bench-201.md](prompt/nas-bench-201.md)

