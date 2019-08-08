# BMN: Boundary-Matching Network

A pytorch-version implementation codes of paper:
 "BMN: Boundary-Matching Network for Temporal Action Proposal Generation",
  which is accepted in ICCV 2019. 

[[Arxiv Preprint]](https://arxiv.org/abs/1907.09702)


# Prerequisites

These code is  implemented in Pytorch 0.4.1 + Python3 . 


## Download Datasets

 The author rescale the feature length of all videos 
to same length 100, and he provide the rescaled feature at 
here [BSN](https://github.com/wzmsltw/BSN-boundary-sensitive-network) .


# Training and Testing  of BMN

All configurations of BMN are saved in opts.py, where you can modify training and model parameter.


1. For the first time to run the data, you should this cmd to generate the BM mask matrix:
```
python get_mask.py
```

2. To train the BMN:
```
python main.py --module BMN --mode train
```

3. To get the inference proposal of the validation videos:
```
python main.py --module BMN --mode inference
```

4. To use the soft_nms to reduce the redundancy of the proposals:
```
python main.py --module Post_processing
```

5. To evaluate the proposals with recall and AUC:
```
python main.py --module Evaluation
```

Of course, you can complete all the process above in one line: 

```
sh bmn.sh
```



## Reference

code:[BSN](https://github.com/wzmsltw/BSN-boundary-sensitive-network)

paper:[BMN: Boundary-Matching Network for Temporal Action Proposal Generation](https://arxiv.org/abs/1907.09702)


