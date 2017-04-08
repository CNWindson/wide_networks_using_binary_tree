# What's this
Implementation of wide_networks_using_binary_tree [[1]][Paper] by chainer


# Dependencies

    git clone https://github.com/nutszebra/wide_networks_using_binary_tree.git
    cd wide_networks_using_binary_tree
    git submodule init
    git submodule update

# How to run
    python main.py -p ./ -g 0 

# Details about my implementation

* Data augmentation  
Train: Pictures are randomly resized in the range of [32, 36], then 32x32 patches are extracted randomly and are normalized locally. Horizontal flipping is applied with 0.5 probability.  
Test: Pictures are resized to 32x32, then they are normalized locally. Single image test is used to calculate total accuracy. 

* Optimization  
Momentum SGD with 0.9 momentum  

* Weight decay    
0.0005  

* Batch size  
128  

* lr  
Initial learning rate is 0.2 and is multiplied by 0.2 at [60, 120, 160] epochs. Total epochs is 200.

# Cifar10 result

| network              | d | k | n | number of parameters      | total accuracy (%) |
|:---------------------|---|---|---|---------------------------|-------------------:|
| [[1]][Paper]         | 4 | 6 | 2 | 1.7M                      | 95.23              |
| my implementation    | 4 | 6 | 2 | 1.67M                     | soon               |


<img src="https://github.com/nutszebra/wide_networks_using_binary_tree/blob/master/loss.jpg" alt="loss" title="loss">
<img src="https://github.com/nutszebra/wide_networks_using_binary_tree/blob/master/accuracy.jpg" alt="total accuracy" title="total accuracy">

# References
wide_networks_using_binary_tree [[1]][Paper]  

[paper]: https://arxiv.org/abs/1704.00509 "Paper"
