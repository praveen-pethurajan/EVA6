## PART II
MNIST classification using CNN.   

**Requirements** 
1. Train for less than 20 epochs
2. Use less than 20K parameters
3. Accuracy >= 99.4%
4. Have used BN, Dropout, a Fully connected layer, have used GAP.  

**Network architecture**
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 4, 28, 28]              40
       BatchNorm2d-2            [-1, 4, 28, 28]               8
            Conv2d-3            [-1, 8, 28, 28]             296
       BatchNorm2d-4            [-1, 8, 28, 28]              16
            Conv2d-5           [-1, 12, 28, 28]             876
       BatchNorm2d-6           [-1, 12, 28, 28]              24
         MaxPool2d-7           [-1, 12, 14, 14]               0
            Conv2d-8           [-1, 16, 14, 14]           1,744
       BatchNorm2d-9           [-1, 16, 14, 14]              32
           Conv2d-10           [-1, 24, 12, 12]           3,480
      BatchNorm2d-11           [-1, 24, 12, 12]              48
        MaxPool2d-12             [-1, 24, 6, 6]               0
           Conv2d-13             [-1, 32, 4, 4]           6,944
      BatchNorm2d-14             [-1, 32, 4, 4]              64
        AvgPool2d-15             [-1, 32, 1, 1]               0
           Linear-16                   [-1, 10]             330
================================================================
Total params: 13,902
Trainable params: 13,902
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.42
Params size (MB): 0.05
Estimated Total Size (MB): 0.48
----------------------------------------------------------------
```

* **Training**
1. Trained for 19 epochs
2. Used 1 FC layer
3. SGD optimizer 
4. Learning rate of 0.1

* **Results**
1. Accuracy: 9946/10000 (99.46%)
