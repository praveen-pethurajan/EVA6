### Session 7 Assignment ###

# Problem Statement
- Change the code such that it uses GPU
- Change the architecture to C1C2C3C40  (No MaxPooling, but 3 3x3 layers with stride of 2 instead) (If you can figure out how to use Dilated kernels here instead of MP or strided convolution, then 200pts extra!)
- Total RF must be more than 44
- One of the layers must use Depthwise Separable Convolution
- One of the layers must use Dilated Convolution
- Use GAP (compulsory):- add FC after GAP to target #of classes (optional)
- Use albumentation library and apply:
  - Horizontal flip
  - ShiftScaleRotate
  - CoarseDropout (max_holes = 1, max_height=16px, max_width=1, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
- Achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k. 

## Trained Model Summary

- Parameters - 176,426
- Epoch - 110
- Dropout - 0.1
- L2 - 0.001
- Best Training Accuracy - 85.45%
- Test Accuracy - 82.71%

Summary - 
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 32, 32]             896
              ReLU-2           [-1, 32, 32, 32]               0
       BatchNorm2d-3           [-1, 32, 32, 32]              64
           Dropout-4           [-1, 32, 32, 32]               0
            Conv2d-5           [-1, 64, 32, 32]          18,496
              ReLU-6           [-1, 64, 32, 32]               0
       BatchNorm2d-7           [-1, 64, 32, 32]             128
           Dropout-8           [-1, 64, 32, 32]               0
            Conv2d-9           [-1, 32, 32, 32]           2,080
           Conv2d-10           [-1, 32, 30, 30]           9,248
             ReLU-11           [-1, 32, 30, 30]               0
      BatchNorm2d-12           [-1, 32, 30, 30]              64
          Dropout-13           [-1, 32, 30, 30]               0
           Conv2d-14           [-1, 32, 28, 28]           9,248
             ReLU-15           [-1, 32, 28, 28]               0
      BatchNorm2d-16           [-1, 32, 28, 28]              64
          Dropout-17           [-1, 32, 28, 28]               0
           Conv2d-18           [-1, 32, 26, 26]           9,248
             ReLU-19           [-1, 32, 26, 26]               0
      BatchNorm2d-20           [-1, 32, 26, 26]              64
          Dropout-21           [-1, 32, 26, 26]               0
           Conv2d-22           [-1, 16, 22, 22]           4,624
             ReLU-23           [-1, 16, 22, 22]               0
           Conv2d-24           [-1, 32, 20, 20]           4,640
             ReLU-25           [-1, 32, 20, 20]               0
      BatchNorm2d-26           [-1, 32, 20, 20]              64
          Dropout-27           [-1, 32, 20, 20]               0
           Conv2d-28           [-1, 64, 18, 18]          18,496
             ReLU-29           [-1, 64, 18, 18]               0
      BatchNorm2d-30           [-1, 64, 18, 18]             128
          Dropout-31           [-1, 64, 18, 18]               0
           Conv2d-32           [-1, 64, 16, 16]          36,928
             ReLU-33           [-1, 64, 16, 16]               0
      BatchNorm2d-34           [-1, 64, 16, 16]             128
          Dropout-35           [-1, 64, 16, 16]               0
           Conv2d-36             [-1, 64, 7, 7]          36,928
             ReLU-37             [-1, 64, 7, 7]               0
           Conv2d-38             [-1, 16, 7, 7]           1,040
           Conv2d-39             [-1, 32, 5, 5]           4,640
             ReLU-40             [-1, 32, 5, 5]               0
      BatchNorm2d-41             [-1, 32, 5, 5]              64
          Dropout-42             [-1, 32, 5, 5]               0
           Conv2d-43             [-1, 64, 3, 3]          18,496
        AvgPool2d-44             [-1, 64, 1, 1]               0
           Conv2d-45             [-1, 10, 1, 1]             650
================================================================
Total params: 176,426
Trainable params: 176,426
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 7.28
Params size (MB): 0.67
Estimated Total Size (MB): 7.96
----------------------------------------------------------------
```

## Data Augumentation:-

The following data augumentation techniques are used:-
* HorizontalFlip
* ShiftScaleRotate
* CoarseDropout
* Normalize
* ToTensorV2

```
train_transform = A.Compose([
    A.HorizontalFlip(p=0.4),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
    A.CoarseDropout(max_holes = 1, max_height=4, max_width=4, min_holes = 1, 
                    min_height=1, min_width=1, fill_value=[0.49139968, 0.48215841, 0.44653091]),
    A.Normalize(mean=(0.49139968, 0.48215841, 0.44653091), std=(0.24703223, 0.24348513, 0.26158784)),
    ToTensorV2(),
])


test_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.229, 0.224, 0.225))])
```


## Training Logs:-

```
[1,   391] loss: 0.378 Acc: 26.068
[2,   391] loss: 0.310 Acc: 41.208
[3,   391] loss: 0.274 Acc: 48.500
[4,   391] loss: 0.253 Acc: 52.918
[5,   391] loss: 0.239 Acc: 55.628
[6,   391] loss: 0.226 Acc: 58.418
[7,   391] loss: 0.216 Acc: 60.404
[8,   391] loss: 0.207 Acc: 61.978
[9,   391] loss: 0.199 Acc: 63.432
[10,   391] loss: 0.193 Acc: 64.988
[11,   391] loss: 0.186 Acc: 66.214
[12,   391] loss: 0.180 Acc: 67.172
[13,   391] loss: 0.175 Acc: 68.212
[14,   391] loss: 0.171 Acc: 69.122
[15,   391] loss: 0.166 Acc: 70.174
[16,   391] loss: 0.160 Acc: 71.146
[17,   391] loss: 0.158 Acc: 71.852
[18,   391] loss: 0.155 Acc: 72.042
[19,   391] loss: 0.152 Acc: 72.620
[20,   391] loss: 0.148 Acc: 73.526
[21,   391] loss: 0.147 Acc: 73.796
[22,   391] loss: 0.143 Acc: 74.294
[23,   391] loss: 0.142 Acc: 74.544
[24,   391] loss: 0.139 Acc: 75.074
[25,   391] loss: 0.136 Acc: 75.496
[26,   391] loss: 0.136 Acc: 75.588
[27,   391] loss: 0.133 Acc: 76.348
[28,   391] loss: 0.131 Acc: 76.440
[29,   391] loss: 0.130 Acc: 76.866
[30,   391] loss: 0.128 Acc: 77.338
[31,   391] loss: 0.127 Acc: 77.190
[32,   391] loss: 0.125 Acc: 77.592
[33,   391] loss: 0.124 Acc: 77.854
[34,   391] loss: 0.123 Acc: 78.108
[35,   391] loss: 0.121 Acc: 78.370
[36,   391] loss: 0.120 Acc: 78.578
[37,   391] loss: 0.119 Acc: 78.726
[38,   391] loss: 0.118 Acc: 79.048
[39,   391] loss: 0.116 Acc: 79.282
[40,   391] loss: 0.116 Acc: 79.324
[41,   391] loss: 0.115 Acc: 79.590
[42,   391] loss: 0.113 Acc: 79.768
[43,   391] loss: 0.113 Acc: 79.834
[44,   391] loss: 0.112 Acc: 80.214
[45,   391] loss: 0.111 Acc: 80.154
[46,   391] loss: 0.110 Acc: 80.526
[47,   391] loss: 0.110 Acc: 80.274
[48,   391] loss: 0.108 Acc: 80.692
[49,   391] loss: 0.108 Acc: 80.550
[50,   391] loss: 0.106 Acc: 81.024
[51,   391] loss: 0.107 Acc: 80.930
[52,   391] loss: 0.106 Acc: 81.100
[53,   391] loss: 0.106 Acc: 81.108
[54,   391] loss: 0.104 Acc: 81.264
[55,   391] loss: 0.104 Acc: 81.350
[56,   391] loss: 0.103 Acc: 81.610
[57,   391] loss: 0.101 Acc: 81.912
[58,   391] loss: 0.102 Acc: 81.756
[59,   391] loss: 0.100 Acc: 82.010
[60,   391] loss: 0.100 Acc: 82.188
[61,   391] loss: 0.101 Acc: 82.004
[62,   391] loss: 0.101 Acc: 81.984
[63,   391] loss: 0.098 Acc: 82.546
[64,   391] loss: 0.099 Acc: 82.298
[65,   391] loss: 0.098 Acc: 82.488
[66,   391] loss: 0.097 Acc: 82.526
[67,   391] loss: 0.097 Acc: 82.630
[68,   391] loss: 0.096 Acc: 82.804
[69,   391] loss: 0.096 Acc: 82.754
[70,   391] loss: 0.096 Acc: 82.928
[71,   391] loss: 0.096 Acc: 82.900
[72,   391] loss: 0.094 Acc: 83.060
[73,   391] loss: 0.094 Acc: 83.300
[74,   391] loss: 0.093 Acc: 83.224
[75,   391] loss: 0.094 Acc: 83.402
[76,   391] loss: 0.092 Acc: 83.494
[77,   391] loss: 0.092 Acc: 83.564
[78,   391] loss: 0.091 Acc: 83.750
[79,   391] loss: 0.091 Acc: 83.482
[80,   391] loss: 0.091 Acc: 83.816
[81,   391] loss: 0.091 Acc: 83.674
[82,   391] loss: 0.090 Acc: 83.796
[83,   391] loss: 0.090 Acc: 84.116
[84,   391] loss: 0.089 Acc: 84.106
[85,   391] loss: 0.090 Acc: 83.870
[86,   391] loss: 0.088 Acc: 84.214
[87,   391] loss: 0.088 Acc: 84.396
[88,   391] loss: 0.088 Acc: 84.232
[89,   391] loss: 0.088 Acc: 84.274
[90,   391] loss: 0.087 Acc: 84.420
[91,   391] loss: 0.087 Acc: 84.494
[92,   391] loss: 0.087 Acc: 84.526
[93,   391] loss: 0.086 Acc: 84.652
[94,   391] loss: 0.086 Acc: 84.672
[95,   391] loss: 0.085 Acc: 84.770
[96,   391] loss: 0.085 Acc: 84.880
[97,   391] loss: 0.085 Acc: 84.554
[98,   391] loss: 0.084 Acc: 84.880
[99,   391] loss: 0.085 Acc: 84.886
[100,   391] loss: 0.085 Acc: 84.880
[101,   391] loss: 0.085 Acc: 84.802
[102,   391] loss: 0.083 Acc: 85.146
[103,   391] loss: 0.082 Acc: 85.266
[104,   391] loss: 0.084 Acc: 85.058
[105,   391] loss: 0.083 Acc: 85.120
[106,   391] loss: 0.083 Acc: 85.304
[107,   391] loss: 0.082 Acc: 85.334
[108,   391] loss: 0.082 Acc: 85.270
[109,   391] loss: 0.082 Acc: 85.452
[110,   391] loss: 0.082 Acc: 85.360

```

## Accuracy by class

```
Accuracy of plane : 86 %
Accuracy of   car : 92 %
Accuracy of  bird : 81 %
Accuracy of   cat : 55 %
Accuracy of  deer : 77 %
Accuracy of   dog : 66 %
Accuracy of  frog : 77 %
Accuracy of horse : 80 %
Accuracy of  ship : 96 %
Accuracy of truck : 89 %
```









