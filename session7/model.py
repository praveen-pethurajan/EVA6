import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
  def __init__(self):
    
      super(Net, self).__init__()
      self.conv1 = nn.Sequential(
          nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1),
          nn.ReLU(),
          nn.BatchNorm2d(32),
          nn.Dropout(0.1),

          nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1),
          nn.ReLU(),
          nn.BatchNorm2d(64),
          nn.Dropout(0.1),
      )
      self.trans1 = nn.Sequential(
          nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1))
      )
      self.conv2 = nn.Sequential(
          nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3)),
          nn.ReLU(),
          nn.BatchNorm2d(32),
          nn.Dropout(0.1),
          
          nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3)),
          nn.ReLU(),
          nn.BatchNorm2d(32),
          nn.Dropout(0.1),

          nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3)),
          nn.ReLU(),
          nn.BatchNorm2d(32),
          nn.Dropout(0.1),
      )
      # dilation
      self.trans2 = nn.Sequential(
          nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size = (3, 3), dilation=2),
          nn.ReLU(),
      )
      self.conv3 = nn.Sequential(
          nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3)),
          nn.ReLU(),
          nn.BatchNorm2d(32),
          nn.Dropout(0.1),
          
          nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3)),
          nn.ReLU(),
          nn.BatchNorm2d(64),
          nn.Dropout(0.1),

          nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3)),
          nn.ReLU(),
          nn.BatchNorm2d(64),
          nn.Dropout(0.1),
      )

      self.trans3 = nn.Sequential(
          nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), stride = 2),
          nn.ReLU(),
          nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(1, 1))
      )

      self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3)),
          )
      
      self.avgpool2d = nn.AvgPool2d(kernel_size=3)

      self.conv5 = nn.Sequential(
          nn.Conv2d(in_channels=64, out_channels=10, kernel_size=(1, 1)))
      
  def forward(self, x):
      x = self.conv1(x)
      x = self.trans1(x)
      x = self.conv2(x)
      x = self.trans2(x)
      x = self.conv3(x)
      x = self.trans3(x)
      x = self.conv4(x)
      x = self.avgpool2d(x)
      x = self.conv5(x)
      x = x.view(-1, 10)
      return F.log_softmax(x, dim=-1)