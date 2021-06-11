from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from torchvision import datasets, transforms
import argparse
from torch.optim.lr_scheduler import StepLR,OneCycleLR

training_losses_batch = []
testing_losses_batch = []
training_accuracy_batch = []
testing_accuracy_batch = []
training_losses_layer = []
testing_losses_layer = []
training_accuracy_layer = []
testing_accuracy_layer = []
training_losses_group = []
testing_losses_group = []
training_accuracy_group = []
testing_accuracy_group = []

def normalize_3_way(arg):
    group = False
    layer = False
    batch = False
    if arg == "layer":
        layer = True
    elif arg == "group":
        group = True
    elif arg == "batch":
        batch = True
    else:
        print("Incorrect Normalization Method...")
        exit()

    train_set = datasets.MNIST('../data', 
                    train=True, 
                    download=True,
                    transform=transforms.Compose([
                                        transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,)) 
                                        # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values. 
                                        ]))


    test_set = datasets.MNIST('../data', 
                    train=False, 
                    download=True,
                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                        ]))

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    torch.manual_seed(1)
    if use_cuda:
        torch.cuda.manual_seed(1)

    batch_size=128

    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, **kwargs)


    if layer:
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

                self.convblock1 = nn.Sequential(
                    nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3)),
                    nn.ReLU(),
                    nn.LayerNorm((8, 26, 26)),
                    nn.Dropout(0.05)
                ) # output_size = 26


                # CONVOLUTION BLOCK 1
                self.convblock2 = nn.Sequential(
                    nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3)),
                    nn.ReLU(),
                    nn.LayerNorm((8, 24, 24)),
                    nn.Dropout(0.05)
                ) # output_size = 24

                # TRANSITION BLOCK 1
                self.convblock3 = nn.Sequential(
                    nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3)),
                    nn.ReLU(),
                    nn.LayerNorm((12, 22, 22)),
                    nn.Dropout(0.05)
                ) # output_size = 22
                self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11

                # CONVOLUTION BLOCK 2
                self.convblock4 = nn.Sequential(
                    nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3)),
                    nn.ReLU(),            
                    nn.LayerNorm((12, 9, 9)),
                    nn.Dropout(0.05)
                ) # output_size = 9

                self.convblock5 = nn.Sequential(
                    nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3)),
                    nn.ReLU(),            
                    nn.LayerNorm((16, 7, 7)),
                    nn.Dropout(0.05)
                ) # output_size = 7

                self.convblock6 = nn.Sequential(
                    nn.Conv2d(in_channels=16, out_channels=20, kernel_size=(3, 3)),
                    nn.ReLU(),            
                    nn.LayerNorm((20, 5, 5)),
                    nn.Dropout(0.05)
                ) # output_size = 5
                
                # OUTPUT BLOCK
                self.gap = nn.Sequential(
                    nn.AvgPool2d(kernel_size=5)
                ) # output_size = 1

                self.convblock7 = nn.Sequential(
                    nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1)),
                ) 

            def forward(self, x):
                x = self.convblock1(x)
                x = self.convblock2(x)
                x = self.convblock3(x)
                x = self.pool1(x)
                x = self.convblock4(x)
                x = self.convblock5(x)
                x = self.convblock6(x)
                x = self.gap(x)        
                x = self.convblock7(x)
                x = x.view(-1, 10)
                return F.log_softmax(x, dim=-1)

    elif batch:
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

                # Input Block
                self.convblock1 = nn.Sequential(
                    nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3)),
                    nn.ReLU(),
                    nn.BatchNorm2d(8),
                    nn.Dropout(0.05)
                ) # output_size = 26

                # CONVOLUTION BLOCK 1
                self.convblock2 = nn.Sequential(
                    nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3)),
                    nn.ReLU(),
                    nn.BatchNorm2d(8),
                    nn.Dropout(0.05)
                ) # output_size = 24

                # TRANSITION BLOCK 1
                self.convblock3 = nn.Sequential(
                    nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3)),
                    nn.ReLU(),
                    nn.BatchNorm2d(12),
                    nn.Dropout(0.05)
                ) # output_size = 22
                self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11

                # CONVOLUTION BLOCK 2
                self.convblock4 = nn.Sequential(
                    nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3)),
                    nn.ReLU(),            
                    nn.BatchNorm2d(12),
                    nn.Dropout(0.05)
                ) # output_size = 9

                self.convblock5 = nn.Sequential(
                    nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3)),
                    nn.ReLU(),            
                    nn.BatchNorm2d(16),
                    nn.Dropout(0.05)
                ) # output_size = 7

                self.convblock6 = nn.Sequential(
                    nn.Conv2d(in_channels=16, out_channels=20, kernel_size=(3, 3)),
                    nn.ReLU(),            
                    nn.BatchNorm2d(20),
                    nn.Dropout(0.05)
                ) # output_size = 5
                
                # OUTPUT BLOCK
                self.gap = nn.Sequential(
                    nn.AvgPool2d(kernel_size=5)
                ) # output_size = 1

                self.convblock7 = nn.Sequential(
                    nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1)),
                )

            def forward(self, x):
                x = self.convblock1(x)
                x = self.convblock2(x)
                x = self.convblock3(x)
                x = self.pool1(x)
                x = self.convblock4(x)
                x = self.convblock5(x)
                x = self.convblock6(x)
                x = self.gap(x)        
                x = self.convblock7(x)
                x = x.view(-1, 10)
                return F.log_softmax(x, dim=-1)
    else:
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

                # Input Block
                self.convblock1 = nn.Sequential(
                    nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3)),
                    nn.ReLU(),
                    nn.GroupNorm(4,8),
                    nn.Dropout(0.05)
                ) # output_size = 26

                # CONVOLUTION BLOCK 1
                self.convblock2 = nn.Sequential(
                    nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3)),
                    nn.ReLU(),
                    nn.GroupNorm(4,8),
                    nn.Dropout(0.05)
                ) # output_size = 24

                # TRANSITION BLOCK 1
                self.convblock3 = nn.Sequential(
                    nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3)),
                    nn.ReLU(),
                    nn.GroupNorm(4,12),
                    nn.Dropout(0.05)
                ) # output_size = 22
                self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11

                # CONVOLUTION BLOCK 2
                self.convblock4 = nn.Sequential(
                    nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3)),
                    nn.ReLU(),            
                    nn.GroupNorm(4,12),
                    nn.Dropout(0.05)
                ) # output_size = 9

                self.convblock5 = nn.Sequential(
                    nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3)),
                    nn.ReLU(),            
                    nn.GroupNorm(4,16),
                    nn.Dropout(0.05)
                ) # output_size = 7

                self.convblock6 = nn.Sequential(
                    nn.Conv2d(in_channels=16, out_channels=20, kernel_size=(3, 3)),
                    nn.ReLU(),            
                    nn.GroupNorm(4,20),
                    nn.Dropout(0.05)
                ) # output_size = 5
                
                # OUTPUT BLOCK
                self.gap = nn.Sequential(
                    nn.AvgPool2d(kernel_size=5)
                ) # output_size = 1

                self.convblock7 = nn.Sequential(
                    nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1)),
                )

            def forward(self, x):
                x = self.convblock1(x)
                x = self.convblock2(x)
                x = self.convblock3(x)
                x = self.pool1(x)
                x = self.convblock4(x)
                x = self.convblock5(x)
                x = self.convblock6(x)
                x = self.gap(x)        
                x = self.convblock7(x)
                x = x.view(-1, 10)
                return F.log_softmax(x, dim=-1)

    model = Net().to(device)
    summary(model, input_size=(1, 28, 28))


    def group_train(model, device, train_loader, optimizer, epoch):
        model.train()
        correct = 0
        processed = 0
        lambda_l1 = 0.0001
        total_loss = 0
        l1 = 0
        loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            l1 = 0
            for p in model.parameters():
                l1 = l1 + p.abs().sum()
            loss = loss + lambda_l1 * l1
            total_loss += loss
            loss.backward()
            optimizer.step()
            predictions = output.argmax(dim=1, keepdim=True)
            correct += predictions.eq(target.view_as(predictions)).sum().item()
            processed += len(data)
        training_losses_group.append(total_loss)
        training_accuracy_group.append(100*correct/processed)
        print('Loss = ', loss)
        print('L1 = ', l1)
        print('Train set: Accuracy={:0.1f}'.format(100*correct/processed))


    def layer_train(model, device, train_loader, optimizer, epoch):
        model.train()
        correct = 0
        processed = 0
        total_loss = 0
        loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            total_loss += loss
            loss.backward()
            optimizer.step()
            predictions = output.argmax(dim=1, keepdim=True)
            correct += predictions.eq(target.view_as(predictions)).sum().item()
            processed += len(data)
        training_losses_layer.append(loss)
        training_accuracy_layer.append(100*correct/processed)
        print('Train set: Accuracy={:0.1f}'.format(100*correct/processed))
        print('Loss - ', loss)

    def batch_train(model, device, train_loader, optimizer, epoch):
        model.train()
        correct = 0
        processed = 0
        lambda_l1 = 0.0001
        total_loss = 0
        l1 = 0
        loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            l1 = 0
            for p in model.parameters():
              l1 = l1 + p.abs().sum()
            loss = loss + lambda_l1 * l1
            total_loss += loss
            loss.backward()
            optimizer.step()
            predictions = output.argmax(dim=1, keepdim=True)
            correct += predictions.eq(target.view_as(predictions)).sum().item()
            processed += len(data)
            #pbar.set_description(desc= f'Train set: Accuracy={100*correct/processed:0.1f}')
        training_losses_batch.append(total_loss)
        training_accuracy_batch.append(100*correct/processed)
        print('Train set: Accuracy={:0.1f}'.format(100*correct/processed))
        print('Loss = ', total_loss)
        print('L1 = ', l1)

    def group_test(model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        test_loss = 0.0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                predictions = output.argmax(dim=1, keepdim=True)
                correct += predictions.eq(target.view_as(predictions)).sum().item()

        test_loss /= len(test_loader.dataset)
        testing_losses_group.append(test_loss)

        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        testing_accuracy_group.append(100. * correct / len(test_loader.dataset))


    def layer_test(model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        test_loss = 0.0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                predictions = output.argmax(dim=1, keepdim=True)
                correct += predictions.eq(target.view_as(predictions)).sum().item()

        test_loss /= len(test_loader.dataset)
        testing_losses_layer.append(test_loss)

        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        testing_accuracy_layer.append(100. * correct / len(test_loader.dataset))

    def batch_test(model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        test_loss = 0.0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                predictions = output.argmax(dim=1, keepdim=True)
                correct += predictions.eq(target.view_as(predictions)).sum().item()

        test_loss /= len(test_loader.dataset)
        testing_losses_batch.append(test_loss)

        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        testing_accuracy_batch.append(100. * correct / len(test_loader.dataset))

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    if batch or layer:
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.001)

    scheduler = StepLR(optimizer, step_size=6, gamma=0.1)

    for epoch in range(1, 20):
        print("EPOCH:", epoch, "LR = ", scheduler.get_lr())
        if batch:
            batch_train(model, device, train_loader, optimizer, epoch)
            batch_test(model, device, test_loader)
        elif group:
            group_train(model, device, train_loader, optimizer, epoch)
            group_test(model, device, test_loader)
        elif layer:
            layer_train(model, device, train_loader, optimizer, epoch)
            layer_test(model, device, test_loader)
        scheduler.step()

    model.eval()
    test_loss = 0.0
    correct = 0
    im_pred = {'Wrong': []}
    i = 1
    plt_dt = dict()
    with torch.no_grad():
        for data, target in test_loader:
            if (len(im_pred['Wrong'])<20):
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                i+=1
                plt_dt['Input'], plt_dt['target'], plt_dt['pred'] = data.to('cpu'), target.to('cpu'), pred.to('cpu').view(-1,)

                for id in range(len(data)):
                    if plt_dt['target'][id] != plt_dt['pred'][id]:
                        im_pred['Wrong'] = im_pred['Wrong']+ [{'Image':data[id],'pred':pred[id],'actual' : target[id]}]
   
    plt.figure(figsize=(14,14)) 
    for i in range(20):
        plt.subplot(4,5,i+1)  
        pixels = np.array(im_pred['Wrong'][i]['Image'].cpu() , dtype='uint8')

        # Reshape the array into 28 x 28 array (2-dimensional array)
        pixels = pixels.reshape((28, 28))

        # Plot
        plt.title('Actual Value is {label}\n Predicted Value is {pred}'.format(label=im_pred['Wrong'][i]['actual'].cpu(), pred =im_pred['Wrong'][i]['pred'].cpu()[0]), color='r')
        plt.imshow(pixels, cmap='gray')

    plt.show()

if __name__ == '__main__':
    normalize_3_way('layer')