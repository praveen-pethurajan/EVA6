'''Some helper functions for PyTorch, including:
    - generate transforms
    - load dataset
    - plot graphs
    - identify misclassified images
    - plot images
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

import albumentations as A


class CIFAR_10_Dataset(torch.utils.data.Dataset):
  def __init__(self,  dataset, transformer=None):
        self.dataset = dataset
        self.transforms = transformer
  def __len__(self):
        return len(self.dataset)
  def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img, target = self.dataset[idx]
        img = img.cpu().detach().numpy()
        img = np.asarray(img).reshape((32,32,3))
        if self.transforms is not None:
            image = self.transforms(image=img)
        img = torch.from_numpy(img.reshape(3,32,32))
        return img, target
  
class TinyImageNetDataLoader:

    def __init__(self):
        self.augmentation = 'TinyImageNetAlbumentation'
        
    def calculate_mean_std(self):
        mean,std = (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)
        return mean, std

    def classes(self):
        id_dict = {}
        all_classes = {}
        for i, line in enumerate(open( 'data/tiny-imagenet-200/wnids.txt', 'r')):
            id_dict[line.replace('\n', '')] = i
        
        result = {}
        class_id={}
        for i, line in enumerate(open( 'data/tiny-imagenet-200/words.txt', 'r')):
            n_id, word = line.split('\t')[:2]
            all_classes[n_id] = word
        for key, value in id_dict.items():
            result[value] = (all_classes[key].replace('\n','').split(",")[0])
            class_id[key] = (value,all_classes[key])
            
        return result,class_id  
        
    def get_dataloader(self): 
        
        tinyimagenet_albumentation = eval(self.augmentation)()
        mean,std = self.calculate_mean_std()
        
        train_transforms, test_transforms = tinyimagenet_albumentation.train_transform(mean,std),tinyimagenet_albumentation.test_transform(mean,std)
                                                                              
        trainset = TinyImageNet(root='./data', train=True,download=True, transform=train_transforms) 
            
        testset  = TinyImageNet(root='./data', train=False,transform=test_transforms)

        self.train_loader = torch.utils.data.DataLoader(trainset, 
                                                      batch_size=128, 
                                                      shuffle=True,
                                                      num_workers=4, 
                                                      pin_memory=True)
        self.test_loader = torch.utils.data.DataLoader(testset, 
                                                     batch_size=32,  
                                                     shuffle=False,
                                                     num_workers=4, 
                                                     pin_memory=True)
        return self.train_loader,self.test_loader


class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None,  download=False,train_split=0.7):
        
        self.root = root
        self.transform = transform
        self.data_dir = 'tiny-imagenet-200'

        if download and (not os.path.isdir(os.path.join(self.root, self.data_dir))):
            self.download()

        self.image_paths = []
        self.targets = []

        _,class_id = TinyImageNetDataLoader().classes()
        
        # train images
        train_path = os.path.join(self.root, self.data_dir, 'train')
        for class_dir in os.listdir(train_path):
            train_images_path = os.path.join(train_path, class_dir, 'images')
            for image in os.listdir(train_images_path):
                if image.endswith('.JPEG'):
                    self.image_paths.append(os.path.join(train_images_path, image))
                    self.targets.append(class_id[class_dir][0])

        # val images
        val_path = os.path.join(self.root, self.data_dir, 'val')
        val_images_path = os.path.join(val_path, 'images')
        with open(os.path.join(val_path, 'val_annotations.txt')) as f:
            for line in csv.reader(f, delimiter='\t'):
                self.image_paths.append(os.path.join(val_images_path, line[0]))
                self.targets.append(class_id[line[1]][0])
                
        self.indices = np.arange(len(self.targets))

        random_seed=1
        np.random.seed(random_seed)
        np.random.shuffle(self.indices)

        split_idx = int(len(self.indices) * train_split)
        self.indices = self.indices[:split_idx] if train else self.indices[split_idx:]

    def download(self):
        if (os.path.isdir("./data/tiny-imagenet-200")):
            print('Images already downloaded...')
            return
        r = requests.get('http://cs231n.stanford.edu/tiny-imagenet-200.zip', stream=True)
        print('Downloading TinyImageNet Data')
        zip_ref = zipfile.ZipFile(BytesIO(r.content))
        for file in notebook.tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist())):
            zip_ref.extract(member=file, path='./data/')
        zip_ref.close()


    def __getitem__(self, index):
        
        image_index = self.indices[index] 
        filepath = self.image_paths[image_index]
        img = Image.open(filepath)
        img = img.convert("RGB")
        target = self.targets[image_index]

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.indices)


def train_transform(train):
  albumentation_train_list = []
  train_list = []
  if "totensor" in train:
    train_list.append(transforms.ToTensor())
  if "normalize_normal" in train:
    train_list.append(transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)))
  if "normalize_mean" in train:
    train_list.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
  if "randomcrop" in train:
    train_list.append(transforms.RandomCrop(32, padding=4))
  if "horizontal_flip" in train:
    train_list.append(transforms.RandomHorizontalFlip())
  if "random_rotate" in train:
    train_list.append(transforms.RandomRotation((-5.0, 5.0), fill=(0,0,0)))
  if "cutout" in train:
    albumentation_train_list.append(A.CoarseDropout(p=0.5, max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=16, min_width=16, fill_value=(0.4914, 0.4822, 0.4465), mask_fill_value = None))
  if "shift_scale_rotate" in train:
     albumentation_train_list.append(A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5))
  if "grayscale" in train:
     albumentation_train_list.append(A.ToGray(p=0.5))
  
  return transforms.Compose(train_list), A.Compose(albumentation_train_list)



def load_dataset(tensor_train, numpy_train):
  train_dataset = CIFAR_10_Dataset(torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                  transform=tensor_train), numpy_train)

  testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transforms.Compose([transforms.ToTensor(),
                                                                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                                                                   ]))
  return train_dataset, testset


def plot_graph(tr_l, tr_a, te_l, te_a):
  fig, axs = plt.subplots(2,2,figsize=(15,10))
  axs[0, 0].plot(tr_l)
  axs[0, 0].set_title("Training Loss")
  axs[1, 0].plot(tr_a)
  axs[1, 0].set_title("Training Accuracy")
  axs[0, 1].plot(te_l)
  axs[0, 1].set_title("Test Loss")
  axs[1, 1].plot(te_a)
  axs[1, 1].set_title("Test Accuracy")


def denormalize(tensor, mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]):
    single_img = False
    if tensor.ndimension() == 3:
      single_img = True
      tensor = tensor[None,:,:,:]

    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    ret = tensor.mul(std).add(mean)
    return ret[0] if single_img else ret

  
def identify_images(net, criterion, device, testloader, n):
    net.eval()
    correct_images = []
    incorrect_images = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)           
            predicted = outputs.argmax(dim=1, keepdim=True)
            is_correct = predicted.eq(targets.view_as(predicted))
            
            misclassified_inds = (is_correct==0).nonzero()[:,0]
            for mis_ind in misclassified_inds:
              if len(incorrect_images) == n:
                break
              incorrect_images.append({
                  "target": targets[mis_ind].cpu().numpy(),
                  "pred": predicted[mis_ind][0].cpu().numpy(),
                  "img": inputs[mis_ind]
              })

            correct_inds = (is_correct==1).nonzero()[:,0]
            for ind in correct_inds:
              if len(correct_images) == n:
                break
              correct_images.append({
                  "target": targets[ind].cpu().numpy(),
                  "pred": predicted[ind][0].cpu().numpy(),
                  "img": inputs[ind]
              })
    return correct_images, incorrect_images
  
  
def plot_images(img_data, classes):
    figure = plt.figure(figsize=(10, 10))

    num_of_images = len(img_data)
    for index in range(1, num_of_images + 1):
        img = denormalize(img_data[index-1]["img"])
        plt.subplot(5, 5, index)
        plt.axis('off')
        img = img.cpu().numpy()
        maxValue = np.amax(img)
        minValue = np.amin(img)
        img = np.clip(img, 0, 1)
        img = img/np.amax(img)
        img = np.clip(img, 0, 1)
        plt.imshow(np.transpose(img, (1, 2, 0)))
        plt.title("Predicted: %s\nActual: %s" % (classes[img_data[index-1]["pred"]], classes[img_data[index-1]["target"]]))

    plt.tight_layout()
