import torchvision
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from PIL import Image
import os.path
import torch
from torchvision import models, transforms
import numpy as np
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from swin_transformer import SwinTransformer

def GetFileFromThisRootDir(dir,ext = None):  
  allfiles = []  
  needExtFilter = (ext != None)  
  for root,dirs,files in os.walk(dir):  
    for filespath in files:  
      filepath = os.path.join(root, filespath)  
      extension = os.path.splitext(filepath)[1][1:]  
      if needExtFilter and extension in ext:  
        allfiles.append(filepath)  
      elif not needExtFilter:  
        allfiles.append(filepath)  
  # print(allfiles)
  return allfiles 


class Mydata(Dataset):
    def __init__(self,image,label,transform):
        self.image = image
        self.label = label
        self.transform = transform
        
    def __getitem__(self, idx):
        img = Image.open(self.image[idx])
        label = self.label[idx]
        img = self.transform(img)
        return img,torch.tensor(label)
    
    def __len__(self):
        return len(self.image)

def deal(data_path,batch=32,train_rate=0.8):
    
#读取数据，将数据的路径全部读取出来，然后进行验证集和训练集的划分
    data_num = 2100    
    all_ind = np.array(range(data_num))
    # data_path = ""
    image_all = GetFileFromThisRootDir(data_path)
    image_all = np.array(image_all)
    label = []
    for i in range(data_num):
        label.append(int(i/100))
    label = np.array(label)
    np.random.seed(0)
    np.random.shuffle(all_ind)
    #图像预处理
    transform1 = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])
    

    
    #划分训练、验证
    train_ind = all_ind[:int(train_rate*data_num)]
    test_ind = all_ind[int(train_rate*data_num) : ]
    test_num = test_ind.shape[0]
    
    train_data = image_all[train_ind] #image是列表，train_ind是数组
    test_data = image_all[test_ind]  #image是列表，train_ind是数组
    
    train_label = label[train_ind]
    test_label = label[test_ind]
    #构造dataset
    dataset = {"train":Mydata(train_data,train_label,transform1),
                    "test":Mydata(test_data,test_label,transform1)}
    
    dataloader = {'train':DataLoader(dataset["train"],
                                     batch_size = batch,
                                     shuffle=True),
                  'test':DataLoader(dataset["test"],
                                     batch_size = batch,
                                     shuffle=True)
                  }
    
    return dataloader





#data_path = "/media/yr/新加卷1/ly/SAR/00Original/"
#batch = 32
#dataload = deal(data_path,batch)
#for batch_data in dataload ["train"]:
 #   print(batch_data[0].shape)
  #  print(batch_data[1].shape)

