from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import datasets,transforms
import torchmetrics
import lightning as L


# define the model

class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,10),
        )
    def forward(self,x):
        x = x.flatten(1)
        return self.net(x)

# define lightning model 

class LightningModel(L.LightningModule):
    def __init__(self,model,lr):
        super().__init__()
        self.train_acc = torchmetrics.Accuracy(task='multiclass',num_classes=10)
        self.val_acc = torchmetrics.Accuracy(task='multiclass',num_classes=10)
        self.test_acc = torchmetrics.Accuracy(task='multiclass',num_classes=10)
        self.model = model
        self.lr = lr
    def forward(self,x):
        return self.model(x)
    def shared_step(self,batch):
        images,labels = batch
        logits = self(images)
        loss=F.cross_entropy(logits,labels)
        preds = logits.argmax(dim=1)
        return labels,loss,preds
    def training_step(self,batch,batch_idx):
        labels,loss,preds = self.shared_step(batch)
        self.train_acc(preds,labels)
        self.log('train_acc',self.train_acc,prog_bar=True,on_step=False,on_epoch=True)
        self.log('training_loss',loss,prog_bar=True,on_step=False,on_epoch=True)
        return loss
    def validation_step(self,batch,batch_idx):
        labels,loss,preds = self.shared_step(batch)
        self.val_acc(preds,labels)
        self.log('validation_loss',loss,prog_bar=True)
        self.log('val_acc',self.val_acc,prog_bar=True,on_step=False,on_epoch=True)
    def test_step(self,batch,batch_idx):
        labels,_,preds = self.shared_step(batch)
        self.test_acc(preds,labels)
        self.log('test_acc',self.test_acc)
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(),lr=self.lr)
        return opt
    
class MNISTDataModule(L.LightningDataModule):
    def __init__(self,data_dir='./mnist',batch_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
    def prepare_data(self):
        # only happens once in case of multiple gpus
        datasets.MNIST(self.data_dir,train=True,download=True)
        datasets.MNIST(self.data_dir,train=False,download=True)
    def setup(self,stage: str):
        # happens parallely across gpus
        ds = datasets.MNIST(root=self.data_dir,train=True,download=False,transform=transforms.ToTensor())
        self.train_ds,self.val_ds = random_split(ds,lengths=[55000,5000],generator=torch.Generator().manual_seed(42))
        self.test_ds = datasets.MNIST(root=self.data_dir,train=False,download=False,transform=transforms.ToTensor())
        self.predict_ds = datasets.MNIST(root=self.data_dir,train=False,download=False,transform=transforms.ToTensor())
    # functions that return dataloaders
    def train_dataloader(self):
        return DataLoader(self.train_ds,self.batch_size,shuffle=True,drop_last=True)
    def test_dataloader(self):
        return DataLoader(self.test_ds,self.batch_size,shuffle=False)
    def val_dataloader(self):
        return DataLoader(self.val_ds,self.batch_size,shuffle=False)
    def predict_dataloader(self):
        return DataLoader(self.predict_ds,self.batch_size,shuffle=False)
    