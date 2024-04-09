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
        self.log('validation_acc',self.val_acc,prog_bar=True,on_step=False,on_epoch=True)
    def test_step(self,batch,batch_idx):
        labels,_,preds = self.shared_step(batch)
        self.test_acc(preds,labels)
        self.log('accuracy',self.test_acc)
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(),lr=self.lr)
        return opt
    

if __name__=='__main__':
    # load datasets

    train_ds = datasets.MNIST(root='./mnist',train=True,transform=transforms.ToTensor())
    test_ds = datasets.MNIST(root='./mnist',train=False,transform=transforms.ToTensor())

    train_ds,val_ds =  random_split(train_ds,lengths=[55000,5000])

    # define dataloaders
    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        num_workers=4,
        batch_size=128,
        drop_last=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        batch_size=128,
        num_workers=4,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_ds,
        shuffle=False,
        batch_size=128
    )

    # instantiate the models and define the trainer and begin training

    model = MNISTClassifier()
    lightning_model = LightningModel(model=model,lr=1e-3)

    trainer = L.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=10,
    )

    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # Testing the model

    train_acc = trainer.test(dataloaders=train_loader)[0]['accuracy']
    val_acc = trainer.test(dataloaders=val_loader)[0]['accuracy']
    test_acc = trainer.test(dataloaders=test_loader)[0]['accuracy']

    print(f"Training Accuracy : {train_acc} | Validation Accuracy : {val_acc} | Testing Accuracy : {test_acc}")

    # save the model

    MODEL_PATH = 'mnist-lightning.pt'
    torch.save(lightning_model.state_dict(),MODEL_PATH)