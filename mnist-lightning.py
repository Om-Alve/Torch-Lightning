import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import datasets,transforms
import torchmetrics
import lightning as L
from utilities import MNISTClassifier,LightningModel,MNISTDataModule
from lightning.pytorch.callbacks import ModelCheckpoint

if __name__=='__main__':
    
    numepochs = 10

    # Instantiate the Lightning DataModule

    dm = MNISTDataModule()
    dm.setup(stage='fit')
    total_steps = len(dm.train_dataloader()) * numepochs

    # instantiate the models and define the trainer and begin training

    model = MNISTClassifier()
    lightning_model = LightningModel(model=model,lr=1e-3,t_max=total_steps)
    callback = ModelCheckpoint(save_top_k=1,mode='max',monitor='val_acc',save_last=True)
    trainer = L.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=numepochs,
        callbacks=[callback],
    )

    trainer.fit(
        model=lightning_model,
        datamodule=dm,
    )

    # Testing the model

    train_acc = trainer.validate(dataloaders=dm.train_dataloader(),ckpt_path='best')[0]['val_acc']
    val_acc = trainer.validate(datamodule=dm,ckpt_path='best')[0]['val_acc']
    test_acc = trainer.test(datamodule=dm,ckpt_path='best')[0]['test_acc']

    print(f"Training Accuracy : {train_acc} | Validation Accuracy : {val_acc} | Testing Accuracy : {test_acc}")

    # save the model

    MODEL_PATH = 'mnist-lightning.pt'
    torch.save(lightning_model.state_dict(),MODEL_PATH)