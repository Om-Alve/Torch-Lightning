import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import datasets,transforms
import torchmetrics
import lightning as L
from utilities import MNISTClassifier,LightningModel,MNISTDataModule
    

if __name__=='__main__':
    
    # Instantiate the Lightning DataModule

    dm = MNISTDataModule()

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
        datamodule=dm,
    )

    # Testing the model

    train_acc = trainer.validate(dataloaders=dm.train_dataloader())[0]['val_acc']
    val_acc = trainer.validate(datamodule=dm)[0]['val_acc']
    test_acc = trainer.test(datamodule=dm)[0]['test_acc']

    print(f"Training Accuracy : {train_acc} | Validation Accuracy : {val_acc} | Testing Accuracy : {test_acc}")

    # save the model

    MODEL_PATH = 'mnist-lightning.pt'
    torch.save(lightning_model.state_dict(),MODEL_PATH)