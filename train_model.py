import matplotlib.pyplot as plt
import time
import numpy as np
import shutil
import os
import argparse
import cv2
import torch
import torch.nn as nn
import base_model
import resnet18

from torchvision import datasets, transforms
from torchvision import models
from torch.autograd import Variable
from torchsummary import summary
from trainer import AverageMeter
from trainer import Trainer
from EarlyStopping import EarlyStopper


parser = argparse.ArgumentParser(description='Tiny ImageNet Model Training...')
parser.add_argument('--checkpoint', default=None, type=str, help='Checkpoint model file if stopped abruptly before')
parser.add_argument('--epochs', default=None, type=int, help="Number of training epochs")

args = parser.parse_args()

# Hyperparameters
LR = 0.01
SGD_MOMENTUM = 0.9
WEIGHT_DECAY = 0.00001
EPOCHS = 1

class TrainingOrchestrator:
    def __init__(self):
        print("\n\nImporting the data and setting up data loaders...")
        self.trainer = Trainer()

        print("\nInitializing the CNN model...")
        # model = base_model.Base_CNN()
        self.model = resnet18.resnet18() 

        print("\nChecking if a GPU is available...")
        self.use_gpu = torch.cuda.is_available()
        # Initialize new model
        if self.use_gpu:
            self.model = self.model.cuda()
            print ("Using GPU")
        else:
            print ("Using CPU as GPU is unavailable")    

        summary(self.model, (3, 64, 64))

        print("\nInitializing Early Stopper...")
        self.early_stopper = EarlyStopper()

        # Define loss function and optimizer for CNN
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=LR, momentum=SGD_MOMENTUM, weight_decay=WEIGHT_DECAY)

        self.highest_accuracy = 0
        self.highest_accuracy_5 = 0
        self.start_epochs = 0
        self.total_epochs = EPOCHS

    def check_custom_epochs(self):
        # If number of epochs is specified
        if args.epochs:
            self.total_epochs = args.epochs
    
    def check_if_model_checkpoint_available(self):
        # If checkpoint is available, load model from checkpoint
        if args.checkpoint:
            model_file = args.checkpoint
            
            if not os.path.isfile(model_file):
                print("Invalid checkpoint file...")
                exit()
            
            if self.use_gpu:
                checkpoint = torch.load(model_file)
            else:
                checkpoint = torch.load(model_file, map_location='cpu')

            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.highest_accuracy = checkpoint['best_accuracy']
            self.start_epochs = checkpoint['epoch']
    
    def set_lr_scheduler(self):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=3, verbose=True)

    def orchestrate_training(self):
        print("\nInitiating training...")

        for epoch in range(self.start_epochs, self.total_epochs):

            print("Epoch: ", epoch)
            print("-------------------")
            # Train for one epoch
            self.trainer.train(self.model, self.criterion, self.optimizer, epoch, self.use_gpu)

            # Evaluate on the validation set
            accuracy, accuracy_5, val_loss = self.trainer.validate(self.model, self.criterion, epoch, self.use_gpu)

            # Checkpointing the model after every epoch
            self.trainer.save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': self.model.state_dict(),
                        'best_accuracy': accuracy,
                        'optimizer' : self.optimizer.state_dict(),
                    })
            
            # If this epoch's model proves to be the best till now, save it as best model
            if accuracy == max(accuracy, self.highest_accuracy):
                self.highest_accuracy = accuracy
                self.highest_accuracy_5 = accuracy_5
                self.trainer.save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': self.model.state_dict(),
                        'best_accuracy': self.highest_accuracy,
                        'optimizer' : self.optimizer.state_dict()
                },'./models/best_model.pth.tar')
            
            # Reducing LR on Plateau
            self.scheduler.step(val_loss)

            # Check early stopping
            self.early_stopper.check_loss_history(val_loss)
            if self.early_stopper.stop:
                print("Stopping training...")
                break
        
        print("Training complete...")
        print("Best accuracy@1: {0}, accuracy@5: {1}".format(self.highest_accuracy.item(), self.highest_accuracy_5.item()))




if __name__ == "__main__":

    orchestrator = TrainingOrchestrator()
    
    orchestrator.check_custom_epochs()
    orchestrator.check_if_model_checkpoint_available()
    orchestrator.set_lr_scheduler()
    orchestrator.orchestrate_training()
    
    



        
