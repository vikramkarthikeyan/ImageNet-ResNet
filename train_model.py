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

parser = argparse.ArgumentParser(description='Tiny ImageNet Model Training...')
parser.add_argument('--checkpoint', default=None, type=str, help='Checkpoint model file if stopped abruptly before')

args = parser.parse_args()

# Hyperparameters
LR = 0.00001  
SGD_MOMENTUM = 0.9
WEIGHT_DECAY = 0.00001

if __name__ == "__main__":

    print("\n\nImporting the data and setting up data loaders...")
    trainer = Trainer()

    print("\nInitializing the CNN model...")
    # model = base_model.Base_CNN()
    model = resnet18.resnet18() 

    print("\nChecking if a GPU is available...")
    use_gpu = torch.cuda.is_available()
    # Initialize new model
    if use_gpu:
        model = model.cuda()
        print ("Using GPU")
    else:
        print ("Using CPU as GPU is unavailable")    

    print("\nModel Summary...")
    summary(model, (3, 64, 64))

    # Define loss function and optimizer for CNN
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=SGD_MOMENTUM, weight_decay=WEIGHT_DECAY)

    highest_accuracy = 0
    start_epochs = 0
    total_epochs = 10


    # If checkpoint is available, load model from checkpoint
    if args.checkpoint:
        model_file = args.checkpoint
        
        if not os.path.isfile(model_file):
            print("Invalid checkpoint file...")
            exit()
        
        if use_gpu:
            checkpoint = torch.load(model_file)
        else:
            checkpoint = torch.load(model_file, map_location='cpu')

        model = model.load_state_dict(checkpoint['state_dict'])
        optimizer = optimizer.load_state_dict(checkpoint['optimizer'])
        highest_accuracy = checkpoint['best_accuracy']
        start_epochs = checkpoint['epoch']

    print("\nInitiating training...")
    for epoch in range(start_epochs, total_epochs):

        print("-------------------------------------------------------")
        print("Epoch: ", epoch)
        print("-------------------")
        # Train for one epoch
        trainer.train(model, criterion, optimizer, epoch, use_gpu)

        print("\nTraining Done...\n\nPerform Validation...")
        # Evaluate on the validation set
        accuracy = trainer.validate(model, criterion, epoch, use_gpu)

        # Checkpointing the model after every epoch
        trainer.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_accuracy': accuracy,
                    'optimizer' : optimizer.state_dict(),
                })
        
        # If this epoch's model proves to be the best till now, save it as best model
        if accuracy == max(accuracy, highest_accuracy):
            highest_accuracy = accuracy
            trainer.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_accuracy': highest_accuracy,
                    'optimizer' : optimizer.state_dict()
            },'./models/best_model.pth.tar')

    
    print("Training complete...")
    print("Best accuracy: ", highest_accuracy)



        
