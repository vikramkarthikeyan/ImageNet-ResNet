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

from torchvision import datasets, transforms
from torchvision import models
from torch.autograd import Variable
from torchsummary import summary
from trainer import AverageMeter
from trainer import Trainer

parser = argparse.ArgumentParser(description='Tiny ImageNet Model Training...')

parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

# Hyperparameters
LR = 0.001  

if __name__ == "__main__":
        
    print "\n\nImporting the data and setting up data loaders..."
    trainer = Trainer()

    print "\nInitializing the CNN model..."
    model = base_model.Base_CNN()

    print "\nChecking if a GPU is available..."
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()
        print ('Using GPU')
    else:
        print ('Using CPU as GPU is unavailable')

    print "\nModel Summary..."
    summary(model, (3, 64, 64))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    highest_accuracy = 0

    print "Initiating training..."
    for epoch in range(0, 2):

        print "-------------------------------------------------------"
        print "Epoch: ", epoch
        print "-------------------"
        # Train for one epoch
        trainer.train(model, criterion, optimizer, epoch, use_gpu)

        # Evaluate on the validation set
        accuracy = trainer.validate(model, criterion, epoch, use_gpu)

        print accuracy

        highest_accuracy = max(accuracy, highest_accuracy)

    # Saving the model
    trainer.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': highest_accuracy,
                'optimizer' : optimizer.state_dict(),
            })



        