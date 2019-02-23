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
from EarlyStopping import EarlyStopper
from AverageMeter import AverageMeter


# Followed PyTorch's ImageNet documentation as Tiny ImageNet has just fewer classes
# https://github.com/pytorch/examples/blob/master/imagenet/main.py
class Trainer:

    def __init__(self, training_batch_size=256, validation_batch_size=10, data="./tiny-imagenet-200/"):

        # Data loaders are written to reduce the number of images stored in-memory
        self.train_batch_size = training_batch_size 
        self.validation_batch_size = validation_batch_size

        training_path = os.path.join(data, 'train')
        validation_path = os.path.join(data, 'val/images') 

        # mean and std values acquired from ImageNet statistics
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Perform Data Augmentation by Randomly Flipping Training Images
        training_data = datasets.ImageFolder(training_path,
                                        transform=transforms.Compose([#transforms.RandomResizedCrop(224),
                                                                        transforms.RandomHorizontalFlip(),
                                                                        transforms.ToTensor(),
                                                                        normalize]))
        # Get Validation Images
        validation_data = datasets.ImageFolder(validation_path,
                                            transform=transforms.Compose([#transforms.Resize(256),
                                                                            #transforms.CenterCrop(224),
                                                                            transforms.ToTensor(),
                                                                            normalize]))
         # Create training dataloader
        self.train_loader = torch.utils.data.DataLoader(training_data, batch_size=self.train_batch_size, shuffle=True,
                                                             num_workers=5)
        # Create validation dataloader
        self.validation_loader = torch.utils.data.DataLoader(validation_data,
                                                                  batch_size=self.validation_batch_size,
                                                                  shuffle=False, num_workers=5)

        self.class_names = training_data.classes
        self.num_classes = len(training_data.classes)
        self.early_stopper = EarlyStopper()

        self.validation_accuracy = list()
    
    def train(self, model, criterion, optimizer, epoch, usegpu):
        batch_time = AverageMeter()
        losses = AverageMeter()

        # switch to train mode
        model.train()

        end = time.time()

        for i, (data, target) in enumerate(self.train_loader):

            data, target = Variable(data), Variable(target, requires_grad=False)

            if usegpu:
                data = data.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            # Compute Model output
            output = model(data)

            # Compute Loss
            loss = criterion(output, target)

            # measure accuracy and record loss
            losses.update(loss.item(), data.size(0))

            # Clear(zero) Gradients for theta
            optimizer.zero_grad()

            # Perform BackProp wrt theta
            loss.backward()

            # Update theta
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            print('\rTraining - Epoch [{:04d}] Batch [{:04d}/{:04d}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        epoch, i, len(self.train_loader), batch_time=batch_time,
                        loss=losses), end="")


    def validate(self, model, criterion, epoch, usegpu):
        batch_time = AverageMeter()
        losses = AverageMeter()
        accuracy = AverageMeter()

        # switch to evaluate mode
        model.eval()

        validation_loss = 0
        correct_predictions = 0
        validation_size = len(self.validation_loader.dataset)

        print("\n")

        with torch.no_grad():
            end = time.time()
            for i, (data, target) in enumerate(self.validation_loader):
                correct_predictions_epoch = 0
                if usegpu:
                    data = data.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)

                # compute output
                output = model(data)
                loss = criterion(output, target)
                validation_loss += loss

                # To Measure Accuracy:
                # Step 1: get index of maximum value among output classes
                value, index = torch.max(output.data, 1) 

                # Step 2: Compute total no of correct predictions 
                for j in range(0, self.validation_batch_size):
                    if index[j] == target.data[j]:
                        correct_predictions += 1
                        correct_predictions_epoch += 1

                # Step 3: Measure accuracy and record loss
                losses.update(loss.item(), data.size(0))
                accuracy.update(100 * correct_predictions_epoch/float(self.validation_batch_size))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                print('\rValidation - Batch [{:04d}/{:04d}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Accuracy {accuracy.val} ({accuracy.avg})\t'.format(
                        i, len(self.validation_loader), batch_time=batch_time,
                        loss=losses, accuracy=accuracy), end="")

            # average loss = sum of loss over all batches/num of batches
            average_validation_loss = validation_loss / (
                validation_size / self.validation_batch_size)

            # calculate total accuracy for the current epoch
            self.validation_accuracy_epoch = 100.0 * correct_predictions / (validation_size)

            # add validation accuracy to list for visualization
            self.validation_accuracy.append(self.validation_accuracy_epoch)

            print('\n\nValidation Epoch {}: Average loss: {:.6f} \t Accuracy: {}/{} ({:.2f}%)\n'.
                  format(epoch, average_validation_loss, correct_predictions, validation_size,
                         self.validation_accuracy_epoch))

        return self.validation_accuracy_epoch, validation_loss


    def save_checkpoint(self, state, filename='./models/checkpoint.pth.tar'):
        torch.save(state, filename)
