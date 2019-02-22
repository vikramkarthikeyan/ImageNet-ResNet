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


def generate_class_list(data, class_list):
    filename = os.path.join(data, 'words.txt')

    with open(filename, "r") as f:
        data = f.readlines()
    
    class_dict = {}

    for line in data:
        entries = line.split("\t")
        if entries[0] in class_list:
            class_dict[entries[0]] = entries[1].split(",")[0].rstrip() 

    return class_dict

# Class directly used from https://github.com/pytorch/examples/blob/master/imagenet/main.py
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Followed PyTorch's ImageNet documentation as Tiny ImageNet has just fewer classes
# https://github.com/pytorch/examples/blob/master/imagenet/main.py
class Trainer:

    def __init__(self, training_batch_size=100, validation_batch_size=10, data="./tiny-imagenet-200/"):

        # Data loaders are written to reduce the number of images stored in-memory
        self.train_batch_size = training_batch_size 
        self.validation_batch_size = validation_batch_size

        train_root = os.path.join(data, 'train')
        validation_root = os.path.join(data, 'val/images') 

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Perform Data Augmentation by Randomly Flipping Training Images
        training_data = datasets.ImageFolder(train_root,
                                        transform=transforms.Compose([#transforms.RandomResizedCrop(224),
                                                                        transforms.RandomHorizontalFlip(),
                                                                        transforms.ToTensor(),
                                                                        normalize]))
        # Resize Validation Images
        validation_data = datasets.ImageFolder(validation_root,
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
        self.tiny_class = generate_class_list(data, self.class_names)

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

            print('Epoch [{0}] Batch [{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        epoch, i, len(self.train_loader), batch_time=batch_time,
                        loss=losses))


    def validate(self, model, criterion, epoch, usegpu):
        batch_time = AverageMeter()
        losses = AverageMeter()
        accuracy = AverageMeter()

        # switch to evaluate mode
        model.eval()

        validation_loss = 0
        correct_predictions = 0

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
                    if index[j] == target.data[j]:  # if index equal to target label, record correct classification
                        correct_predictions += 1
                        correct_predictions_epoch += 1

                # Step 3: Measure accuracy and record loss
                losses.update(loss.item(), data.size(0))
                accuracy.update(100 * correct_predictions_epoch/float(self.validation_batch_size))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                print('\rBatch [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Accuracy {accuracy.val} ({accuracy.avg})\t'.format(
                        i, len(self.validation_loader), batch_time=batch_time,
                        loss=losses, accuracy=accuracy))

            # average loss = sum of loss over all batches/num of batches
            average_validation_loss = validation_loss / (
                len(self.validation_loader.dataset) / self.validation_batch_size)

            # calculate total accuracy for the current epoch
            self.validation_accuracy_cur_epoch = 100.0 * correct_predictions / (len(self.validation_loader.dataset))
            # add accuracy for current epoch to list
            self.validation_accuracy.append(self.validation_accuracy_cur_epoch)

            print('\nValidation Epoch {}: Average loss: {:.6f} \t Accuracy: {}/{} ({:.2f}%)\n'.
                  format(epoch, average_validation_loss, correct_predictions, len(self.validation_loader.dataset),
                         self.validation_accuracy_cur_epoch))

        return self.validation_accuracy_cur_epoch


    def save_checkpoint(self, state, filename='./models/checkpoint.pth.tar'):
        torch.save(state, filename)
