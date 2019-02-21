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

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

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

        train_root = os.path.join(data, 'train')  # this is path to training images folder
        validation_root = os.path.join(data, 'val/images')  # this is path to validation images folder

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Perform Data Augmentation by Randomly Flipping Training Images
        training_data = datasets.ImageFolder(train_root,
                                        transform=transforms.Compose([transforms.RandomResizedCrop(224),
                                                                        transforms.RandomHorizontalFlip(),
                                                                        transforms.ToTensor(),
                                                                        normalize]))
        # Resize Validation Images
        validation_data = datasets.ImageFolder(validation_root,
                                            transform=transforms.Compose([transforms.Resize(256),
                                                                            transforms.CenterCrop(224),
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
    
    def train(self, model, criterion, optimizer, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        # switch to train mode
        model.train()

        end = time.time()
        for i, (input, target) in enumerate(self.train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            # if args.gpu is not None:
            #     input = input.cuda(args.gpu, non_blocking=True)
            # target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            print input.shape
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc[0], input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            print batch_time.val
            print losses.val
            print top1.val

            # print('Epoch: [{0}][{1}/{2}]\t'
            #         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            #         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #         'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
            #         epoch, i, len(self.train_loader), batch_time=batch_time,
            #         data_time=data_time, loss=losses, top1=top1))


    def validate(self, model, criterion):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        # switch to evaluate mode
        model.eval()

        with torch.no_grad():
            end = time.time()
            for i, (input, target) in enumerate(self.validation_loader):
                # if args.gpu is not None:
                #     input = input.cuda(args.gpu, non_blocking=True)
                # target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                output = model(input)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(acc1[0], input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()


                print('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        i, len(self.validation_loader), batch_time=batch_time, loss=losses,
                        top1=top1))

            print(' * Acc@1 {top1.avg:.3f}'
                .format(top1=top1))

        return top1.avg


    def save_checkpoint(self, state, filename='checkpoint.pth.tar'):
        torch.save(state, filename)

LR = 0.001  

if __name__ == "__main__":
    print "\n\nImporting the data and setting up data loaders..."
    trainer = Trainer()

    print "Initializing the CNN model..."
    model = base_model.Base_CNN()

    summary(model, (3, 224, 224))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    highest_accuracy = 0

    print "Initiating training..."
    for epoch in range(0, 10):

        print "-------------------------------------------"
        print "Epoch: ", epoch
        # Train for one epoch
        trainer.train(model, criterion, optimizer, epoch)

        # Evaluate on the validation set
        accuracy = trainer.validate(model, criterion)

        print accuracy

        highest_accuracy = max(accuracy, highest_accuracy)

    trainer.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': highest_accuracy,
                'optimizer' : optimizer.state_dict(),
            })



        