
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--frames_path',         	   type=str,                default='/video_frames/data/', 				     help='dirctory for video frames, anotation file need to be in YOLO format')
parser.add_argument('--Eval_mode',           	   type=str,                default='TEE',                 					 help='The evaluation mode TEE for TEEM_faster_RCNN and base for original faster_RCNN ')
parser.add_argument('--num_workers',         	   type=int,                default=4,                                       help='num_workers in dataloader')
parser.add_argument('--batch_size',         	   type=int,                default=64,                                      help='batch size for training and evaluation')
parser.add_argument('--epoch',         	           type=int,                default=30,                                       help='number of training epochs')
parser.add_argument('--LR',                        type=float,              default=1e-3,                                    help='Learningrate')
parser.add_argument('--gamma',                     type=float,              default=1e-1,                                    help='sigmma value for lr Scheduler')
parser.add_argument('--ST',                        type=int,                default=15,                                      help='step size for lr Scheduler')
parser.add_argument('--CSV_Path',                  type=str,                default='/dataset/', 				             help='directory to csv files contains video frame pairs and annontation value')
parser.add_argument('--TEEMs_path',                type=str,                default='/TrainedTEEMs/',                        help='num_workers in dataloader')
parser.add_argument('--video_frames_path',         type=str,                default='/dataset/frames',                       help='List of IoU threshols to measure mAP')
parser.add_argument('--MultipleGPU',               type=bool,                default=True,                                   help='True if training on multiple gpus')
args = parser.parse_args()

import os
from TEE_BackBone import TEE_Backbone_Network
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import csv
from PIL import Image
from csv import reader
from TEEM_module import TEEM
from TEE_utils import TEEM_dataset,train_TEEMs_one_epoch,TEEM_val_one_epoch
from tqdm import tqdm

CSV_File = os.getcwd() + args.CSV_Path
video_frames_path = os.getcwd() + args.video_frames_path
trained_TEEMs_path = os.getcwd() + args.TEEMs_path


Vatic_train_dataset = TEEM_dataset(video_frames_path, CSV_File, mode = "train")
Vatic_eval_dataset = TEEM_dataset(video_frames_path, CSV_File, mode = "eval")

TEEMs_train_data_loader = DataLoader(
					    dataset = Vatic_train_dataset,
					    num_workers = args.num_workers,
					    batch_size = args.batch_size,
					    shuffle = True
						)


TEEMs_eval_data_loader = DataLoader(
					    dataset = Vatic_eval_dataset,
					    num_workers = args.num_workers,
					    batch_size = args.batch_size,
					    shuffle = True
						)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
backbone = TEE_Backbone_Network(pretrained=True,min_size=224)
for name, param in backbone.named_parameters():
			param.requires_grad = False 
backbone.eval()
backbone.to(device)

TEEM1 = TEEM(256,128)
TEEM2 = TEEM(512,256)
TEEM3 = TEEM(1024,128)
TEEM4 = TEEM(2048,256)

TEEM1.to(device)
TEEM2.to(device)
TEEM3.to(device)
TEEM4.to(device)

if args.MultipleGPU:
	backbone =  torch.nn.DataParallel(backbone)
	TEEM1 =  torch.nn.DataParallel(TEEM1)
	TEEM2 =  torch.nn.DataParallel(TEEM2)
	TEEM3 =  torch.nn.DataParallel(TEEM3)
	TEEM4 =  torch.nn.DataParallel(TEEM4)

# Optimizer
optimizer1 = torch.optim.Adam(TEEM1.parameters(), lr=args.LR)
optimizer2 = torch.optim.Adam(TEEM2.parameters(), lr=args.LR)
optimizer3 = torch.optim.Adam(TEEM3.parameters(), lr=args.LR)
optimizer4 = torch.optim.Adam(TEEM4.parameters(), lr=args.LR)

# Learning Rate Scheduler
lr_scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size = args.ST, gamma = args.gamma)
lr_scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size = args.ST, gamma = args.gamma)
lr_scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size = args.ST, gamma = args.gamma)
lr_scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size = args.ST, gamma = args.gamma)

#Loss Function
criterion = nn.CrossEntropyLoss()

training_accuracy =list()
validatioin_accuracy = list()

for epoch in tqdm(range(args.epoch)):
    
    ###Training
    TEEM1.train()
    TEEM2.train()
    TEEM3.train()
    TEEM4.train()

    acc = train_TEEMs_one_epoch(TEEMs_train_data_loader,backbone,TEEM1,TEEM2,TEEM3,TEEM4,criterion,optimizer1,optimizer2,optimizer3,optimizer4,device)
    training_accuracy.append(acc)
    
    #Print Epoch Details
    print("\nTraining")
    print("Epoch {}".format(epoch+1))
    print("Acc1 : {}".format(round(acc[0], 4)))
    print("Acc2 : {}".format(round(acc[1], 4)))
    print("Acc3 : {}".format(round(acc[2], 4)))
    print("Acc4 : {}".format(round(acc[3], 4)))

    lr_scheduler1.step()
    lr_scheduler2.step()
    lr_scheduler3.step()
    lr_scheduler4.step()
    
    #Evaluation
    TEEM1.eval()
    TEEM2.eval()
    TEEM3.eval()
    TEEM4.eval()
    
    acc= TEEM_val_one_epoch(TEEMs_eval_data_loader,backbone,TEEM1,TEEM2,TEEM3,TEEM4,criterion,device)
    validatioin_accuracy.append(acc)
    
    #Print Epoch Details
    print(acc)
    print("\nValidation")
    print("Acc1 : {}".format(round(acc[0], 4)))
    print("Acc2 : {}".format(round(acc[1], 4)))
    print("Acc3 : {}".format(round(acc[2], 4)))
    print("Acc4 : {}".format(round(acc[3], 4)))

#torch.save(TEEM0.module.state_dict(),os.path.join(model_path+IoU,"TEEM0.pth"))    
torch.save(TEEM1.module.state_dict(),os.path.join(trained_TEEMs_path,"TEEM1.pth"))
torch.save(TEEM2.module.state_dict(),os.path.join(trained_TEEMs_path,"TEEM2.pth"))
torch.save(TEEM3.module.state_dict(),os.path.join(trained_TEEMs_path,"TEEM3.pth"))
torch.save(TEEM4.module.state_dict(),os.path.join(trained_TEEMs_path,"TEEM4.pth"))

with open(trained_TEEMs_path+"train_acc.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(training_accuracy)

with open(trained_TEEMs_path+"validation_acc.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(validatioin_accuracy)

print("--- TEEMs Saved ---")
