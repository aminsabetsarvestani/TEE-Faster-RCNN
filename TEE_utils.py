
import os
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

class TEEM_dataset(Dataset):

	def __init__(self, images_root, csv_file_root,mode = "train"):
		super().__init__()
		self.mode = mode
		self.root_dir = images_root
		read_obj = open(csv_file_root + self.mode  +'.csv', 'r')
		csv_reader = reader(read_obj)
		self.list_of_tuples = list(map(tuple, csv_reader))
        
	def __getitem__(self, idx):

		frm1 = os.path.join(self.root_dir,str(self.list_of_tuples[idx][0])+".jpg")
		frm2 = os.path.join(self.root_dir,str(self.list_of_tuples[idx][1])+".jpg")

		img_left = Image.open(frm1)
		img_right= Image.open(frm2)
		norm = T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
		resize = T.Resize(size=(224, 224))
		#resizeand romalize frames
		img_left = resize (img_left)
		img_right = resize(img_right)      
		if self.mode == "train" or self.mode == "eval":
		    ### Preparing class label
			label = self.list_of_tuples[idx][2]
			label = torch.tensor(int(label), dtype = torch.long)
			if label>1:
				print('Error: label Should be 0 or 1')
			
			image1 = TF.to_tensor(img_left)
			image2 = TF.to_tensor(img_right)

			image1 = norm(image1)
			image2 = norm(image2)

			return image1,image2,label
        
		elif self.mode == "test":
		    
		    ### Apply Transforms on image
			image1 = TF.to_tensor(img_left)
			image2 = TF.to_tensor(img_right)
			image1 = norm(image1)
			image2 = norm(image2)
			return image1,image2
            
        
	def __len__(self):
		return len(self.list_of_tuples)


def train_TEEMs_one_epoch(train_data_loader,backbone,TEEM1,TEEM2,TEEM3,TEEM4,criterion,optimizer1,optimizer2,optimizer3,optimizer4,device):

	epoch_loss1 = []
	epoch_acc1 = []
	epoch_loss2 = []
	epoch_acc2 = []
	epoch_loss3 = []
	epoch_acc3 = []
	epoch_loss4 = []
	epoch_acc4 = []
	correct1 = 0
	correct2= 0
	correct3 = 0
	correct4 = 0

	total = 0
	for frm1, frm2, labels in train_data_loader:

		labels = labels.to(device)

		frm1 = frm1.to(device)
		frm2 = frm2.to(device)
		
		F1 = backbone(frm1)
		F2 = backbone(frm2)
		
		#Reseting Gradients
		optimizer1.zero_grad()
		preds = TEEM1(F1['0'],F2['0'])
		#Calculating Loss
		_loss = criterion(preds,labels)
		loss = _loss.item()
		epoch_loss1.append(loss)
		#Calculating Accuracy
		#Backward
		_loss.backward(retain_graph = True)
		optimizer1.step()
		_, preds = torch.max(preds.data, 1)
		correct1 += (preds == labels).sum().item()
		
		optimizer2.zero_grad()
		preds = TEEM2(F1['1'],F2['1'])
		#Calculating Loss
		_loss = criterion(preds,labels)
		loss = _loss.item()
		epoch_loss2.append(loss)
		#Calculating Accuracy
		#Backward
		_loss.backward(retain_graph = True)
		optimizer2.step()
		_, preds = torch.max(preds.data, 1)
		correct2 += (preds == labels).sum().item()
		
		optimizer3.zero_grad()
		preds = TEEM3(F1['2'],F2['2'])
		#Calculating Loss
		_loss = criterion(preds,labels)
		loss = _loss.item()
		epoch_loss3.append(loss)
		#Backward
		_loss.backward(retain_graph = True)
		optimizer3.step()
		_, preds = torch.max(preds.data, 1)
		correct3 += (preds == labels).sum().item()
		
		optimizer4.zero_grad()
		preds = TEEM4(F1['3'],F2['3'])		
		#Calculating Loss
		_loss = criterion(preds,labels)
		loss = _loss.item()
		epoch_loss4.append(loss)
		#Calculating Accuracy
		
		#Backward
		_loss.backward()
		optimizer4.step()
		_, preds = torch.max(preds.data, 1)
		correct4 += (preds == labels).sum().item()		

		total += labels.size(0)

        ###Overall Epoch Results
	epoch_acc1 = (100 * correct1 / total)
	epoch_acc2 = (100 * correct2 / total)
	epoch_acc3 = (100 * correct3 / total)
	epoch_acc4 = (100 * correct4 / total)  	
	
	return  [epoch_acc1,epoch_acc2,epoch_acc3,epoch_acc4]




def TEEM_val_one_epoch(eval_data_loader,backbone,TEEM1,TEEM2,TEEM3,TEEM4,criterion,device):
    
	epoch_loss1 = []
	epoch_acc1 = []
	epoch_loss2 = []
	epoch_acc2 = []
	epoch_loss3 = []
	epoch_acc3 = []
	epoch_loss4 = []
	epoch_acc4 = []
	correct1 = 0
	correct2= 0
	correct3 = 0
	correct4 = 0
	total = 0
	with torch.no_grad():
		for frm1, frm2, labels in eval_data_loader:

			labels = labels.to(device)

			frm1 = frm1.to(device)
			frm2 = frm2.to(device)
			
			F1 = backbone(frm1)
			F2 = backbone(frm2)
			
		
			preds = TEEM1(F1['0'],F2['0'])
			_loss = criterion(preds,labels)
			loss = _loss.item()
			epoch_loss1.append(loss)
			_, preds = torch.max(preds.data, 1)
			correct1 += (preds == labels).sum().item()
			
			preds = TEEM2(F1['1'],F2['1'])
			_loss = criterion(preds,labels)
			loss = _loss.item()
			epoch_loss2.append(loss)
			_, preds = torch.max(preds.data, 1)
			correct2 += (preds == labels).sum().item()

			preds = TEEM3(F1['2'],F2['2'])
			_loss = criterion(preds,labels)
			loss = _loss.item()
			epoch_loss3.append(loss)
			_, preds = torch.max(preds.data, 1)
			correct3 += (preds == labels).sum().item()
		
			preds = TEEM4(F1['3'],F2['3'])		
			_loss = criterion(preds,labels)
			loss = _loss.item()
			epoch_loss4.append(loss)
			_, preds = torch.max(preds.data, 1)
			correct4 += (preds == labels).sum().item()
			

			total += labels.size(0)

	epoch_acc1 = (100 * correct1 / total)
	epoch_acc2 = (100 * correct2 / total)
	epoch_acc3 = (100 * correct3 / total)
	epoch_acc4 = (100 * correct4 / total)  	

	return [epoch_acc1,epoch_acc2,epoch_acc3,epoch_acc4]

