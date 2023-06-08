import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np
import cv2
import torch
from tqdm import tqdm
from torchsummary import summary
import argparse
import torch.optim as optim
from torch.autograd import Variable
import sys, os, time, pickle
from tqdm import tqdm


from saicinpainting.evaluation.data import InpaintingDataset
from saicinpainting.training.data.datasets import make_default_train_dataloader
from suppliment import initialize_gpu,Waveblock,WaveMix,HybridLoss,calc_curr_performance, Model
 

 
parser = argparse.ArgumentParser()


parser.add_argument("-vis", "--Visual_example", default="Train_samples", help = "Path to save visual examples")
parser.add_argument("-batch", "--Batch_size", default=40, help = "Batch Size")
parser.add_argument("-gpu", "--GPU_number", default=-1, help = "GPU_number")
parser.add_argument("-tdata", "--Data_Path", default="/home/Drive3/Dharshan/Venv/lama/ImageNet/train", help = "Path to train dataset")



args = parser.parse_args()

Visual_example_loc=str(args.Visual_example)
if int(args.GPU_number)==-1:
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
	device = torch.device("cuda:"+str(args.GPU_number) if torch.cuda.is_available() else "cpu")
initialize_gpu(args.GPU_number)
print(device)

TrainBatchSize=int(args.Batch_size)
print(args.Data_Path)
TrainDataPath=str(args.Data_Path)


eval_loader=InpaintingDataset(datadir="/home/Drive3/Dharshan/Venv/lama/ImageNet/eval/random_medium_224/",img_suffix=".png")

# TrainDataLoaderConfig={'indir': TrainDataPath, 'out_size': 96, 'mask_gen_kwargs': {'irregular_proba': 1, 'irregular_kwargs': {'max_angle': 4, 'max_len': 50, 'max_width': 30, 'max_times': 5, 'min_times': 1}, 'box_proba': 1, 'box_kwargs': {'margin': 10, 'bbox_min_size': 15, 'bbox_max_size': 30, 'max_times': 4, 'min_times': 0}, 'segm_proba': 0}, 'transform_variant': 'distortions', 
# 						'dataloader_kwargs': {'batch_size': TrainBatchSize, 'shuffle': True, 'num_workers': 2}}

TrainDataLoaderConfig={'indir': TrainDataPath, 'out_size': 224, 'mask_gen_kwargs': {'irregular_proba': 1, 'irregular_kwargs': {'max_angle': 4, 'max_len': 35, 'max_width': 30, 'max_times': 10, 'min_times': 4}, 'box_proba': 1, 'box_kwargs': {'margin': 0, 'bbox_min_size': 30, 'bbox_max_size': 75, 'max_times': 5, 'min_times': 2}, 'segm_proba': 0}, 'transform_variant': 'distortions', 
						'dataloader_kwargs': {'batch_size': TrainBatchSize, 'shuffle': True, 'num_workers': 2}}  ### IMAGENET

### MODEL ARCHITECTURE ###

# Module parameters
BATCH_SIZE=64
IMAGE_SIZE=(224,224)
MASK_SIZE=12
MODEL_DEPTH=7
MODEL_EMBEDDING=128
FF_CHANNEL=128
REDUCTION_CONV=1
NUM_DWT=1


NUM_MODELS=4 # number of models not modules


model = Model(
	num_classes = 3,
	depth = MODEL_DEPTH,
	mult = 2,
	ff_channel = FF_CHANNEL,
	final_dim = MODEL_EMBEDDING,
	dropout = 0.5,
	num_models=NUM_MODELS
).to(device)


# summary(model, [(3,224,224), (1,224,224)])



### END MODEL ARCHITECTURE ###

base_path="WavemixModels"
extra="_thin_mask"


TRAIN=True  # Train (T) or Evaluate (F)
MAX_EVAL=10
VAL_CYCLE=1 # every nth epoch it will validate
EPOCHS=100

train_loader=make_default_train_dataloader(**TrainDataLoaderConfig)

scaler = torch.cuda.amp.GradScaler()
optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
criterion = HybridLoss().to(device)
prev_loss=float("inf")

Losses=calc_curr_performance(model,eval_loader)
Final_losses={}
for metric in Losses.keys():
	Final_losses[metric]=np.array(Losses[metric]).mean()
print("#### Performance before training:",Final_losses)

start_time=time.time()
for epoch in range(EPOCHS):
	start_index=np.random.randint(BATCH_SIZE)
	t0 = time.time()
	running_loss = 0.0

	model.train()
	for i, data in enumerate(tqdm(train_loader), 0):
		# get the inputs; data is a list of [inputs, labels]
		image, mask = data["image"].to(device), data["mask"].to(device)
		GT = image.clone().detach() ## GT
		GT=GT.to(device)
		image[:, :, :] = image[:, :, :] * (1-mask)
		inputs=image

		optimizer.zero_grad()
		outputs = model(inputs, mask)
		cv2.imwrite("Visual_example/"+Visual_example_loc+"/test_input.png",cv2.cvtColor(inputs[0].permute([1,2,0]).cpu().detach().numpy()*255, cv2.COLOR_RGB2BGR))
		cv2.imwrite("Visual_example/"+Visual_example_loc+"/test_output.png",cv2.cvtColor(outputs[0].permute([1,2,0]).cpu().detach().numpy()*255, cv2.COLOR_RGB2BGR))
		

		with torch.cuda.amp.autocast():  # Automatic mixed precision
			loss = criterion(outputs, 1-mask.to(device), GT)
		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()
	
		# print statistics
		running_loss += loss.detach().item()
		if i % int(len(train_loader)/50)==int(len(train_loader)/50)-1:    
			print('[%d, %5d] loss: %.3f' %
				  (epoch + 1, i + 1, running_loss))
			
			if prev_loss >= running_loss:
				cv2.imwrite("Visual_example/"+Visual_example_loc+"/"+"temp_input"+str(epoch)+".png",cv2.cvtColor(inputs[0].permute([1,2,0]).cpu().detach().numpy()*255, cv2.COLOR_RGB2BGR))
				cv2.imwrite("Visual_example/"+Visual_example_loc+"/"+"temp_output"+str(epoch)+".png",cv2.cvtColor(outputs[0].permute([1,2,0]).cpu().detach().numpy()*255, cv2.COLOR_RGB2BGR))
				PATH = base_path + str('chkpoint__IMG__'+str(MASK_SIZE)+'__MODEL__D'+str(MODEL_DEPTH)+'_E'+str(MODEL_EMBEDDING)+'_N'+str(NUM_MODELS)+'_C'+str(REDUCTION_CONV)+'_F'+str(FF_CHANNEL)+'_#dwt='+str(NUM_DWT)+extra+'.pth')
				torch.save(model.state_dict(), PATH)
				prev_loss=running_loss
				print("saving chkpoint")
				
			running_loss = 0.0
			# break

	if epoch%VAL_CYCLE==0:
		num_images=min(TrainBatchSize, 3)
		print(num_images)
		try:
			start_index=np.random.randint(len(inputs)-num_images)
		except:
			start_index=0
		for i in range(num_images):
			cv2.imwrite("Visual_example/"+Visual_example_loc+"/"+str(epoch)+"__"+str(i)+"_Input.png",inputs[start_index+i].permute([1,2,0]).cpu().detach().numpy()*255)
			cv2.imwrite("Visual_example/"+Visual_example_loc+"/"+str(epoch)+"__"+str(i)+"_Output.png",outputs[start_index+i].permute([1,2,0]).cpu().detach().numpy()*255)
		
		Losses=calc_curr_performance(model,eval_loader)
		Final_losses={}
		for metric in Losses.keys():
			Final_losses[metric]=np.array(Losses[metric]).mean()
		print("#### Performance in epoch ",epoch+1,"training: ",Final_losses)

print('Finished Training')
print(f"Time for AdamW {time.time()-start_time:.4f}")
	

# else:
# 	PATH = base_path + str('chkpoint__IMG__'+str(MASK_SIZE)+'__MODEL__D'+str(MODEL_DEPTH)+'_E'+str(MODEL_EMBEDDING)+'_N'+str(NUM_MODELS)+'_C'+str(REDUCTION_CONV)+'_F'+str(FF_CHANNEL)+'_#dwt='+str(NUM_DWT)+extra+'.pth')
# 	model.load_state_dict(torch.load(PATH))
# 	print("LOADED WEIGHTS!!!")
# 	print("PERFORMANCE: \n",calc_curr_performance(model,eval_loader))
# 	for i, data in enumerate(tqdm(eval_loader), 0):
# 		if i >= MAX_EVAL:
# 			break

# 		image, mask = torch.Tensor(data["image"]).to(device), torch.Tensor(data["mask"]).to(device)
# 		labels = image.clone().detach() ## expected output
# 		image[:, :, :] = image[:, :, :] * (1-mask)
# 		inputs=image.reshape(-1,3,96,96)
# 		mask=mask.reshape(-1,1,96,96)
# 		outputs = model(inputs, mask)
# 		cv2.imwrite("Visual_example/"+str(i)+"_Output.png",outputs[0].permute([1,2,0]).cpu().detach().numpy()*255)
# 		cv2.imwrite("Visual_example/"+str(i)+"_Input.png",inputs[0].permute([1,2,0]).cpu().detach().numpy()*255)
