import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np
import cv2
import torch
from tqdm import tqdm
from torchsummary import summary
import argparse
import torchvision
from tqdm import tqdm


from saicinpainting.evaluation.data import InpaintingDataset
from saicinpainting.training.data.datasets import make_default_train_dataloader, make_default_val_dataloader
from suppliment import initialize_gpu,Waveblock,WaveMix,HybridLoss,calc_curr_performance
from discriminator import CnnNetwork, Resnet18, Resnet34, Resnet50, Resnet34_W_Mask
from discriminator import Model
 
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import autograd
 
parser = argparse.ArgumentParser()


parser.add_argument("-vis", "--Visual_example", default="Train_samples", help = "Path to save visual examples")
parser.add_argument("-batch", "--Batch_size", default=40, help = "Batch Size")
parser.add_argument("-gpu", "--GPU_number", default=-1, help = "GPU_number")


args = parser.parse_args()

Visual_example_loc=str(args.Visual_example)
if int(args.GPU_number)==-1:
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
	device = torch.device("cuda:"+str(args.GPU_number) if torch.cuda.is_available() else "cpu")
initialize_gpu(args.GPU_number)
print("wavemixGAN:   "+str(device))

TrainBatchSize=int(args.Batch_size)


# eval_loader=InpaintingDataset(datadir="my_dataset/eval/random_thick_96/",img_suffix=".png")

ValDataLoaderConfig={'dataloader_kwargs': {'batch_size': 8, 'shuffle': False, 'num_workers': 2}}

# eval_loader=InpaintingDataset(datadir="ImageNet/eval/random_medium_224/",img_suffix=".png")

eval_loader=make_default_val_dataloader(indir="ImageNet/eval/random_medium_224/",img_suffix=".png", **ValDataLoaderConfig)




# for data in train_loader:
# 	try:
# 		img=data["image"]
# 		print(img.shape)
# 		mask=data["mask"]
# 		break
# 	except:
# 		continue
# img[:, :, :] = img[:, :, :] * mask
# plt.imshow(img[0].permute(1,2,0))
# plt.show()


### MODEL ARCHITECTURE ###

BATCH_SIZE=64
IMAGE_SIZE=(224,224)
MASK_SIZE=12
MODEL_DEPTH=7 #16
MODEL_EMBEDDING=128 #256
NUM_MODELS=1
FF_CHANNEL=128
REDUCTION_CONV=1
NUM_DWT=1 
# ALPHA=0.75
MAX_EVAL=10
base_path="WavemixModels"
extra="_thin_mask"
LOAD=False

###########################################################################################################################################



# PATH = base_path + str('chkpoint__IMG__'+str(MASK_SIZE)+'__MODEL__D'+str(MODEL_DEPTH)+'_E'+str(MODEL_EMBEDDING)+'_N'+str(NUM_MODELS)+'_C'+str(REDUCTION_CONV)+'_F'+str(FF_CHANNEL)+'_#dwt='+str(NUM_DWT)+extra+'.pth')
# print(PATH)
# model.load_state_dict(torch.load(PATH))
# print("LOADED WEIGHTS!!!")

###########################################################################################################################################


def gradient_penalty(D, xr, xf):
    """

    :param D:
    :param xr: [b, 2]
    :param xf: [b, 2]
    :return:
    """
    # [b, 1]
    t = torch.rand(TrainBatchSize, 1, 1, 1).to(device)
    # interpolation
    # print(t.shape, xr.shape)
    mid = t * xr + (1 - t) * xf
    # set it to require grad info
    mid.requires_grad_()

    pred = D(mid)
    grads = autograd.grad(outputs=pred, inputs=mid,
                          grad_outputs=torch.ones_like(pred),
                          create_graph=True, retain_graph=True, only_inputs=True)[0]

    gp = torch.pow(grads.norm(2, dim=1) - 1, 2).mean()

    return gp



import torch.optim as optim
from torch.autograd import Variable
import sys, os, time, pickle

netG=Model(num_classes = 3,
			depth = MODEL_DEPTH,
			mult = 2,
			ff_channel = FF_CHANNEL,
			final_dim = MODEL_EMBEDDING, ## 64
			dropout = 0.5,
			num_models=NUM_MODELS
			).to(device)

# summary(netG, [(3,224,224), (1,224,224)])

# exit()

# PATH = base_path + str('chkpoint__IMG__'+str(MASK_SIZE)+'__MODEL__D'+str(MODEL_DEPTH)+'_E'+str(MODEL_EMBEDDING)+'_N'+str(NUM_MODELS)+'_C'+str(REDUCTION_CONV)+'_F'+str(FF_CHANNEL)+'_#dwt='+str(NUM_DWT)+extra+'.pth')
# print(PATH)
# netG.load_state_dict(torch.load(PATH))
# print("LOADED GEN WEIGHTS!!!")

netD=Resnet18(False).to(device)


Losses=calc_curr_performance(netG,eval_loader,entire_dataset=False)
Final_losses={}
for metric in Losses.keys():
	Final_losses[metric]=np.array(Losses[metric]).mean()
print("="*50)
print("####PRE TRAIN:",Final_losses)
print("="*50)


lr=0.0002
beta1 = 0.5
num_epochs = 1000
criterion = nn.BCELoss()


# fixed_noise = torch.randn(64, nz, 1, 1, device=device)

real_label = 1.
fake_label = 0.

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

img_list = []
G_losses = []
D_losses = []
iters = 0


TrainDataLoaderConfig={'indir': 'ImageNet/train', 'out_size': 224, 'mask_gen_kwargs': {'irregular_proba': 1, 'irregular_kwargs': {'max_angle': 4, 'max_len': 35, 'max_width': 30, 'max_times': 10, 'min_times': 4}, 'box_proba': 1, 'box_kwargs': {'margin': 0, 'bbox_min_size': 30, 'bbox_max_size': 75, 'max_times': 5, 'min_times': 2}, 'segm_proba': 0}, 'transform_variant': 'distortions', 
						'dataloader_kwargs': {'batch_size': TrainBatchSize, 'shuffle': False, 'num_workers': 2}}  ### IMAGENET

	# train_loader=make_default_val_dataloader(indir="ImageNet/eval/random_medium_224/",img_suffix=".png", out_size=224, **ValDataLoaderConfig) # val loader
train_loader=make_default_train_dataloader(**TrainDataLoaderConfig)


prev_lpips=float("inf")

for epoch in range(num_epochs):
	# For each batch in the dataloader
	with tqdm(train_loader, unit="batch") as tepoch:
		tepoch.set_description(f"Epoch {epoch+1}")
		for i, data in enumerate(tepoch, 0):

			image, mask = data["image"].to(device), data["mask"].to(device)
			labels = image.clone().detach() ## expected output
			image[:, :, :] = image[:, :, :] * (1-mask)
			inputs=image

			############################
			# (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
			###########################
			## Train with all-real batch
			netD.zero_grad()
			# Format batch
			real_cpu = labels.to(device)
			b_size = real_cpu.size(0)
			label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
			# Forward pass real batch through D
			output = netD(real_cpu).view(-1)
			# Calculate loss on all-real batch
			errD_real = criterion(output, label)
			# Calculate gradients for D in backward pass
			errD_real.backward()
			D_x = output.mean().item()

			## Train with all-fake batch
			# Generate batch of latent vectors
			# noise = torch.randn(b_size, nz, 1, 1, device=device)
			# Generate fake image batch with G
			fake = netG(inputs, mask)
			label.fill_(fake_label)
			# Classify all fake batch with D
			output = netD(fake.detach()).view(-1)
			# Calculate D's loss on the all-fake batch
			errD_fake = criterion(output, label)
			# Calculate the gradients for this batch, accumulated (summed) with previous gradients
			errD_fake.backward()
			D_G_z1 = output.mean().item()
			# Compute error of D as sum over the fake and the real batches
			errD = errD_real + errD_fake
			# Calculate the Gradient Penalty:
			gp = 0.2 * gradient_penalty(netD, real_cpu, fake.detach())
			gp.backward()

			# Update D
			optimizerD.step()

			############################
			# (2) Update G network: maximize log(D(G(z)))
			###########################
			netG.zero_grad()
			label.fill_(real_label)  # fake labels are real for generator cost
			# Since we just updated D, perform another forward pass of all-fake batch through D
			output = netD(fake).view(-1)
			# Calculate G's loss based on this output
			errG = criterion(output, label)
			# Calculate gradients for G
			errG.backward()
			D_G_z2 = output.mean().item()
			# Update G
			optimizerG.step()

			tepoch.set_postfix_str(f" D_loss : {errD.item():.4f} G_loss : {errG.item():.4f}")




			# Output training stats
			if i % 500 == 0:
				print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
					  % (epoch, num_epochs, i, len(train_loader),
						 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

			# Save Losses for plotting later
			G_losses.append(errG.item())
			D_losses.append(errD.item())

			if i % int(len(train_loader)/500)==int(len(train_loader)/500)-1:
				break

			# Check how the generator is doing by saving G's output on fixed_noise
			# if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(train_loader)-1)):
			# 	with torch.no_grad():
			# 		fake = netG(fixed_noise).detach().cpu()
			# 	img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

			iters += 1
		

		outputs = netG(inputs, mask).detach()
		cv2.imwrite("Visual_example/"+Visual_example_loc+"/"+str(epoch)+"_temp_input"+".png",cv2.cvtColor(inputs[0].permute([1,2,0]).cpu().detach().numpy()*255, cv2.COLOR_RGB2BGR))
		cv2.imwrite("Visual_example/"+Visual_example_loc+"/"+str(epoch)+"_temp_output"+".png",cv2.cvtColor(outputs[0].permute([1,2,0]).cpu().detach().numpy()*255, cv2.COLOR_RGB2BGR))
		cv2.imwrite("Visual_example/"+Visual_example_loc+"/"+str(epoch)+"_1_temp_input"+".png",cv2.cvtColor(inputs[1].permute([1,2,0]).cpu().detach().numpy()*255, cv2.COLOR_RGB2BGR))
		cv2.imwrite("Visual_example/"+Visual_example_loc+"/"+str(epoch)+"_1_temp_output"+".png",cv2.cvtColor(outputs[1].permute([1,2,0]).cpu().detach().numpy()*255, cv2.COLOR_RGB2BGR))
		# labels_plus = torch.cat([labels, mask.reshape(-1,1,224,224)], dim=1)
		# otputs_plus = torch.cat([outputs, mask.reshape(-1,1,224,224)], dim=1)
		P_bar = netD(labels).reshape(-1)
		N_bar = netD(outputs).reshape(-1)


		Losses=calc_curr_performance(netG,eval_loader,entire_dataset=False)
		Final_losses={}
		for metric in Losses.keys():
			Final_losses[metric]=np.array(Losses[metric]).mean()

		if prev_lpips>=Final_losses["LPIPS"]:
			PATH = base_path + str('chkpoint__IMG__'+str(MASK_SIZE)+'__MODEL__D'+str(MODEL_DEPTH)+'_E'+str(MODEL_EMBEDDING)+'_N'+str(NUM_MODELS)+'_C'+str(REDUCTION_CONV)+'_F'+str(FF_CHANNEL)+'_#dwt='+str(NUM_DWT)+extra+'_newSSIM.pth')
			D_PATH = base_path +'discriminator_'+extra+'_newSSIM.pth'
			torch.save(netG.state_dict(), PATH)
			torch.save(netD.state_dict(), D_PATH)
			prev_lpips=Final_losses["LPIPS"]
			print("saving chkpoint")
		
		print("="*50)
		print("####EPOCH:"+str(epoch),Final_losses)
		print("####EPOCH:"+str(epoch),P_bar.mean().item(), N_bar.mean().item()) 
		print("="*50)

if not LOAD:

	# TrainDataLoaderConfig={'indir': 'my_dataset/train', 'out_size': 96, 'mask_gen_kwargs': {'irregular_proba': 1, 'irregular_kwargs': {'max_angle': 4, 'max_len': 50, 'max_width': 60, 'max_times': 5, 'min_times': 1}, 'box_proba': 1, 'box_kwargs': {'margin': 10, 'bbox_min_size': 15, 'bbox_max_size': 40, 'max_times': 4, 'min_times': 1}, 'segm_proba': 0}, 'transform_variant': 'distortions', 
	# 					'dataloader_kwargs': {'batch_size': TrainBatchSize, 'shuffle': False, 'num_workers': 2}}  ### Default

	# TrainDataLoaderConfig={'indir': 'my_dataset/train', 'out_size': 96, 'mask_gen_kwargs': {'irregular_proba': 1, 'irregular_kwargs': {'max_angle': 4, 'max_len': 50, 'max_width': 30, 'max_times': 5, 'min_times': 1}, 'box_proba': 1, 'box_kwargs': {'margin': 10, 'bbox_min_size': 15, 'bbox_max_size': 30, 'max_times': 4, 'min_times': 0}, 'segm_proba': 0}, 'transform_variant': 'distortions', 
	# 					'dataloader_kwargs': {'batch_size': TrainBatchSize, 'shuffle': True, 'num_workers': 2}} ###STL10

	
	GAN.discriminator.train()
	GAN.model.train()
	prev_loss = np.ones(3)*float("inf")


	for epoch in range(100):  # loop over the dataset multiple times
		epoch_loss=np.zeros(3)
		with tqdm(train_loader, unit="batch") as tepoch:
			tepoch.set_description(f"Epoch {epoch+1}")
			for i, data in enumerate(tepoch, 0):
				D_loss = GAN.D_train(data,1)
				# G_loss, ADV_loss = GAN.G_train(data, 1,count=i)

				epoch_loss[0] += D_loss/ len(train_loader)
				epoch_loss[1] += 0
				epoch_loss[2] += 0
				tepoch.set_postfix_str(f" D_loss : {epoch_loss[0]:.4f} G_loss : {epoch_loss[1]:.4f} ADV_loss : {epoch_loss[2]:.4f} ")

	######################################################################################################################################################

				if i % int(len(train_loader)/100)==int(len(train_loader)/100)-1:	# print every 2000 mini-batches
					image, mask = data["image"].to(device), data["mask"].to(device)
					labels = image.clone().detach() ## expected output
					image[:, :, :] = image[:, :, :] * (1-mask)
					inputs=image
					outputs=GAN.model(inputs, mask)

					print('[%d, %5d] D loss: %.3f Gen loss: %.3f ADV loss: %.3f' %
						  (epoch + 1, i + 1, epoch_loss[0], epoch_loss[1], epoch_loss[2]))
					
					GAN.validate_model(epoch)
					GAN.validate_discriminator(epoch)
				
					if prev_loss[1] >= epoch_loss[1]:
						cv2.imwrite("Visual_example/"+Visual_example_loc+"/"+str(epoch)+"_temp_input"+".png",cv2.cvtColor(inputs[0].permute([1,2,0]).cpu().detach().numpy()*255, cv2.COLOR_RGB2BGR))
						cv2.imwrite("Visual_example/"+Visual_example_loc+"/"+str(epoch)+"_temp_output"+".png",cv2.cvtColor(outputs[0].permute([1,2,0]).cpu().detach().numpy()*255, cv2.COLOR_RGB2BGR))
						PATH = base_path + str('chkpoint__IMG__'+str(MASK_SIZE)+'__MODEL__D'+str(MODEL_DEPTH)+'_E'+str(MODEL_EMBEDDING)+'_N'+str(NUM_MODELS)+'_C'+str(REDUCTION_CONV)+'_F'+str(FF_CHANNEL)+'_#dwt='+str(NUM_DWT)+extra+'_newSSIM.pth')
						D_PATH = base_path +'discriminator_'+extra+'_newSSIM.pth'

						torch.save(GAN.model.state_dict(), PATH)
						torch.save(GAN.discriminator.state_dict(), D_PATH)
						prev_loss=epoch_loss
						print("saving chkpoint")

					break
				# test_loss = 0
				# total = 0
				# if epoch%1==0:
				# 	index=np.random.randint(len(inputs)-3)
				# 	for i in range(3):
				# 		cv2.imwrite("Visual_example/"+Visual_example_loc+"/"+str(epoch)+"__"+str(i)+"_Input.png",inputs[index+i].permute([1,2,0]).cpu().detach().numpy()*255)
				# 		cv2.imwrite("Visual_example/"+Visual_example_loc+"/"+str(epoch)+"__"+str(i)+"_Output.png",outputs[index+i].permute([1,2,0]).cpu().detach().numpy()*255)
				# 	Losses=calc_curr_performance(self.model,eval_loader,entire_dataset=False)
				# 	Final_losses={}
				# 	for metric in Losses.keys():
				# 		Final_losses[metric]=np.array(Losses[metric]).mean()
				# 	print("####epoch ",epoch+1,"training: ",Final_losses)

	######################################################################################################################################################

else:
	PATH = base_path + str('chkpoint__IMG__'+str(MASK_SIZE)+'__MODEL__D'+str(MODEL_DEPTH)+'_E'+str(MODEL_EMBEDDING)+'_N'+str(NUM_MODELS)+'_C'+str(REDUCTION_CONV)+'_F'+str(FF_CHANNEL)+'_#dwt='+str(NUM_DWT)+extra+'.pth')
	model.load_state_dict(torch.load(PATH))
	print("LOADED WEIGHTS!!!")
	print("PERFORMANCE: \n",calc_curr_performance(model,eval_loader,entire_dataset=True))
	for i, data in enumerate(tqdm(eval_loader), 0):
		if i >= MAX_EVAL:
			break

		image, mask = torch.Tensor(data["image"]).to(device), torch.Tensor(data["mask"]).to(device)
		labels = image.clone().detach() ## expected output
		image[:, :, :] = image[:, :, :] * (1-mask)
		inputs=image.reshape(-1,3,96,96)
		mask=mask.reshape(-1,1,96,96)
		outputs = model(inputs, mask)
		cv2.imwrite("Visual_example/"+str(i)+"_Output.png",outputs[0].permute([1,2,0]).cpu().detach().numpy()*255)
		cv2.imwrite("Visual_example/"+str(i)+"_Input.png",inputs[0].permute([1,2,0]).cpu().detach().numpy()*255)
