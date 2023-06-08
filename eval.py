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


from saicinpainting.evaluation.data import InpaintingDataset
from saicinpainting.training.data.datasets import make_default_train_dataloader, make_default_val_dataloader
from suppliment import initialize_gpu,Waveblock,WaveMix,HybridLoss,calc_curr_performance

from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.transforms.functional import crop, center_crop
from torchvision.transforms import Resize


parser = argparse.ArgumentParser()


parser.add_argument("-vis", "--Visual_example", default="Eval", help = "Path to save visual examples")
parser.add_argument("-batch", "--Batch_size", default=40, help = "Batch Size")
parser.add_argument("-gpu", "--GPU_number", default=-1, help = "GPU_number")
parser.add_argument("-read", "--read_path", default="/home/Drive3/Dharshan/Venv/lama/ImageNet/eval/random_medium_224/", help = "read path")
parser.add_argument("-save", "--save_path", default="output1", help = "save path")
parser.add_argument("-model", "--model_path", default=None, help = "model path")


args = parser.parse_args()
Visual_example_loc=str(args.Visual_example)
if int(args.GPU_number)==-1:
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
	device = torch.device("cuda:"+str(args.GPU_number) if torch.cuda.is_available() else "cpu")
initialize_gpu(args.GPU_number)
print(device)

TrainBatchSize=int(args.Batch_size)
if args.model_path==None:
	PATH = "WavemixModelschkpoint__IMG__12__MODEL__D7_E128_N4_C1_F128_#dwt=1_thin_mask.pth"
else:
	PATH = args.model_path





# eval_loader=InpaintingDataset(datadir="my_dataset/eval/random_thick_96/",img_suffix=".png")

ValDataLoaderConfig={'dataloader_kwargs': {'batch_size': TrainBatchSize, 'shuffle': False, 'num_workers': 2}}

# eval_loader=make_default_val_dataloader(indir="ImageNet/eval/random_medium_224/",img_suffix=".png", **ValDataLoaderConfig)
eval_loader=make_default_val_dataloader(indir=args.read_path,img_suffix=".png", **ValDataLoaderConfig)




### MODEL ARCHITECTURE ###

# Module parameters
IMAGE_SIZE=(224,224)
MASK_SIZE=12
MODEL_DEPTH=7
MODEL_EMBEDDING=128
FF_CHANNEL=128
REDUCTION_CONV=1
NUM_DWT=1


NUM_MODELS=4 # number of models not modules
# ALPHA=0.75
MAX_EVAL=10
LOAD=False

class Model(nn.Module):
	def __init__(
		self,
		*,
		num_classes,
		depth,
		# mult_dim = 32,
		mult = 2,
		ff_channel = 16,
		final_dim = 16,
		dropout = 0.,
		num_models=2
	):
		super().__init__()
		
		self.layers = nn.ModuleList([])
		for _ in range(num_models):
			self.wave = WaveMix(num_classes = 3,depth = MODEL_DEPTH,mult = 2,ff_channel = FF_CHANNEL,final_dim = MODEL_EMBEDDING,dropout = 0.5).to(device)
			if str(args.GPU_number)=="-1":
				self.wave = nn.DataParallel(self.wave)
			else:
				self.layers.append(self.wave)
		
	def forward(self, img, mask):
		x=img
		# print("%%",x.device, mask.device)
		# mask=torch.rand(96,96)
		for module in self.layers:
			x = module(x, mask) + x
			
		x=x*mask+img*(1-mask)

		return x

###########################################################################################################################################

model = Model(
	num_classes = 3,
	depth = MODEL_DEPTH,
	mult = 2,
	ff_channel = FF_CHANNEL,
	final_dim = MODEL_EMBEDDING, ## 64
	dropout = 0.5,
	num_models=NUM_MODELS
).to(device)


# PATH = base_path + str('chkpoint__IMG__'+str(MASK_SIZE)+'__MODEL__D'+str(MODEL_DEPTH)+'_E'+str(MODEL_EMBEDDING)+'_N'+str(NUM_MODELS)+'_C'+str(REDUCTION_CONV)+'_F'+str(FF_CHANNEL)+'_#dwt='+str(NUM_DWT)+extra+'.pth')

print(PATH)
model.load_state_dict(torch.load(PATH))
print("LOADED WEIGHTS!!!")
Losses=calc_curr_performance(model,eval_loader, entire_dataset=False)
Final_losses={}
for metric in Losses.keys():
	Final_losses[metric]=np.array(Losses[metric]).mean()
print("PERFORMANCE: \n",Final_losses)


Topn=100
Indices=[]

Indices=np.argsort(Losses["LPIPS"])[:Topn]

print(Indices)

for i,data in tqdm(enumerate(eval_loader)):
	if i in Indices:
		img, mask=torch.Tensor(data["image"]),torch.Tensor(data["mask"])
		h,w=img.shape[2], img.shape[3]
		if h>224 or w>224:
			h=224
			w=224
			img=Resize([224,])(img)
			mask=Resize([224,])(mask)
			img=crop(img, 0, 0, h, w)
			mask=crop(mask, 0, 0, h, w)
		ground_truth=img.clone().detach()
		img[:, :, :] = img[:, :, :] * (1-mask)
		masked_img=img

		

		out=model.forward((masked_img.reshape(-1,3,h,w)).to(device), mask.to(device))
		cv2.imwrite("Visual_example/"+Visual_example_loc+"/"+str(Losses["LPIPS"][i])+"_eval_img.png",cv2.cvtColor(img[0].permute([1,2,0]).cpu().detach().numpy()*255, cv2.COLOR_RGB2BGR))
		# print(ground_truth.shape)
		cv2.imwrite("Visual_example/"+Visual_example_loc+"/"+str(Losses["LPIPS"][i])+"_eval_gt.png",cv2.cvtColor(ground_truth[0].permute([1,2,0]).numpy()*255, cv2.COLOR_RGB2BGR))
		cv2.imwrite("Visual_example/"+Visual_example_loc+"/"+str(Losses["LPIPS"][i])+"_eval_out.png",cv2.cvtColor(out[0].permute([1,2,0]).cpu().detach().numpy()*255, cv2.COLOR_RGB2BGR))