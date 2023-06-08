 
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
import pywt
import torch.nn as nn
import functools
from math import ceil
import pywt
from tqdm import tqdm

from saicinpainting.evaluation.losses import base_loss
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import lpips
from torchmetrics.functional import structural_similarity_index_measure

device=0
loss_fn=None

def initialize_gpu(GPU):
	global device, loss_fn
	if GPU==-1:
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	else:
		device = torch.device('cuda:'+str(GPU))
	loss_fn = lpips.LPIPS(net='alex').to(device) # best forward scores

	print(device)
	# loss_fn = lpips.LPIPS(net='vgg').to(device)



REDUCTION_CONV=1
NUM_DWT=1 # lite

def sfb1d(lo, hi, g0, g1, mode='zero', dim=-1):
    """ 1D synthesis filter bank of an image tensor
    """
    C = lo.shape[1]
    d = dim % 4
    # If g0, g1 are not tensors, make them. If they are, then assume that they
    # are in the right order
    if not isinstance(g0, torch.Tensor):
        g0 = torch.tensor(np.copy(np.array(g0).ravel()),
                          dtype=torch.float, device=lo.device)
    if not isinstance(g1, torch.Tensor):
        g1 = torch.tensor(np.copy(np.array(g1).ravel()),
                          dtype=torch.float, device=lo.device)
    L = g0.numel()
    shape = [1,1,1,1]
    shape[d] = L
    N = 2*lo.shape[d]
    # If g aren't in the right shape, make them so
    if g0.shape != tuple(shape):
        g0 = g0.reshape(*shape)
    if g1.shape != tuple(shape):
        g1 = g1.reshape(*shape)

    s = (2, 1) if d == 2 else (1,2)
    g0 = torch.cat([g0]*C,dim=0)
    g1 = torch.cat([g1]*C,dim=0)
    if mode == 'per' or mode == 'periodization':
        y = F.conv_transpose2d(lo, g0, stride=s, groups=C) + \
            F.conv_transpose2d(hi, g1, stride=s, groups=C)
        if d == 2:
            y[:,:,:L-2] = y[:,:,:L-2] + y[:,:,N:N+L-2]
            y = y[:,:,:N]
        else:
            y[:,:,:,:L-2] = y[:,:,:,:L-2] + y[:,:,:,N:N+L-2]
            y = y[:,:,:,:N]
        y = roll(y, 1-L//2, dim=dim)
    else:
        if mode == 'zero' or mode == 'symmetric' or mode == 'reflect' or \
                mode == 'periodic':
            pad = (L-2, 0) if d == 2 else (0, L-2)
            y = F.conv_transpose2d(lo, g0, stride=s, padding=pad, groups=C) + \
                F.conv_transpose2d(hi, g1, stride=s, padding=pad, groups=C)
        else:
            raise ValueError("Unkown pad type: {}".format(mode))

    return y

def reflect(x, minx, maxx):
    """Reflect the values in matrix *x* about the scalar values *minx* and
    *maxx*.  Hence a vector *x* containing a long linearly increasing series is
    converted into a waveform which ramps linearly up and down between *minx*
    and *maxx*.  If *x* contains integers and *minx* and *maxx* are (integers +
    0.5), the ramps will have repeated max and min samples.
    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
    .. codeauthor:: Nick Kingsbury, Cambridge University, January 1999.
    """
    x = np.asanyarray(x)
    rng = maxx - minx
    rng_by_2 = 2 * rng
    mod = np.fmod(x - minx, rng_by_2)
    normed_mod = np.where(mod < 0, mod + rng_by_2, mod)
    out = np.where(normed_mod >= rng, rng_by_2 - normed_mod, normed_mod) + minx
    return np.array(out, dtype=x.dtype)

def mode_to_int(mode):
    if mode == 'zero':
        return 0
    elif mode == 'symmetric':
        return 1
    elif mode == 'per' or mode == 'periodization':
        return 2
    elif mode == 'constant':
        return 3
    elif mode == 'reflect':
        return 4
    elif mode == 'replicate':
        return 5
    elif mode == 'periodic':
        return 6
    else:
        raise ValueError("Unkown pad type: {}".format(mode))

def int_to_mode(mode):
    if mode == 0:
        return 'zero'
    elif mode == 1:
        return 'symmetric'
    elif mode == 2:
        return 'periodization'
    elif mode == 3:
        return 'constant'
    elif mode == 4:
        return 'reflect'
    elif mode == 5:
        return 'replicate'
    elif mode == 6:
        return 'periodic'
    else:
        raise ValueError("Unkown pad type: {}".format(mode))

def afb1d(x, h0, h1, mode='zero', dim=-1):
    """ 1D analysis filter bank (along one dimension only) of an image
    Inputs:
        x (tensor): 4D input with the last two dimensions the spatial input
        h0 (tensor): 4D input for the lowpass filter. Should have shape (1, 1,
            h, 1) or (1, 1, 1, w)
        h1 (tensor): 4D input for the highpass filter. Should have shape (1, 1,
            h, 1) or (1, 1, 1, w)
        mode (str): padding method
        dim (int) - dimension of filtering. d=2 is for a vertical filter (called
            column filtering but filters across the rows). d=3 is for a
            horizontal filter, (called row filtering but filters across the
            columns).
    Returns:
        lohi: lowpass and highpass subbands concatenated along the channel
            dimension
    """
    C = x.shape[1]
    # Convert the dim to positive
    d = dim % 4
    s = (2, 1) if d == 2 else (1, 2)
    N = x.shape[d]
    # If h0, h1 are not tensors, make them. If they are, then assume that they
    # are in the right order
    if not isinstance(h0, torch.Tensor):
        h0 = torch.tensor(np.copy(np.array(h0).ravel()[::-1]),
                          dtype=torch.float, device=x.device)
    if not isinstance(h1, torch.Tensor):
        h1 = torch.tensor(np.copy(np.array(h1).ravel()[::-1]),
                          dtype=torch.float, device=x.device)
    L = h0.numel()
    L2 = L // 2
    shape = [1,1,1,1]
    shape[d] = L
    # If h aren't in the right shape, make them so
    if h0.shape != tuple(shape):
        h0 = h0.reshape(*shape)
    if h1.shape != tuple(shape):
        h1 = h1.reshape(*shape)
    h = torch.cat([h0, h1] * C, dim=0)

    if mode == 'per' or mode == 'periodization':
        if x.shape[dim] % 2 == 1:
            if d == 2:
                x = torch.cat((x, x[:,:,-1:]), dim=2)
            else:
                x = torch.cat((x, x[:,:,:,-1:]), dim=3)
            N += 1
        x = roll(x, -L2, dim=d)
        pad = (L-1, 0) if d == 2 else (0, L-1)
        lohi = F.conv2d(x, h, padding=pad, stride=s, groups=C)
        N2 = N//2
        if d == 2:
            lohi[:,:,:L2] = lohi[:,:,:L2] + lohi[:,:,N2:N2+L2]
            lohi = lohi[:,:,:N2]
        else:
            lohi[:,:,:,:L2] = lohi[:,:,:,:L2] + lohi[:,:,:,N2:N2+L2]
            lohi = lohi[:,:,:,:N2]
    else:
        # Calculate the pad size
        outsize = pywt.dwt_coeff_len(N, L, mode=mode)
        p = 2 * (outsize - 1) - N + L
        if mode == 'zero':
            # Sadly, pytorch only allows for same padding before and after, if
            # we need to do more padding after for odd length signals, have to
            # prepad
            if p % 2 == 1:
                pad = (0, 0, 0, 1) if d == 2 else (0, 1, 0, 0)
                x = F.pad(x, pad)
            pad = (p//2, 0) if d == 2 else (0, p//2)
            # Calculate the high and lowpass
            lohi = F.conv2d(x, h, padding=pad, stride=s, groups=C)
        elif mode == 'symmetric' or mode == 'reflect' or mode == 'periodic':
            pad = (0, 0, p//2, (p+1)//2) if d == 2 else (p//2, (p+1)//2, 0, 0)
            x = mypad(x, pad=pad, mode=mode)
            lohi = F.conv2d(x, h, stride=s, groups=C)
        else:
            raise ValueError("Unkown pad type: {}".format(mode))

    return lohi



class AFB2D(Function):
    """ Does a single level 2d wavelet decomposition of an input. Does separate
    row and column filtering by two calls to
    :py:func:`pytorch_wavelets.dwt.lowlevel.afb1d`
    Needs to have the tensors in the right form. Because this function defines
    its own backward pass, saves on memory by not having to save the input
    tensors.
    Inputs:
        x (torch.Tensor): Input to decompose
        h0_row: row lowpass
        h1_row: row highpass
        h0_col: col lowpass
        h1_col: col highpass
        mode (int): use mode_to_int to get the int code here
    We encode the mode as an integer rather than a string as gradcheck causes an
    error when a string is provided.
    Returns:
        y: Tensor of shape (N, C*4, H, W)
    """
    @staticmethod
    def forward(ctx, x, h0_row, h1_row, h0_col, h1_col, mode):
        ctx.save_for_backward(h0_row, h1_row, h0_col, h1_col)
        ctx.shape = x.shape[-2:]
        mode = int_to_mode(mode)
        ctx.mode = mode
        lohi = afb1d(x, h0_row, h1_row, mode=mode, dim=3)
        y = afb1d(lohi, h0_col, h1_col, mode=mode, dim=2)
        s = y.shape
        y = y.reshape(s[0], -1, 4, s[-2], s[-1])
        low = y[:,:,0].contiguous()
        highs = y[:,:,1:].contiguous()
        return low, highs

    @staticmethod
    def backward(ctx, low, highs):
        dx = None
        if ctx.needs_input_grad[0]:
            mode = ctx.mode
            h0_row, h1_row, h0_col, h1_col = ctx.saved_tensors
            lh, hl, hh = torch.unbind(highs, dim=2)
            lo = sfb1d(low, lh, h0_col, h1_col, mode=mode, dim=2)
            hi = sfb1d(hl, hh, h0_col, h1_col, mode=mode, dim=2)
            dx = sfb1d(lo, hi, h0_row, h1_row, mode=mode, dim=3)
            if dx.shape[-2] > ctx.shape[-2] and dx.shape[-1] > ctx.shape[-1]:
                dx = dx[:,:,:ctx.shape[-2], :ctx.shape[-1]]
            elif dx.shape[-2] > ctx.shape[-2]:
                dx = dx[:,:,:ctx.shape[-2]]
            elif dx.shape[-1] > ctx.shape[-1]:
                dx = dx[:,:,:,:ctx.shape[-1]]
        return dx, None, None, None, None, None

# cuda0 = torch.device('cuda:0')
# device = cuda0

def prep_filt_afb2d(h0_col, h1_col, h0_row=None, h1_row=None, device=device):
    """
    Prepares the filters to be of the right form for the afb2d function.  In
    particular, makes the tensors the right shape. It takes mirror images of
    them as as afb2d uses conv2d which acts like normal correlation.
    Inputs:
        h0_col (array-like): low pass column filter bank
        h1_col (array-like): high pass column filter bank
        h0_row (array-like): low pass row filter bank. If none, will assume the
            same as column filter
        h1_row (array-like): high pass row filter bank. If none, will assume the
            same as column filter
        device: which device to put the tensors on to
    Returns:
        (h0_col, h1_col, h0_row, h1_row)
    """
    h0_col, h1_col = prep_filt_afb1d(h0_col, h1_col, device)
    if h0_row is None:
        h0_row, h1_col = h0_col, h1_col
    else:
        h0_row, h1_row = prep_filt_afb1d(h0_row, h1_row, device)

    h0_col = h0_col.reshape((1, 1, -1, 1))
    h1_col = h1_col.reshape((1, 1, -1, 1))
    h0_row = h0_row.reshape((1, 1, 1, -1))
    h1_row = h1_row.reshape((1, 1, 1, -1))
    return h0_col, h1_col, h0_row, h1_row


def prep_filt_afb1d(h0, h1, device=device):
    """
    Prepares the filters to be of the right form for the afb2d function.  In
    particular, makes the tensors the right shape. It takes mirror images of
    them as as afb2d uses conv2d which acts like normal correlation.
    Inputs:
        h0 (array-like): low pass column filter bank
        h1 (array-like): high pass column filter bank
        device: which device to put the tensors on to
    Returns:
        (h0, h1)
    """
    h0 = np.array(h0[::-1]).ravel()
    h1 = np.array(h1[::-1]).ravel()
    t = torch.get_default_dtype()
    h0 = torch.tensor(h0, device=device, dtype=t).reshape((1, 1, -1))
    h1 = torch.tensor(h1, device=device, dtype=t).reshape((1, 1, -1))
    return h0, h1

class DWTForward(nn.Module):
    """ Performs a 2d DWT Forward decomposition of an image
    Args:
        J (int): Number of levels of decomposition
        wave (str or pywt.Wavelet or tuple(ndarray)): Which wavelet to use.
            Can be:
            1) a string to pass to pywt.Wavelet constructor
            2) a pywt.Wavelet class
            3) a tuple of numpy arrays, either (h0, h1) or (h0_col, h1_col, h0_row, h1_row)
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme
        """
    def __init__(self, J=1, wave='db1', mode='zero'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            h0_col, h1_col = wave.dec_lo, wave.dec_hi
            h0_row, h1_row = h0_col, h1_col
        else:
            if len(wave) == 2:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = h0_col, h1_col
            elif len(wave) == 4:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = wave[2], wave[3]

        # Prepare the filters
        filts = prep_filt_afb2d(h0_col, h1_col, h0_row, h1_row)
        self.register_buffer('h0_col', filts[0])
        self.register_buffer('h1_col', filts[1])
        self.register_buffer('h0_row', filts[2])
        self.register_buffer('h1_row', filts[3])
        self.J = J
        self.mode = mode

    def forward(self, x):
        """ Forward pass of the DWT.
        Args:
            x (tensor): Input of shape :math:`(N, C_{in}, H_{in}, W_{in})`
        Returns:
            (yl, yh)
                tuple of lowpass (yl) and bandpass (yh) coefficients.
                yh is a list of length J with the first entry
                being the finest scale coefficients. yl has shape
                :math:`(N, C_{in}, H_{in}', W_{in}')` and yh has shape
                :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. The new
                dimension in yh iterates over the LH, HL and HH coefficients.
        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.
        """
        yh = []
        ll = x
        mode = mode_to_int(self.mode)

        # Do a multilevel transform
        for j in range(self.J):
            # Do 1 level of the transform
            ll, high = AFB2D.apply(
                ll, self.h0_col, self.h1_col, self.h0_row, self.h1_row, mode)
            yh.append(high)

        return ll, yh

from numpy.lib.function_base import hamming
class Waveblock(nn.Module):
    def __init__(
        self,
        *,
        mult = 2,
        ff_channel = 16,
        final_dim = 16,
        dropout = 0.5,
    ):
        super().__init__()
        
      
        self.feedforward = nn.Sequential(
                nn.Conv2d(final_dim, final_dim*mult,1),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv2d(final_dim*mult, ff_channel, 1),
                nn.BatchNorm2d(ff_channel)
            
            )

        self.reduction = nn.Conv2d(final_dim, int(final_dim/4), REDUCTION_CONV, padding=int(REDUCTION_CONV/2)).to(device) ## CHANGE to 4
        
        self.ff1 = nn.ConvTranspose2d(ff_channel, int(final_dim/NUM_DWT), 4, stride=2, padding=1)
        self.depthconv = nn.Sequential(
            nn.Conv2d(final_dim, final_dim, 5, groups=final_dim, padding=2),
            nn.GELU(),
            nn.BatchNorm2d(final_dim),
        )
        
    def forward(self, x):
        b, c, h, w = x.shape
        
        x = self.reduction(x)

        xf1 = DWTForward(J=1, mode='zero', wave='db1').to(device)

        Y1, Yh = xf1(x)
        x1 = torch.reshape(Yh[0], (b, int(c*3/4), int(h/2), int(h/2)))
        x1 = torch.cat((Y1,x1), dim = 1)

        x1 = self.feedforward(x1)
        x1 = self.ff1(x1)
        
        if NUM_DWT==3:
            x = torch.cat((x1,x2,x3), dim = 1)
        elif NUM_DWT==2:
            x = torch.cat((x1,x2), dim = 1)
        elif NUM_DWT==1:
            x = x1
        
        x = self.depthconv(x)
        
        return x

### MODEL ###


class WaveMix(nn.Module):
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
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(Waveblock(mult = mult, ff_channel = ff_channel, final_dim = final_dim, dropout = dropout))

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(final_dim*2,int(final_dim/2), 4, stride=2, padding=1),
            nn.ConvTranspose2d(int(final_dim/2), int(final_dim/2), 3, stride=1, padding=1),
            nn.BatchNorm2d(int(final_dim/2))
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(int(final_dim/2)+3,num_classes, 5, stride=1, padding=2),
        )
        
        self.conv = nn.Sequential(
            nn.Conv2d(5, int(final_dim/2), 3, 1, 1),
            nn.Conv2d(int(final_dim/2), final_dim, 3, 2, 1),
        )
        

    def forward(self, img, mask):
        global device

        ones = torch.ones(img.size(0), 1, img.size(2), img.size(3))
        size_img=img.shape[2]

        if img.is_cuda:
            ones = ones.to(device)
            mask = mask.to(device)

        mask=mask.reshape(-1,1,size_img,size_img)
        xnow = torch.cat([img, ones, mask], dim=1)
        x = self.conv(xnow)   
        first_conv=x

        for attn in self.layers:
            x = attn(x)+x

        x=torch.cat([x,first_conv],dim=1)  # skip connection
        x = self.decoder(x)
        x=torch.cat([x,img],dim=1)  # skip connection
        out=self.decoder2(x)

        return out



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
            self.wave = WaveMix(num_classes = num_classes,depth = depth,mult = 2,ff_channel = ff_channel,final_dim = final_dim,dropout = 0.5)
            self.layers.append(self.wave)
        
    def forward(self, img, mask):
        x=img

        for module in self.layers:
            x = module(x, mask) + x
            
        x=x*mask+img*(1-mask)

        return x

########################################################################

### EVAL METRICES ###

from piqa import SSIM
from skimage.metrics import structural_similarity as ssim_eval
from skimage.metrics import peak_signal_noise_ratio as psnr_eval
from piqa.utils import set_debug
set_debug(False)

def EvalMetrics(out,gt):
    losses={}
    psnr=[]
    ssim=[]
    l1=[]
    l2=[]
    fid=[]
    FID_score=base_loss.FIDScore()

    h=gt.shape[2]
    gt=gt.reshape(-1,3,h,h)
    for x,y in zip(out,gt):
        psnr.append(psnr_eval(y.numpy(),x.numpy()))
        x=x*1.0
        y=y*1.0
        l1.append(nn.L1Loss()(y,x))
        l2.append(nn.MSELoss()(y,x))
    
    losses["L1"]=np.array(l1).mean()
    losses["L2"]=np.array(l2).mean()
    losses["PSNR"]=np.array(psnr).mean()
    losses["SSIM"]=np.array(structural_similarity_index_measure(out,gt)).mean()
    losses["LPIPS"]=loss_fn(gt.to(device),out.to(device)).mean().item()
    return losses
    

class HybridLoss(SSIM):
    def __init__(self, alpha=0.5):
        super(HybridLoss, self).__init__()
        self.alpha=alpha
        

    def forward(self, x,masks, y):
        l_lpips = loss_fn(x,y)
        l_lpips = l_lpips.mean() 
        l1 = l_lpips + (1-self.alpha)*(nn.L1Loss()(x , y )) + (self.alpha)*(1. - super().forward(x*masks, y*masks))*10
        losses=l1*1000

        return losses

def calc_curr_performance(model,valloader, save_imgs=False, save_path="Visual_example/Eval/", entire_dataset=False):
    global device
    Losses={"L1":[],"L2":[],"PSNR":[],"SSIM":[],"LPIPS":[]}
    j=0
    kernel = np.ones((1, 1), np.uint8)
    for i, data in enumerate(tqdm(valloader)):
        img, mask=torch.Tensor(data["image"]),torch.Tensor(data["mask"])
        
        h=img.shape[2]

        ground_truth=img.clone().detach()
        # print(ground_truth.shape)
        img[:, :, :] = img[:, :, :] * (1-mask)
        masked_img=img
        # print(ground_truth.max())

        
        out=model.forward((masked_img.reshape(-1,3,h,h)).to(device), mask.to(device))
        losses=EvalMetrics(out.cpu().detach(),ground_truth)

        if not entire_dataset and i >100:
            break

        for metric in losses.keys():
            Losses[metric].append(losses[metric])
        
    return Losses