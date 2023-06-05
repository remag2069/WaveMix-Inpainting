# WaveMix-Inpainting
 
  




### Model Architecture 

![image](https://github.com/remag2069/WaveMix-Inpainting/blob/main/resources/model.png)

Image inpainting refers to the synthesis of missing regions in an image, which can help restore occluded or degraded areas and also serve as a precursor task for self-supervision. The current state-of-the-art models for image inpainting are computationally heavy as they are based on vision transformer backbones in adversarial or diffusion settings. This paper diverges from vision transformers by using a computationally-efficient WaveMix-based fully convolutional architecture, which uses a 2D-discrete wavelet transform (DWT) for spatial and multi-resolution token-mixing along with convolutional layers. The proposed model outperforms the current-state-of-the-art models for large mask inpainting on reconstruction quality while also using less than half the parameter count and considerably lower training and evaluation times. 


## Parameters <!-- Have to change -->

- `num_classes`: int.  
Number of classes to classify/segment.
- `depth`: int.  
Number of WaveMix blocks.
- `mult`: int.  
Expansion of channels in the MLP (FeedForward) layer. 
- `ff_channel`: int.  
No. of output channels from the MLP (FeedForward) layer. 
- `final_dim`: int.  
Final dimension of output tensor after initial Conv layers. Channel dimension when tensor is fed to WaveBlocks.
- `dropout`: float between `[0, 1]`, default `0.`.  
Dropout rate. 
- `level`: int.  
Number of levels of 2D wavelet transform to be used in Waveblocks. Currently supports levels from 1 to 4.
- `stride`: int.  
Stride used in the initial convolutional layers to reduce the input resolution before being fed to Waveblocks. 
- `initial_conv`: str.  
Deciding between strided convolution or patchifying convolutions in the intial conv layer. Used for classification. 'pachify' or 'strided'.
- `patch_size`: int.  
Size of each non-overlaping patch in case of patchifying convolution. Should be a multiple of 4.


#### Cite the following papers 
```
@misc{
kumar2023resource,
title={Resource efficient image inpainting},
author={Dharshan Sampath Kumar and Pranav Jeevan P and Amit Sethi},
year={2023},
url={https://openreview.net/forum?id=OJILbuOodvm}
}


@misc{
p2022wavemix,
title={WaveMix: Multi-Resolution Token Mixing for Images},
author={Pranav Jeevan P and Amit Sethi},
year={2022},
url={https://openreview.net/forum?id=tBoSm4hUWV}
}


<!-- YET TO CITE LAMA -->

``` 
