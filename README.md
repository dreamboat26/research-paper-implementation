# A-Neural-Algorithm-of-Artistic-Style-Paper-Implementation
Tensorflow implementation of paper "A Neural Algorithm of Artistic Style" (https://arxiv.org/abs/1508.06576)

#### In this notebook, we'll implement the paper and reconstruct the results of the said paper. The steps of the process is as follows. Also, the notebook is created to facilitate _self-learning_ approach.

_Step 1: Preprocessing the input image_

_Step 2: Computing the output for selected layers for the content image and all the layers for style image._

_Step 3: What are loss functions in this problem and computing the loss functions._

_Step 3A: Content Loss for reconstruction of the content image._
      
_Step 3B: Style Loss for reconstruction of the style from a style image irrespective of content placement of the image._
      
_Step 4: Creating combined Tensorflow model, running it to minimize both the losses and optimize the input noise variable._

![](https://github.com/meet-minimalist/A-Neural-Algorithm-of-Artistic-Style-Paper-Implementation/blob/master/Final%20Results.jpg)
![](https://github.com/meet-minimalist/A-Neural-Algorithm-of-Artistic-Style-Paper-Implementation/blob/master/final_gif.gif)

#### Files:
1. Final Results.jpg - Combined image for all the results. 
2. helper.py - Used for pre-processing the image and post-processing the image
3. tf_helper.py - Used to compute the layer wise output for a given image
4. paper folder - contains the paper
5. tensorflow_vgg folder - contains the helper vgg16_avg_pool.py function to load the pre-trained weights ".npy" file
6. image_resources/content - contrains content image files used as a content images in style transfer
7. image_resources/style - contains style image files used as a style image in style transfer
8. image_resources/outputs - contains outputs of the notebook.
9. Other resources - contains resources for notebook and cut outs of paper.

#### References:
1. Paper link: https://arxiv.org/abs/1508.06576

2. VGG16 Tensorflow Model - https://github.com/machrisaa/tensorflow-vgg
Pre-trained VGG16 tensorflow model along with helper files. Big shoutout to the owner. Also, vgg16.npy can be downloaded from the link provided in this repository. I have modified the vgg16.py file to facilitate average pooling instead of max pooling.

3. Denoising loss suggestion - https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/15_Style_Transfer.ipynb
