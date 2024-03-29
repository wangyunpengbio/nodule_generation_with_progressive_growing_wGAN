# Nodule Generation with Progressive Growing wGAN
- Description: TensorFlow Implementation of progressive growing wGAN
- Function: Generate nodule images with high quality
- Citation: If you are interested in this respository, you can refer to the following article.
- Paper link: http://qims.amegroups.com/article/view/43812/html

```
Cite this article as:
Wang Y, Zhou L, Wang M, Shao C, Shi L, Yang S, Zhang Z, Feng M, Shan F, Liu L.
Combination of generative adversarial etwork and convolutional neural network for automatic subcentimeter pulmonary adenocarcinoma classification.
Quant Imaging Med Surg 2020;10(6):1249-1264. doi: 10.21037/qims-19-982
```

## Repository content

The weights of pre-trained network can be found in this repository. We also include the related scripts for automatically generating pulmonary nodules. Everyone can easily generate the image of three subtypes pulmonary nodules through simple configuration. In addition, we provide a small anonymous demo dataset.

## Install dependencies

Related packages:
```
Python >= 3.6
tensorflow-gpu == 1.12.0
cuda >= 9.0
tqdm
Pillow
numpy
```
We highly recommend using conda for doing this

```
conda create -n nodule python=3.6
conda activate nodule
pip install tensorflow-gpu==1.12.0
```

## Code Structure
- 1-dataset_display.py: transform picture from tfrecords file to png format and display
- 2-inference.py: the script for the generation of three subtypes pulmonary nodules.
- imgs/: some demo images
- **demo/weight/(Need to be downloaded from the cloud storage)**: weight of the well-trained GAN. Because of the big size of weight file, this file can be downloaded from Google drive or Baidu drive
  - Google drive：https://drive.google.com/file/d/1gucmt-J_cDgjRbCDNA0ExMibYPRCu2mg/view?usp=sharing
  - Baidu drive：https://pan.baidu.com/s/1IqkRsF5OJBkKlnEJHpaZvQ password：a01v
- demo/output/: the default output directory
- processData/: an small anonymous demo dataset, AIS-r*.tfrecords means different resolution of images
- tfutil.py: some functions needed during running process
- pix2pix/: Files related with pix2pix in the article, including the code, network architecture and demo pix2pix input dataset. 


## Usage

### Dataset display
```
$ python 1-dataset_display.py -h
usage: 1-dataset_display.py [-h] [--tfrecords TFRECORDS] [--number NUMBER] [--outdir OUTDIR]

Dataset demo pipeline: transform picture from tfrecords file to png format

optional arguments:
  -h, --help            show this help message and exit
  --tfrecords TFRECORDS the tfrecords file path
  --number NUMBER       set the number of dataset images to show
  --outdir OUTDIR       the output path


You can simply run by:

python 1-dataset_display.py --tfrecords processData/AIS/AIS-r06.tfrecords --number 10 --outdir demo/output

This will display 10 adenocarcinomas in situ (AIS) nodules images in the dataset with the resolution of 64*64 pixel and output them under output directory.
```


### Nodule generation
```
$ python 2-inference.py -h
usage: 2-inference.py [-h] [--model MODEL] [--number NUMBER] [--outdir OUTDIR]

Nodule generation pipeline

optional arguments:
  -h, --help       show this help message and exit
  --model MODEL    the model weight file path
  --number NUMBER  set the number of output images
  --outdir OUTDIR  the output path

You can simply run by:

python 2-inference.py --model demo/weight/AIS.pkl --number 10 --outdir demo/output

This can generate 10 adenocarcinomas in situ (AIS) nodules images and output them under output directory.
```

## Background
Efficient and accurate diagnosis of pulmonary adenocarcinoma before surgery is of great significance to clinicians. Although computed tomography (CT) examinations are widely used in practice, it is still difficult and time-consuming for radiologists to distinguish between different types of subcentimeter pulmonary nodules. Although many deep learning algorithms have been proposed, their superior performance depends on large amounts of data, which is difficult to collect in medical imaging area. To optimize clinical decision-making and provide small dataset algorithm design ideas, we proposed an automatic classification system for subcentimeter pulmonary adenocarcinoma, combining convolutional neural network (CNN) and generative adversarial network (GAN).

## Methods
Our system consists of two parts, GAN-based image synthesis and CNN classification. On the one hand, several existing popular GAN techniques were employed to augment the dataset and comprehensive experiments were conducted to evaluate the quality of GAN synthesis. On the other hand, our classification system processes based on 2D nodule-centered CT patches, without the need of manual labeling information. 

## Structure of the GAN
The architecture of the GAN used for pulmonary adenocarcinoma generation was shown below. The generator consisted of nine convolution layers. Firstly, 512-dimensional random noise was fed into a fully connected layer, and then the 4×4 pixels feature map was generated by the first convolution layer. Afterwards, the feature map was passed through four blocks, which was composed of two convolution layers. The detailed structure of the block was shown below.These blocks constantly doubled the height and width of the feature map, finally generating 64×64 pixels image. 

The discriminator also consisted of nine convolution layers. Mirrored with the generator, it started with 64×64 pixels image, and passed through four blocks, which was composed of two convolution layers, ended with 4×4 pixels feature map. Mini-batch discrimination was added into the final convolution layer to make training more stable. Finally, this feature map was passed through two small fully connected layers to get the last true or false target.Both generator and discriminator contained 22M parameters, with 3×3 kernel of convolution layer.

![Architecture of the progressive growing wGAN.png](https://github.com/wangyunpengbio/nodule_generation_with_progressive_growing_wGAN/raw/master/imgs/Architecture.png)
![Architecture of the progressive growing wGAN.png](https://github.com/wangyunpengbio/nodule_generation_with_progressive_growing_wGAN/raw/master/imgs/block.png)

## Latent space interpolation
Generative adversarial network is trained to produce data samples that resemble the training set. Because the number of model parameters is significantly smaller than the training data, the models are forced to discover efficient data representations. The model samples from a set of latent variables in a high dimensional space, called latent space, and generates observable data values. Generative models are often evaluated by examining samples from the latent space. Interpolation is always used to traverse between two known locations in latent space. Researches on generative models always use interpolation to prove that GAN do not just simply memorize the dataset. This approach provides smooth transitions between two decoded images. We chose two MIA images generated by GAN and do linear interpolation between their latent codes. The following figure presented the transition process of the nodule image, from the center of the lung to the adjacent lung wall.

![Latent space interpolation between two images.png](https://github.com/wangyunpengbio/nodule_generation_with_progressive_growing_wGAN/raw/master/imgs/Interpolation.png)


## Reference code
- [pgGAN pytorch](https://github.com/github-pengge/PyTorch-progressive_growing_of_gans)
- [pgGAN tensorflow](https://github.com/zhangqianhui/progressive_growing_of_gans_tensorflow)
- [pgGAN tensorflow](https://github.com/tkarras/progressive_growing_of_gans)
- [pix2pix keras](https://phillipi.github.io/pix2pix)
