# VisionGuard (CV & NLP Project)

## Authors:

- Iñigo Aduna (r0973686)
- Ahmad BeigRezaei (r0969764)
- Minoo Safaeikouchaksaraei (r0972740)

## Supervisor:

- Floris De Feyter: floris.defeyter@kuleuven.be

## 1.- Business Understanding

<div align="justify">In the expansive realm of human-machine interaction, the challenge of discerning human emotions and levels of attentiveness is crucial, particularly in critical settings such as driving. As we advance deeper into the era of automation and artificial intelligence, incorporating Facial Expression Recognition (FER) technology into automotive systems emerges as a promising strategy to boost safety and enhance communication between drivers and their vehicles. Utilizing the latest in computer vision and deep learning, FER technology can fundamentally transform our interaction with automotive technology, leading to safer roadways and more intuitive driving experiences.</div>

<div align="justify">In the ever-changing landscape of vehicular safety, continuously monitoring a driver’s emotional state and attention level is both a technological imperative and a necessity. Traditional driver monitoring techniques tend to be intrusive and often fail to deliver real-time, actionable insights. The emergence of advanced computer vision and natural language processing technologies presents a substantial opportunity to improve driver safety and vehicle interaction through FER technology. However, continuous surveillance introduces privacy and data security concerns, requiring a thoughtful approach that optimizes functionality while minimizing intrusiveness.</div>

<div align="justify">We will train an efficient model for the Facial Emotion Recognition (FER) task using the VGG-16 architecture. The project will be developed in Jupyter Notebooks, hosted on the Flemish Supercomputer with a runtime environment provided by the NVIDIA A100 GPU. To address privacy and security concerns regarding continuous monitoring, we will employ two distinct approaches. Firstly, using the ImageNet-C blur function, we will test how blurring images at different severity levels affects the performance of our model. Secondly, we will evaluate the performance of the VGG-16 model in different configurations to ensure robustness and effectiveness in real-world applications.</div>


## 2.- Data understanding

The dataset is formed by eight different classes (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral, Contempt) by merging these 3 datasets.

### 2.1.1 Extended Cohn-Kanade (CK+):

<div align="justify">This dataset includes images extracted from 593 video sequences of a total of 123 different subjects, ranging in age from 18 to 50 years, and featuring a diverse mix of genders and ethnic backgrounds. Each video captures the subject transitioning from a neutral facial expression to a targeted peak expression. Videos are recorded at 30 frames per second (FPS) and are available in resolutions of either 640x490 or 640x480 pixels. These images are categorized into seven primary expression labels (Anger, Disgust, Fear, Happy, Sadness, Surprise, Contempt).</div>

### 2.1.2 FER-2013:

<div align="justify">The dataset comprises approximately 35,000 RGB facial images, each standardized to a resolution of 48x48 pixels. These images are categorized into seven primary expression labels (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral). The distribution of images across these categories varies, with the 'Disgust' expression having the fewest examples at approximately 600 images. In contrast, the other expressions each have about 5,000 samples.</div>

### 2.1.3 AffectNet:

<div align="justify">AffectNet contains a comprehensive collection of 41,000 facial expression images. These images are classified into eight distinct categories of facial expressions, which are as follows (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral, Contempt). Additionally, each image in the dataset is annotated with measures of valence and arousal intensity, providing deeper insight into the emotional states depicted.</div>

### Notebooks:

| Name              | Description                                                                                      |
|-------------------|--------------------------------------------------------------------------------------------------|
| Data_Integration  | This notebook illustrates the features of the original datasets from our projects and integrates them into a consolidated dataset. |

## 3.- Data preparation

The models used in this project are pre-trained on the ImageNet dataset for object detection in images of size 224x224x3. To adapt our dataset to these models, we have implemented several image processing steps.

### 3.1 Image Scaling:

<div align="justify">All images in our dataset will be scaled to a resolution of 224x224 pixels to match the input size used during the pre-training of our models.</div>

### 3.2 Data Channels:

<div align="justify">We will convert the images in our dataset to use RGB channels to ensure consistency with the pre-training conditions.</div>

### 3.3 Data Augmentation:

<div align="justify">To enhance the variety and balance of our dataset, we will apply various data augmentation techniques. These include adjustments to brightness, as well as the application of rotation, scaling, translation, and zooming.</div>

### Notebooks:

| Name              | Description                                                                                      |
|-------------------|--------------------------------------------------------------------------------------------------|
| Data_Analysis     | This section conducts data analysis on the integrated dataset to finalize the proposed dataset.  |
| Preprocessing     | The preprocessing phase converts image formats from grayscale to RGB, ensuring proper distribution of values across channels. It converts .png images to .jpg and applies data augmentation techniques. |
| Corruption        | To evaluate how blurring affects images at different severity levels, we will use the blur function (img, s) from ImageNet-C. We will generate six distinct test sets, one for each level of severity corruption. |

## 4.- Modelling

## Modelling
We utilized the VGG-16 architecture for our emotion classification model. Below are the notebooks used in this stage:

### Notebooks:

| Name                                | Description                                                                                                                           |
|-------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| on_AffectNetFinal-Blurring-lv4.ipynb | Implements and trains the VGG-16 architecture using data augmentation techniques with blurring to enhance the model's robustness against image variations. |
| on_AffectNet-Final-nonblur.ipynb    | Details the implementation and training of VGG-16 without blurring techniques, providing a comparison point to evaluate the effectiveness of data augmentation with blurring. |
| VGG16_Final_VSC_Server.ipynb        | Consolidates the final results of the VGG-16 training, including performance metrics and visualization of the results.                |
| VGG16_BinarryClassification_AffNetBinary.ipynb   |  Implements and trains the VGG-16 architecture on binary classification task of AffNetBinary Dataset. |


## 5.- Evaluation
| Name              | Description                                                                                      |
|-------------------|--------------------------------------------------------------------------------------------------|
| Inferance.ipynb        | Load our trained models from saved .pth files and make inferance of the model.        |

## 6.- Project References

This project is based on several research papers and code repositories that are pivotal for the advancement of attention mechanisms and robustness in artificial intelligence models. Below are the key references used:

### 6.1 Academic Articles

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). "Attention Is All You Need." *Neural Information Processing Systems (NIPS)*. Available at: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

2. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." *International Conference on Learning Representations*. Available at: [https://openreview.net/forum?id=YicbFdNTTy](https://openreview.net/forum?id=YicbFdNTTy)

3. Li, S., Zhao, W., & Roy-Chowdhury, A. K. (2021). "Vision Transformers for Facial Emotion Recognition." *Conference on Computer Vision and Pattern Recognition*. Link not available.

### 6.2 GitHub Repositories

- [Vision Transformers for Facial Emotion Recognition](https://github.com/kode-git/vfer): A repository implementing Vision Transformers for facial emotion recognition.

- [Imagenet -C ](https://github.com/hendrycks/robustness): This repository contains implementations and techniques to enhance the robustness of machine learning models through data augmentation.




