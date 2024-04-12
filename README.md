# VisionGuard (CV & NLP Project)

## Authors:

- Iñigo Aduna (r0973686)
- Ahmad BeigRezaei (r0969764)
- Minoo Safaeikouchaksaraei (r0972740)

## Supervisor:

- Floris De Feyter: floris.defeyter@kuleuven.be

## 1.- Business Understanding

In the expansive realm of human-machine interaction, the challenge of discerning human emotions and levels of attentiveness is crucial, particularly in critical settings such as driving. As we advance deeper into the era of automation and artificial intelligence, incorporating Facial Expression Recognition (FER) technology into automotive systems emerges as a promising strategy to boost safety and enhance communication between drivers and their vehicles. Utilizing the latest in computer vision and deep learning, FER technology can fundamentally transform our interaction with automotive technology, leading to safer roadways and more intuitive driving experiences.

In the ever-changing landscape of vehicular safety, continuously monitoring a driver’s emotional state and attention level is both a technological imperative and a necessity. Traditional driver monitoring techniques tend to be intrusive and often fail to deliver real-time, actionable insights. The emergence of advanced computer vision and natural language processing technologies presents a substantial opportunity to improve driver safety and vehicle interaction through FER technology. However, continuous surveillance introduces privacy and data security concerns, requiring a thoughtful approach that optimizes functionality while minimizing intrusiveness.

We will train three distinct and efficient models for the Facial Emotion Recognition (FER) task. Two models will be based on Facial Detection combined with Convolutional Neural Networks (CNN), and one model will utilize Vision Transformers (ViT). The project will be developed in Jupyter Notebooks, hosted on the Flemish Supercomputer with a runtime environment provided by the NVIDIA A100 GPU. Continuous monitoring introduces concerns regarding the privacy and security of individuals. To address these issues, we will employ two distinct approaches. Firstly, using the ImageNet-C blur function, we will test how blurring images at different severity levels affects the performance of our models. Secondly, we will evaluate the performance of a small model designed to fit on an edge device for edge processing and compare its outcomes with the other models.


## 2.- Data understanding

The dataset is formed by eight different classes (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral, Contempt) by merging these 3 datsets. 

2.1.1 **Extended Cohn-Kanade (CK+):** This dataset includes images extracted from 593 video sequences of a total of 123 different subjects, ranging in age from 18 to 50 years, and featuring a diverse mix of genders and ethnic backgrounds. Each video captures the subject transitioning from a neutral facial expression to a targeted peak expression. Videos are recorded at 30 frames per second (FPS) and are available in resolutions of either 640x490 or 640x480 pixels. These images are categorized into seven primary expression labels (Anger, Disgust, Fear, Happy, Sadness, Surprise, Contempt).

2.1.2 **FER-2013:** The dataset comprises approximately 35,000 RGB facial images, each standardized to a resolution of 48x48 pixels. These images are categorized into seven primary expression labels (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral). The distribution of images across these categories varies, with the 'Disgust' expression having the fewest examples at approximately 600 images. In contrast, the other expressions each have about 5,000 samples.

2.1.3 **AffectNet:** AffectNet contains a comprehensive collection of 41,000 facial expression images. These images are classified into eight distinct categories of facial expressions, which are as follows (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral, Contempt). Additionally, each image in the dataset is annotated with measures of valence and arousal intensity, providing deeper insight into the emotional states depicted.

### Notebooks: 
| Name              | Description                                                                                      |
|-------------------|--------------------------------------------------------------------------------------------------|
| Data_Integration  | This notebook illustrates the features of the original datasets from our projects and integrates them into a consolidated dataset. |

## 3.- Data preparation

The models used in this project are pre-trained on the ImageNet dataset for object detection in images of size 224x224x3. To adapt our dataset to these models, we have implemented several image processing steps.

3.1 **Image Scaling:** All images in our dataset will be scaled to a resolution of 224x224 pixels to match the input size used during the pre-training of our models.

3.2 **Data Channels:** We will convert the images in our dataset to use RGB channels to ensure consistency with the pre-training conditions.

3.3 **Data Augmentation:** To enhance the variety and balance of our dataset, we will apply various data augmentation techniques. These include adjustments to brightness, as well as the application of rotation, scaling, translation, and zooming.

### Notebooks: 
| Name              | Description                                                                                      |
|-------------------|--------------------------------------------------------------------------------------------------|
| Data_Analysis     | This section conducts data analysis on the integrated dataset to finalize the proposed dataset.  |
| Preprocessing     | The preprocessing phase converts image formats from grayscale to RGB, ensuring proper distribution of values across channels. It converts .png images to .jpg and applies data augmentation techniques. |
| Corruption     | To evaluate how blurring affects images at different severity levels, we will use the blur function (img, s) from ImageNet-C. We will generate six distinct test sets, one for each level of severity corruption. |


## 4.- Modelling

## 5.- Evaluation

## 6.- Deployment