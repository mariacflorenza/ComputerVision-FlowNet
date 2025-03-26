# FlowNet: A Convolutional Neural Network Approach to Optical Flow

## Introduction  

This project focuses on using the FlowNet model for optical flow estimation and applying it to track objects in image sequences. Optical flow is a crucial component in computer vision, used to estimate the motion of objects between consecutive frames in a video sequence. The FlowNet architecture, originally proposed by Dosovitskiy et al., has been widely adopted for its effectiveness in predicting dense optical flow.

### Project Overview
The primary goal of this project is to utilize the FlowNet model to estimate optical flow and apply it for tracking objects in image sequences. To enhance the performance of optical flow estimation, we fine-tune a pre-trained FlowNet model using custom datasets. By leveraging the pre-trained weights, we aim to adapt the model to specific scenarios and improve its accuracy on new data. The fine-tuning process involves freezing most of the layers in the network and only updating the weights of the final layers to better capture the nuances of the new dataset.

## Prerequisites

Ensure you have the following dependencies installed:

```
torch>=1.2
torchvision
numpy
spatial-correlation-sampler>=0.2.1
tensorboard
imageio
argparse
path
tqdm
scipy
```

To install the dependencies, run:

```
pip install -r requirements.txt
```

## Fine Tuning
Usage
To fine-tune the FlowNet model, run the following command:
```
python3 fine_tune.py --pretrained path/to/pretrained/model.pth
```
Replace path/to/pretrained/model.pth with the actual path to your pre-trained FlowNet model. For example : flownets_EPE1.951.pth
## Optical Flow Estimation
We use the FlowNet model from this GitHub repository: FlowNetPytorch, where you can also find pre-trained models.  

To calculate the optical flow using the specified model and save the results within the image folder, creating a flow directory, run the following command:
```
python run_inference.py --data path/to/image/folder --pretrained path/to/pretrained/model.pth
```

Replace path/to/image/folder with the path to your image folder and path/to/pretrained/model.pth with the path to your pre-trained FlowNet model.

## Results


## Acknowledgements
This project was developed as part of the Computer Vision course led and supervised by Pierre-Henri Conze at IMT Atlantique.

## Authors
Catalina ARDILA, email: <dely.ardila-medina@imt-atlantique.net>

Maria FLORENZA, email: <maria.florenza-lamberti@imt-atlantique.net>

Nhan NGUYEN, email: <nhan.nguyen@imt-atlantique.net>
