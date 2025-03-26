# Visual tracking of video objects using FlowNet

## Table of Contents

1. [Introduction](#introduction)
2. [Optical Flow Estimation](#optical-flow-estimation)
3. [Self-Supervised Fine-tuning Strategy](#self-supervised-fine-tuning-strategy)
4. [Results Visualization and Inference Notebook](#results-visualization-and-inference-notebook)

### Introduction
It focuses on using the FlowNet model for optical flow estimation and applying it to track objects in image sequences. Optical flow is a crucial component in computer vision, used to estimate the motion of objects between consecutive frames in a video sequence. The [FlowNet architecture](https://arxiv.org/pdf/1504.06852), originally proposed by Dosovitskiy et al., has been widely adopted for its effectiveness in predicting dense optical flow.

The primary goal of this project is:
1. To utilize FlowNet model to estimate optical flow and apply it for tracking objects in image sequences.
2. To enhance the performance of optical flow estimation, we **fine-tune** a **pre-trained FlowNet model** using **custom datasets**. By leveraging the pre-trained weights, we aim to adapt the model to specific scenarios and improve its accuracy on new data. The fine-tuning process involves freezing most of the layers in the network and only updating the weights of the final layers to better capture the nuances of the new dataset.

**!! Note**: 
1. For this project, the majority of the code is written in Python. Moreoever, we worked mainly on **Jupyter Notebook** and **Visual Studio Code**. However, due to the limitted resources, we manage to **use only the CPU** (the runtime type).
2. The simulated results were for the **FlowNetS** architecture because of time constraint of this project.
3. Because of the large amount of data for `sequences-train` and `sequences-test`, we will provide you the link to access to these sequences through Google Drive: [`sequences-train`](https://drive.google.com/drive/folders/1equxxDidVitH5tJERKP7FZSi8OmI9VQa?usp=sharing) and [`sequences-test`](https://drive.google.com/drive/folders/1rMXW_fZVQSXOc7bsaGjwEDFWf8LDI9iK?usp=sharing)
 
**!! Slide**: You can also find the final presentation of this project [here](./final-restitution-team-2.pdf).

### Prerequisites

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

## Optical Flow Estimation
We use the FlowNet model from this GitHub repository: [FlowNetPytorch](https://github.com/ClementPinard/FlowNetPytorch), where you can also find [pre-trained models](https://drive.google.com/drive/folders/1dTpSyc7rIYYG19p1uiDfilcsmSPNy-_3).  

To calculate the optical flow using the specified model and save the results within the image folder, creating a flow directory, run the following command:
```
python3 run_inference.py <sequence_name> <mode>
!python3 run_inference.py --sequences_path /path/to/sequence --model_path /path/to/pretrained/model.pth  --mode <sequential/direct> --sequence <sequence_name>
```
- `--sequences_path`: Path to specify the sequences from the training (`train`) or test (`test`) set.  
- `--model_path`: Path to the pretrained FlowNet model
- `--mode`: Tracking mode (`direct` or `sequential`).  
- `--sequence`: Specifies the category that you want to use between these following categories: `bear`, `book`, `bag`, `camel`, `rhino` and `swan`.

Replace path/to/image/folder with the path to your image folder and path/to/pretrained/model.pth with the path to your pre-trained FlowNet model.  You can select either the direct or sequential tracking method and specify the sequence for inference.

### Inference for All Sequences

A script was created to iteratively run inference for all sequences, automating the process instead of running each sequence manually.  

The script processes all sequences defined in the dataset and applies the selected tracking mode. To execute it, use the following command:  

```bash
python3 all_sequences_inference.py --pretrained /path/to/pretrained/model.pth --mode direct/sequential --sequence train/test
```

- `--pretrained`: Path to the pretrained FlowNet model.  
- `--mode`: Tracking mode (`direct` or `sequential`).  
- `--sequence`: Specifies whether to process sequences from the training (`train`) or test (`test`) set.  

This will compute optical flow and propagate masks iteratively for each sequence in the dataset, storing the results automatically.
## Self-Supervised Fine-tuning Strategy
To adapt the FlowNet model to the specific tracking task, fine-tuning was performed by re-training the final layer on the dataset. This helps the model better capture motion patterns relevant to the sequences used in inference.

In order to fine-tune the FlowNet model, run the following command:
```
python3 fine_tune.py --pretrained path/to/pretrained/model.pth
```
Replace path/to/pretrained/model.pth with the actual path to your pre-trained FlowNet model. For example : flownets_EPE1.951.pth

## Results Visualization and Inference Notebook

A **Jupyter Notebook** is provided to run inference and analyze the results. It allows you to:
* Run inference on selected sequences using the pretrained FlowNet model.
* Visualize predicted masks, comparing them between direct and sequential tracking methods.
* Generate GIFs to observe mask propagation over time.
* Compute and display evaluation metrics for tracking performance.

You can access the notebook here: [Results Analysis Notebook](FlowNet_Trackingipynb)  


## Acknowledgements  

This project was developed as part of the Computer Vision course led and supervised by P.-H. Conze, V. Burdin, R. Fablet, P. Papadakis, L. Bergantin and G. Andrade-Miranda at IMT Atlantique - Bretagne-Pays de la Loire. 

## Authors
Catalina ARDILA, email: <dely.ardila-medina@imt-atlantique.net>

Maria FLORENZA, email: <maria.florenza-lamberti@imt-atlantique.net>

Nhan NGUYEN, email: <nhan.nguyen@imt-atlantique.net>


