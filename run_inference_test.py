import argparse
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import models
from tqdm import tqdm
import torchvision.transforms as transforms
import flow_transforms
from imageio.v2 import imread, imwrite
import numpy as np
from util import flow2rgb
import cv2
from skimage.measure import regionprops
from skimage.segmentation import mark_boundaries
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp2d
import os

def dice_assessment(groundtruth, estimated, label=255):
    A = groundtruth == label
    B = estimated == label
    TP = len(np.nonzero(A*B)[0])
    FN = len(np.nonzero(A*(~B))[0])
    FP = len(np.nonzero((~A)*B)[0])
    DICE = 0
    if (FP+2*TP+FN) != 0:
        DICE = float(2)*TP/(FP+2*TP+FN)
    return DICE*100

def seg2bmap(seg,width=None,height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.

    Arguments:
        seg     : Segments labeled from 1..k.
        width   : Width of desired bmap  <= seg.shape[1]
        height  : Height of desired bmap <= seg.shape[0]

    Returns:
        bmap (ndarray):	Binary boundary map.
    """

    seg = seg.astype(bool)
    seg[seg>0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width  = seg.shape[1] if width  is None else width
    height = seg.shape[0] if height is None else height

    h,w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (width>w | height>h | abs(ar1-ar2)>0.01),\
        'Can''t convert %dx%d seg to %dx%d bmap.'%(w,h,width,height)

    e  = np.zeros_like(seg)
    s  = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:,:-1]    = seg[:,1:]
    s[:-1,:]    = seg[1:,:]
    se[:-1,:-1] = seg[1:,1:]

    b        = seg^e | seg^s | seg^se
    b[-1,:]  = seg[-1,:]^e[-1,:]
    b[:,-1]  = seg[:,-1]^s[:,-1]
    b[-1,-1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height,width))
        for x in range(w):
            for y in range(h):
                if b[y,x]:
                    j = 1+np.floor((y-1)+height / h)
                    i = 1+np.floor((x-1)+width  / h)
                    bmap[j,i] = 1

    return bmap

def centroid_assessment(groundtruth,estimated):
    a = regionprops(groundtruth)
    b = regionprops(estimated)
    return np.linalg.norm(np.array(a[0].centroid)-np.array(b[0].centroid))

def db_eval_boundary(foreground_mask,gt_mask,bound_th=0.008):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.

    Arguments:
        foreground_mask (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.

    Returns:
        F (float): boundaries F-measure
    """
    assert np.atleast_3d(foreground_mask).shape[2] == 1

    bound_pix = bound_th if bound_th >= 1 else \
            np.ceil(bound_th*np.linalg.norm(foreground_mask.shape))

    # Get the pixel boundaries of both masks
    fg_boundary = seg2bmap(foreground_mask);
    gt_boundary = seg2bmap(gt_mask);

    from skimage.morphology import binary_dilation,disk

    fg_dil = binary_dilation(fg_boundary,disk(bound_pix))
    gt_dil = binary_dilation(gt_boundary,disk(bound_pix))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg     = np.sum(fg_boundary)
    n_gt     = np.sum(gt_boundary)

    #% Compute precision and recall
    if n_fg == 0 and  n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0  and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match)/float(n_fg)
        recall    = np.sum(gt_match)/float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2*precision*recall/(precision+recall);

    return F*100.

def concatenation(unary_flow, to_ref_flow):
    x0 = np.arange(unary_flow.shape[1]) 
    y0 = np.arange(unary_flow.shape[0])

    # To interpolate in 2D, we use RegularGridInterpolator
    fx = RegularGridInterpolator((y0, x0), to_ref_flow[:,:,1], method='linear', bounds_error=False, fill_value=0)
    fy = RegularGridInterpolator((y0, x0), to_ref_flow[:,:,0], method='linear', bounds_error=False, fill_value=0)

    xx, yy = np.meshgrid(x0, y0)

    z_x = fx((yy, xx))
    z_y = fy((yy, xx))

    result = np.stack([z_y, z_x], axis=-1)
    return result

def set_cuda():
    # setting the cuda device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    return device

def load_model(model_path, cuda_device):
    # Importing the model
    network_data = torch.load(model_path, map_location=cuda_device)
    print("=> using pre-trained model '{}'".format(network_data['arch']))
    model = models.__dict__[network_data['arch']](network_data).to(cuda_device)
    model.eval()
    cudnn.benchmark = True

    if 'div_flow' in network_data.keys():
        div_flow = network_data['div_flow']
    else:
        div_flow = 20.0

    return model, div_flow

@torch.no_grad()
def inference(name_sequence, cuda_device, model, div_flow, path_sequence):
    
    sequence_files = [f for f in os.listdir(path_sequence) if f.startswith(name_sequence) and f.endswith(".bmp")]
    nb_start = 1
    nb_end = len(sequence_files)

    flow_dir = Path('results') / 'flow'
    flow_dir.mkdir(parents=True, exist_ok=True)

    input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
        transforms.Normalize(mean=[0.411,0.432,0.45], std=[1,1,1])
    ])

    name = name_sequence + '-'

    for i in tqdm(range(nb_start, nb_end)):
        n_img_1 = i
        n_img_2 = i+1

        path_img1 = str(Path(path_sequence) / (name + str(n_img_1).zfill(3) + ".bmp"))
        path_img2 = str(Path(path_sequence) / (name + str(n_img_2).zfill(3) + ".bmp"))

        img1 = imread(path_img1)
        img2 = imread(path_img2)
        img1 = input_transform(img1)
        img2 = input_transform(img2)

        input_var = torch.cat([img1, img2]).unsqueeze(0)
        input_var = input_var.to(cuda_device)
        output = model(input_var)
        output = F.interpolate(output, size=img1.size()[-2:], mode = "bilinear", align_corners=False)
        
        flow_output = output.squeeze(0)

        file_name = str(flow_dir / (name + "flow-" + str(n_img_1).zfill(3) + "-" + str(n_img_2).zfill(3)))
        rgb_flow = flow2rgb(div_flow*flow_output, max_value=None)
        to_save_rgb = (rgb_flow * 255).astype(np.uint8).transpose(1,2,0)
        imwrite(file_name + '.png', to_save_rgb)

        to_save_np = (div_flow*flow_output).cpu().numpy().transpose(1,2,0)
        np.save(file_name + '.npy', to_save_np)

@torch.no_grad()
def simple_inference(img1, img2, name, model, cuda_device, save = False):
    div_flow = 20.0
    flow_dir = Path('sequences-train') / 'flow-sequential'
    flow_dir.mkdir(parents=True, exist_ok=True)

    input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
        transforms.Normalize(mean=[0.411,0.432,0.45], std=[1,1,1])
    ])

    img1 = input_transform(img1)
    img2 = input_transform(img2)

    input_var = torch.cat([img1, img2]).unsqueeze(0)
    input_var = input_var.to(cuda_device)
    output = model(input_var)
    output = F.interpolate(output, size=img1.size()[-2:], mode="bilinear", align_corners=False)
    flow_output = output.squeeze(0)

    # create the flow image for the function
    name = name
    file_name = str(flow_dir / name)
    rgb_flow = flow2rgb(div_flow * flow_output, max_value=None)
    to_save_rgb = (rgb_flow * 255).astype(np.uint8).transpose(1, 2, 0)
    
    # Detach the tensor before converting to numpy
    to_save_np = (div_flow * flow_output).detach().cpu().numpy().transpose(1, 2, 0)

    if save:
        # Save flow image and numpy array
        imwrite(file_name + '.png', to_save_rgb)
        np.save(file_name + '.npy', to_save_np)
        #print(f"Saved flow to {file_name}")
        

    return  to_save_np
    
@torch.no_grad()
def complete_inferece_saving_seq(name_sequence, path_sequence, model, div_flow):
    print("Start of flow calculation for sequential integration")
    
    sequence_files = [f for f in os.listdir(path_sequence) if f.startswith(name_sequence) and f.endswith(".bmp")]
    nb_start = 1
    nb_end = len(sequence_files)

    device = set_cuda()
    # model, div_flow = load_model(path_to_model, device)

    mask = imread(str(Path(path_sequence) / (name_sequence + "-001.png")))

    dice_seq, fmeasures_seq, centroid_assessment_seq = [], [], []
    for i in tqdm(range(nb_start, nb_end)):

        n_img_1 = i
        n_img_2 = i+1

        path_img1 = str(Path(path_sequence) / (name_sequence + "-" + str(n_img_1).zfill(3) + ".bmp"))
        path_img2 = str(Path(path_sequence) / (name_sequence + "-" + str(n_img_2).zfill(3) + ".bmp"))

        mask_cur = imread(str(Path(path_sequence) / (name_sequence + "-" + str(n_img_2).zfill(3) +".png")))

        img1 = imread(path_img1)
        img2 = imread(path_img2)
        black_image = np.zeros((img1.shape[0], img1.shape[1], 3), dtype=np.uint8)

        # flow = simple_inference(img1, img2, name_sequence + str(n_img_1).zfill(3), model, device) # This is line 355, where `flow` is assigned

        # Ensure simple_inference returns the numpy array
        flow = simple_inference(img1, img2, name_sequence + str(n_img_1).zfill(3), model, device, save=True)

        if i == 1:
            flow_conc = flow
        else:
            flow_conc = concatenation(flow, flow_conc)

        mask_predict = propagate_mask(flow_conc, img_current= img2, mask_begin = mask_cur)
        boundaries_predict =  mark_boundaries(black_image, mask_predict, color=(1, 0, 0))
        boundaries_gd      =  mark_boundaries(black_image, mask_cur, color=(0, 1, 0)) 

        # imwrite( "./results/"name_sequence +"-mask_pro_sequential"+ str(n_img_2).zfill(3) +'.png', mask_predict)
        results_seq = f"./results/{name_sequence}-mask/sequential"
        os.makedirs(results_seq, exist_ok=True)

        # Guardar la máscara
        mask_filename = f"{results_seq}/{name_sequence}-{str(n_img_2).zfill(3)}.png"
        imwrite( mask_filename, mask_predict)

        dice_seq.append(dice_assessment(mask, mask_predict))
        fmeasures_seq.append(db_eval_boundary(mask,mask_predict))
        centroid_assessment_seq.append(centroid_assessment(mask,mask_predict))

        np.save("./results/" + name_sequence + "-dice_seq.npy", dice_seq)
        np.save("./results/" + name_sequence + "-fmeasures_seq.npy", fmeasures_seq)
        np.save("./results/" + name_sequence + "-centroid_assessment_seq.npy", centroid_assessment_seq)

    # print("ok " + name_sequence)        

@torch.no_grad()
# Runs inference with direct integration, first frame as reference
def inference_direct(name_sequence, cuda_device, model, div_flow, path_sequence):
    print("Start of flow calculation for direct integration")
    flow_dir = Path('sequences-train') / 'flow-direct'
    flow_dir.mkdir(parents=True, exist_ok=True)

    input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
        transforms.Normalize(mean=[0.411,0.432,0.45], std=[1,1,1])
    ])

    sequence_files = [f for f in os.listdir(path_sequence) if f.startswith(name_sequence) and f.endswith(".bmp")]
    nb_start = 1
    nb_end = len(sequence_files)
    # print(f"nb_start: {nb_start}, nb_end: {nb_end}")

    name = name_sequence + '-'
    
    n_img_1 = 1
    path_img1 = str(Path(path_sequence) / (name + str(n_img_1).zfill(3) + ".bmp"))
    img1 = imread(path_img1)
    img1 = input_transform(img1)

    for i in tqdm(range(nb_start, nb_end)):
        n_img_2 = i+1
        path_img2 = str(Path(path_sequence) / (name + str(n_img_2).zfill(3) + ".bmp"))
        img2 = imread(path_img2)
        img2 = input_transform(img2)

        input_var = torch.cat([img1, img2]).unsqueeze(0)
        input_var = input_var.to(cuda_device)
        output = model(input_var)
        output = F.interpolate(output, size=img1.size()[-2:], mode = "bilinear", align_corners=False)
        
        flow_output = output.squeeze(0)

        file_name = str(flow_dir / (name + "flow-" + str(n_img_1).zfill(3) + "-" + str(n_img_2).zfill(3)))
        rgb_flow = flow2rgb(div_flow*flow_output, max_value=None)
        to_save_rgb = (rgb_flow * 255).astype(np.uint8).transpose(1,2,0)
        imwrite(file_name + '.png', to_save_rgb)

        to_save_np = (div_flow*flow_output).cpu().numpy().transpose(1,2,0)
        # print(file_name + '.npy')
        np.save(file_name + '.npy', to_save_np)

    print("End of flow calculation for direct integration")

def propagate_mask(flow, img_current, mask_begin):
    new_mask = np.zeros(shape=img_current.shape[:2], dtype=np.uint8)
    for x in range(img_current.shape[0]):
        for y in range(img_current.shape[1]):
            x_, y_ = np.rint(x+flow[x,y,1]).astype(int), np.rint(y+flow[x,y,0]).astype(int)
            if (x_>=0) and (x_<img_current.shape[0]) and (y_>=0) and (y_<img_current.shape[1]):
                if mask_begin[x_,y_] > 0:
                    new_mask[x,y] = 255
    return new_mask

@torch.no_grad()
def propagate_mask_direct(name_sequence, path_sequence):
    print("Start of mask propagation for direct integration")
    
    sequence_files = [f for f in os.listdir(path_sequence) if f.startswith(name_sequence) and f.endswith(".bmp")]
    nb_start = 1
    nb_end = len(sequence_files)
    
    flow_dir = Path('sequences-train') / 'flow-direct'
    # Ensure the directory exists
    flow_dir.mkdir(parents=True, exist_ok=True)
    original_mask = imread(str(Path(path_sequence) / (name_sequence + "-001.png")))
    first_img = imread(str(Path(path_sequence) / (name_sequence + "-001.bmp")))
    
    dice_dir, fmeasures_dir, centroid_assessment_dir = [], [], []

    for i in tqdm(range(nb_start+1, nb_end+1)):
        mask_gt = imread(str(Path(path_sequence) / (name_sequence + "-" + str(i).zfill(3) +".png"))) # for evaluation
        flow = np.load( str(flow_dir / (name_sequence + "-flow" + "-001-" + str(i).zfill(3) + '.npy')))    
        current_mask = propagate_mask(flow, img_current= first_img, mask_begin = original_mask)
        
        # imwrite( f"./results/{name_sequence}-mask/direct/{name_sequence}-001-"+ str(i).zfill(3) +'.png', current_mask)
        results_dir = f"./results/{name_sequence}-mask/direct"
        os.makedirs(results_dir, exist_ok=True)
        mask_filename = f"{results_dir}/{name_sequence}-{str(i).zfill(3)}.png"
        imwrite( mask_filename, current_mask)
        # print(f"ok {name_sequence}-mask_pro_dir-001-"+ str(i).zfill(3) +'.png')

        dice_dir.append(dice_assessment(mask_gt, current_mask))
        fmeasures_dir.append(db_eval_boundary(mask_gt,current_mask))
        centroid_assessment_dir.append(centroid_assessment(mask_gt,current_mask))

        np.save("./results/" + name_sequence + "-dice_dir.npy", dice_dir)
        np.save("./results/" + name_sequence + "-fmeasures_dir.npy", fmeasures_dir)
        np.save("./results/" + name_sequence + "-centroid_assessment_dir.npy", centroid_assessment_dir)

    print("End of mask propagation for direct integration")

def main():
    parser = argparse.ArgumentParser(description="PyTorch FlowNet inference")
    parser.add_argument('--sequences_path', metavar='DIR', help='path to images folder')
    parser.add_argument('--model_path', metavar='PTH', help='path to pre-trained model')
    parser.add_argument('--mode', type=str, default='sequential', choices=['sequential', 'direct', 'inference'],
                        help='Choose mode: complete_inferece_saving_seq, inference_direct, or inference')
    parser.add_argument('--sequence', '-s', metavar='PREFIX', default='bear', help='Prefix for image sequences')
    # parser.add_argument('--start', type=int, default=1, help='Start frame number')
    # parser.add_argument('--end', type=int, default=26, help='End frame number')

    global path_to_model  # Declare path_to_model as a global variable

    args = parser.parse_args()

    path_to_model = args.model_path  # Assign the argument value to path_to_model

    device = set_cuda()
    
    if args.mode == 'sequential':
        model, div_flow = load_model(path_to_model, device)
        complete_inferece_saving_seq(args.sequence, args.sequences_path, model, div_flow)
        inference(args.sequence, device, model, div_flow, args.sequences_path)
    elif args.mode == 'direct':
        model, div_flow = load_model(path_to_model, device)
        inference_direct(args.sequence, device, model, div_flow, args.sequences_path)
        propagate_mask_direct(args.sequence, args.sequences_path)
    elif args.mode == 'inference':
        model, div_flow = load_model(path_to_model, device)
        inference(args.sequence, device, model, div_flow, args.sequences_path)

if __name__ == "__main__":
    main()
