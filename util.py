import os
import numpy as np
import shutil
import torch
from skimage.segmentation import mark_boundaries
from skimage import io
import matplotlib.pyplot as plt
import imageio



# MF
def create_img_pairs(folder_path, sequence_name):
    # print("Making image pairs")
    img_pairs = []
    images = sorted(os.listdir(folder_path))
    images = [img for img in images if img.startswith(sequence_name) and img.endswith(".bmp")]
    # .bmp
    # images = [img for img in images if img.endswith(".bmp")]
    for i in range(len(images) - 1):
         img_pairs.append([
            os.path.join(folder_path, images[i]),
            os.path.join(folder_path, images[i + 1])
        ])
        # img_pairs.append([images[i], images[i + 1]])
    return img_pairs

def visualize_boundaries(sequence,mode, image_folder, img_ext='bmp', mask_ext='png'):
    """
    Visualize the predicted mask and the ground truth mask with boundaries
    """
    mask_folder = f'results/{sequence}-mask/{mode}'

    img_files = sorted([f for f in os.listdir(image_folder) if f.startswith(sequence) and f.endswith(img_ext)])
    mask_gt_files = sorted([f for f in os.listdir(image_folder) if f.startswith(sequence) and f.endswith(mask_ext)])
    mask_files = sorted([f for f in os.listdir(mask_folder) if f.startswith(sequence) and f.endswith(mask_ext)])
    num_images = len(img_files)
    # cols = 6
    # rows = math.ceil(num_images / cols)
    plt.figure(figsize=(15, 10))

    if num_images < 30:
        subplot_step = 5
    elif num_images < 60:
        subplot_step = 10
    elif num_images < 90:
        subplot_step = 15
    else:
        subplot_step = 20
    subplot_list = [1,subplot_step,subplot_step*2,subplot_step*3,subplot_step*4, num_images-1]


    for i, (img_file, mask_file, mask_gt_file) in enumerate(zip(img_files, mask_files, mask_gt_files)):
        img_path = os.path.join(image_folder, img_file)
        mask_path = os.path.join(mask_folder, mask_file)

        mask_gt_path = os.path.join(image_folder, mask_gt_file)
        
        img = io.imread(img_path)
        mask = io.imread(mask_path)
        mask_gt = io.imread(mask_gt_path)
        
        img_with_boundaries = mark_boundaries(img, mask, color=(1, 0, 0))
        img_with_gt_boundaries = mark_boundaries(img_with_boundaries, mask_gt, color=(0, 1, 0))

        if i in subplot_list:
            # plt.subplot(rows, cols, i + 1)
            plt.subplot(1,5,i//subplot_step+1)
            plt.imshow(img_with_gt_boundaries)
            plt.title(f'{sequence} - Frame {i + 1}')
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def create_gif(sequence, mode, model, image_folder, img_ext='bmp', mask_ext='png'):
    """ 
    Create a gif with the original image, the predicted mask and the ground truth mask
    """
    if 'finetuned' in model : 
        mask_folder = f'results/{sequence}-mask/{mode}-{model}'
    else:
        mask_folder = f'results/{sequence}-mask/{mode}'

    img_files = sorted([f for f in os.listdir(image_folder) if f.startswith(sequence) and f.endswith(img_ext)])
    mask_gt_files = sorted([f for f in os.listdir(image_folder) if f.startswith(sequence) and f.endswith(mask_ext)])
    mask_files = sorted([f for f in os.listdir(mask_folder) if f.startswith(sequence) and f.endswith(mask_ext)])
    # num_images = len(img_files)

    images = [] 
    
    for i, (img_file, mask_file, mask_gt_file) in enumerate(zip(img_files, mask_files, mask_gt_files)):
        img_path = os.path.join(image_folder, img_file)
        mask_path = os.path.join(mask_folder, mask_file)
        mask_gt_path = os.path.join(image_folder, mask_gt_file)
        
        img = io.imread(img_path)
        mask = io.imread(mask_path)
        mask_gt = io.imread(mask_gt_path)
        
        img_with_boundaries = mark_boundaries(img, mask, color=(1, 0, 0))
        img_with_gt_boundaries = mark_boundaries(img_with_boundaries, mask_gt, color=(0, 1, 0))

        images.append((img_with_gt_boundaries * 255).astype('uint8'))

    # Save
    gif_path = 'results/gif'
    os.makedirs(gif_path, exist_ok=True)
    if 'finetuned' in model : 
        gif_path = f'results/gif/{sequence}-{mode}-{model}.gif'
    else:
        gif_path = f'results/gif/{sequence}-{mode}.gif'

        
    imageio.mimsave(gif_path, images, fps=5)
    print(f'GIF saved in: {gif_path}')


def save_checkpoint(state, is_best, save_path, filename="checkpoint.pth.tar"):
    torch.save(state, os.path.join(save_path, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(save_path, filename),
            os.path.join(save_path, "model_best.pth.tar"),
        )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return "{:.3f} ({:.3f})".format(self.val, self.avg)


def flow2rgb(flow_map, max_value):
    flow_map_np = flow_map.detach().cpu().numpy()
    _, h, w = flow_map_np.shape
    flow_map_np[:, (flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = float("nan")
    rgb_map = np.ones((3, h, w)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map_np / max_value
    else:
        normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    rgb_map[0] += normalized_flow_map[0]
    rgb_map[1] -= 0.5 * (normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[2] += normalized_flow_map[1]
    return rgb_map.clip(0, 1)
