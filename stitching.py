'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function stitch_background() and panorama().
3. If you want to show an image for debugging, please use show_image() function in util.py. 
4. Please do NOT save any intermediate files in your final submission.
'''
import torch
import kornia as K
from typing import Dict
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

# ------------------------------------ Task 1 ------------------------------------ #
def stitch_background(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: input images are a dict of 2 images of torch.Tensor represent an input images for task-1.
    Returns:
        img: stitched_image: torch.Tensor of the output image.
    """
    img = torch.zeros((3, 256, 256)) # assumed 256*256 resolution. Update this as per your logic.

    #TODO: Add your code here. Do not modify the return and input arguments.
    img1 = imgs["t1_1.png"].float()
    img2 = imgs["t1_1.png"].float()

    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)

    img1 = K.color.rgb_to_grayscale(img1)
    img2 = K.color.rgb_to_grayscale(img2)

    img1Map = K.feature.harris_response(img1, k=0.04, grads_mode='sobel', sigmas=None)
    img2Map = K.feature.harris_response(img2, k=0.04, grads_mode='sobel', sigmas=None)

    maxMap1 = torch.max_pool2d(img1Map,3,1,1)
    maxMap2 = torch.max_pool2d(img2Map,3,1,1)

    mask1 = img1Map == maxMap1
    mask2 = img2Map == maxMap2

    nmsImg1 = img1Map * mask1
    nmsImg2 = img2Map * mask2

    threshold1 = 0.05 * nmsImg1.max()
    threshold2 = 0.05 * nmsImg2.max()


    nmsImg1[nmsImg1 < threshold1] = 0
    nmsImg2[nmsImg2 < threshold2] = 0

    nmsImg1 = nmsImg1.squeze()
    nmsImg2 = nmsImg2.squeze()

    

    return img

# ------------------------------------ Task 2 ------------------------------------ #
def panorama(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: dict {filename: CxHxW tensor} for task-2.
    Returns:
        img: panorama, 
        overlap: torch.Tensor of the output image. 
    """
    img = torch.zeros((3, 256, 256)) # assumed 256*256 resolution. Update this as per your logic.
    overlap = torch.empty((3, 256, 256)) # assumed empty 256*256 overlap. Update this as per your logic.

    #TODO: Add your code here. Do not modify the return and input arguments.

    return img, overlap
