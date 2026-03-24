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

    #TODO: Add your code here. Do not modify the return and input arguments.
    img1 = imgs["t1_1.png"].float()
    img2 = imgs["t1_1.png"].float()

    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)

    img1 = K.color.rgb_to_grayscale(img1)
    img2 = K.color.rgb_to_grayscale(img2)

    _, _, width1, height1 = img1.shape
    _, _, width2, height2 = img2.shape
    
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

    nmsImg1 = nmsImg1.squeeze()
    nmsImg2 = nmsImg2.squeeze()

    kpy1, kpx1 = torch.where(nmsImg1 > 0) 
    kpy2, kpx2 = torch.where(nmsImg2 > 0) 

    keypoints1 = torch.stack([kpx1,kpy1],dim=1)
    keypoints2 = torch.stack([kpx2,kpy2],dim=1)

    kpAndDescriptor1 = []

    for kp in keypoints1:
        x = kp[1]
        y = kp[0]
        if y >= 4 and y < height1 - 4 and x >= 4 and x < width1 - 4:
            d = img1[y-4:y+5,x-4:x+5]
            d = d - d.mean()   #normalizing
            d = d / d.std()     #normalizing
            kpAndDescriptor1.append((x,y,d.reshape(-1)))

    kpAndDescriptor2 = []

    for kp in keypoints2:
        x = kp[1]
        y = kp[0]
        if y >= 4 and y < height2 - 4 and x >= 4 and x < width2 - 4:
            d = img1[y-4:y+5,x-4:x+5]
            d = d - d.mean()   #normalizing
            d = d / d.std()     #normalizing
            kpAndDescriptor2.append((x,y,d.reshape(-1)))

    matches = []

    for kpd1 in kpAndDescriptor1:
        d1 = kpd1[2]
        bestDistance = (float('inf'),None,None)
        secBestDistance = (float('inf'),None,None)
        for kpd2 in kpAndDescriptor2:
            d2 = kpd2[2]
            dist = (torch.norm(d1-d2),kpd2[0],kpd2[1])
            if bestDistance[0] > dist[0]:
               secBestDistance = bestDistance
               bestDistance = dist
            elif secBestDistance[0] > dist[0]:
                secBestDistance = dist
        if bestDistance[0]/secBestDistance[0] < .75:
            matches.append((kpd1[0],kpd1[1],bestDistance[0],bestDistance[1]))


    #x1 y1 1 0 0 0 -x1*x2 -y1*x2
    #0 0 0 x1 y1 1 -x1*y2 -y1*y2
    A = torch.zeros((8, 8), dtype=torch.float32)
    bestHomography = (torch.zeros(3, 3, dtype=torch.float32),float('-inf'))
    for _ in range(1000):
        homography = torch.zeros(3, 3, dtype=torch.float32)
        b = torch.zeros((8,), dtype=torch.float32)

        randomSampleInd = torch.randperm(len(matches))[:4]
        print("This is my random sample!: " + str(randomSampleInd))
        randomSample = [matches[j] for j in randomSampleInd]
        col = 0
        print("This is my random sample!: " + str(randomSample))
        for match in randomSample:
            x1, y1, x2, y2 = match
            A[col] = torch.tensor(x1,y1,1,0,0,0,-x1*x2,-y1*x2)
            A[col+1] = torch.tensor(0,0,0,x1,y1,1,-x1*y2,-y1*y2)
            b[col] = x2
            b[col+1] = y2
            col+=2

        h = torch.linalg.solve(A, b)

        homography[0] = h[0:3]
        homography[1] = h[3:7]
        homography[2] = h[7:9]
        homography[2,2] = 1.0

        inlierCount = 0
        for match in matches:
            x1, y1, x2, y2 = match
            a = torch.tensor([x1,y1,1.0],dtype=torch.float32)
            b = homography @ a
            b = b / b[2]
            b2 = torch.tensor([x2,y2],dtype=torch.float32)
            dist = torch.norm(b[:2]-b2)

            if dist < 5.0:
                inlierCount+=1
            
        if bestHomography[1] < inlierCount:
            bestHomography = (homography,inlierCount) 

    #Find new size of image
    imgCorners1 = torch.tensor([[0,0,1],[width1,0,1],[0,height1,1],[width1,height1,1]],dtype=torch.float32) #ones for homogenous coords
    warpedCorners = (bestHomography @ imgCorners1.T).T
    warpedCorners = warpedCorners / warpedCorners[:,2].unsqueeze(1)

    img2Corners = torch.tensor([[0,0,1],[width2,0,1],[0,height2,1],[width2,height2,1]],dtype=torch.float32)

    allCorners = torch.cat([warpedCorners,img2Corners])
    minx = min(allCorners[:,0])
    maxx =  max(allCorners[:,0])
    miny = min(allCorners[:,1])
    maxy =  max(allCorners[:,1])

    translationMatrix = torch.tensor([[1,0,-minx],[0,1,-miny],[0,0,1]],dtype=torch.float32)

    imgW = int(torch.ceil(maxx - minx).item())
    imgH = int(torch.ceil(maxy - miny).item())

    warpedImg1 = K.geometry.transform.warp_perspective(imgs["t1_1.png"].float(), translationMatrix @ bestHomography, (imgW,imgH),mode='bilinear',padding_mode=torch.zeros)
    
    retImg = torch.zeros((3, imgW,imgH)) # assumed 256*256 resolution. Update this as per your logic.
    retImg = warpedImg1.clone()
    retImg[:,int(-miny):int(-miny)+height2,int(-minx):int(-minx)+width2]

    return retImg

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
