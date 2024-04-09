import torch
import torchvision
import cv2
import argparse
import numpy as np
import torch.nn as nn
import glob
import os
import json

from PIL import Image
from infer_utils import draw_segmentation_map, get_outputs
from torchvision.transforms import transforms as transforms
from class_names import INSTANCE_CATEGORY_NAMES as class_names

def get_mean_hsv(image):

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Mask pixels where at least one of the HSV components is non-zero
    mask = cv2.inRange(hsv_image, np.array([1, 1, 1]), np.array([255, 255, 255]))
    
    # Calculate mean of non-zero pixels for each channel
    mean_hue = np.mean(hsv_image[:, :, 0][mask > 0])
    mean_saturation = np.mean(hsv_image[:, :, 1][mask > 0])
    mean_value = np.mean(hsv_image[:, :, 2][mask > 0])
    
    return mean_hue, mean_saturation, mean_value

def Get_Lane_Line_Colors(masks, boxes, labels, orig_image):
    
    Lane_Box_Colors = []
    Im_dummy = orig_image.copy()
    
    Im_dummy = np.array(Im_dummy)
    Im_dummy = cv2.cvtColor(Im_dummy, cv2.COLOR_RGB2BGR)
    
    for box in boxes:
        
        # Set default lane color to white
        Lane_color = 'white'
        
        # Get the lane image and mask
        [x1, y1], [x2, y2] = box
        Lane_Image = Im_dummy[y1:y2, x1:x2]
        lane_mask = masks[boxes.index(box)]
        lane_mask = lane_mask[y1:y2, x1:x2]
        cv2.imwrite('./Lane_Image.jpg', Lane_Image)
        Lane_Image_Gray = cv2.cvtColor(Lane_Image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('./Lane_Image_Gray.jpg', Lane_Image_Gray)
        mask = np.zeros(Lane_Image_Gray.shape, dtype=np.uint8)
        # Contour = cv2.findContours(Lane_Image_Gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(mask, Contour[0], -1, (255, 255, 255), thickness=cv2.FILLED)
        mask[lane_mask] = 255
        cv2.imwrite('./mask.jpg', mask)
        
        # Dilate the mask to get the difference mask
        kernel = np.ones((15,15),np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=1)
        mask_diff = cv2.subtract(dilated, mask)
        mask_diff_Image = cv2.bitwise_and(Lane_Image, Lane_Image, mask=mask_diff)
        cv2.imwrite('./mask_diff_Image.jpg', mask_diff_Image)
        masked_Image = cv2.bitwise_and(Lane_Image, Lane_Image, mask=mask)
        cv2.imwrite('./masked_Image.jpg', masked_Image)

        # Check if the mask has more saturation than the difference mask. More saturation indicates yellow lane.
        if cv2.countNonZero(mask_diff) > 0:
            diff_hue, diff_saturation, diff_value = get_mean_hsv(mask_diff_Image)
            mask_hue, mask_saturation, mask_value = get_mean_hsv(masked_Image)
            if mask_saturation - diff_saturation > 2:
                Lane_color = 'yellow'
        Lane_Box_Colors.append(Lane_color)
    
    # Return the list of colors for each lane
    return Lane_Box_Colors

def Check_Arrow_Direction(Sign_mask, box, orig_image):
    
    Im_dummy = orig_image.copy()
    Im_dummy = np.array(Im_dummy)
    Im_dummy = cv2.cvtColor(Im_dummy, cv2.COLOR_RGB2BGR)

    [x1, y1], [x2, y2] = box
    Sign_Image = Im_dummy[y1:y2, x1:x2]
    Sign_mask = Sign_mask[y1:y2, x1:x2]
    cv2.imwrite('./Sign_Image.jpg', Sign_Image)
    Sign_Image_Gray = cv2.cvtColor(Sign_Image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('./Sign_Image_Gray.jpg', Sign_Image_Gray)
    mask = np.zeros(Sign_Image_Gray.shape, dtype=np.uint8)
    # Contour = cv2.findContours(Lane_Image_Gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(mask, Contour[0], -1, (255, 255, 255), thickness=cv2.FILLED)
    mask[Sign_mask] = 255
    kernel = np.ones((5, 5), np.uint8) 
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    cv2.imwrite('./mask.jpg', mask)
    
    # Find the contours of the mask
    Contour, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    Arrow_RIGHT = cv2.cvtColor(cv2.imread('Init_data/Arrow_RIGHT.png'), cv2.COLOR_BGR2GRAY)
    Arrow_UP = cv2.cvtColor(cv2.imread('Init_data/Arrow_UP.png'), cv2.COLOR_BGR2GRAY)
    Arrow_LEFT = cv2.cvtColor(cv2.imread('Init_data/Arrow_LEFT.png'), cv2.COLOR_BGR2GRAY)
    Arrow_DOWN = cv2.cvtColor(cv2.imread('Init_data/Arrow_DOWN.png'), cv2.COLOR_BGR2GRAY)

    Up_Contour,_ = cv2.findContours(Arrow_UP, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    Right_Contour,_ = cv2.findContours(Arrow_RIGHT, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    Left_Contour,_ = cv2.findContours(Arrow_LEFT, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    Down_Contour,_ = cv2.findContours(Arrow_DOWN, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(Contour) != 0: 
        Right_Dist = cv2.matchShapes(Contour[0], Right_Contour[0], 1, 0.0)
        Up_Dist = cv2.matchShapes(Contour[0], Up_Contour[0], 1, 0.0)
        Left_Dist = cv2.matchShapes(Contour[0], Left_Contour[0], 1, 0.0)
        # Down_Dist = cv2.matchShapes(Contour[0], Down_Contour[0], 1, 0.0)
        Dists = [Right_Dist, Up_Dist, Left_Dist]
        Min_Dist = min(Dists)
        Arrow_Direction = 'None'
        if Right_Dist == Min_Dist and Right_Dist < 2.5:
            Arrow_Direction = 'Right'
            print('Arrow_Dist:', Right_Dist)
        elif Up_Dist == Min_Dist and Up_Dist < 2.5:
            Arrow_Direction = 'Up'
            print('Arrow_Dist:', Up_Dist)
            cv2.imwrite('Arrow_Contours.png', mask)
        elif Left_Dist == Min_Dist and Left_Dist < 2.5:
            Arrow_Direction = 'Left'
            print('Arrow_Dist:', Left_Dist)
        # elif Down_Dist == Min_Dist and Down_Dist < 2.5:
        #     Arrow_Direction = 'Down'
        #     print('Arrow_Dist:', Down_Dist)
        print(f'Arrow Direction:', Arrow_Direction)
        return Arrow_Direction
    return 'None'


parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', 
    '--input',
    default='/media/storage/lost+found/CV/P3/P3Data/Sequences/scene4/Undist/Front_Frames',
    # required=True, 
    help='path to the input data'
)
parser.add_argument(
    '-t', 
    '--threshold', 
    default=0.8, 
    type=float,
    help='score threshold for discarding detection'
)
parser.add_argument(
    '-w',
    '--weights',
    # default='out/checkpoint.pth',
    default='outputs/training/road_line/model_15.pth',
    help='path to the trained weight file'
)
parser.add_argument(
    '--show',
    action='store_true',
    help='whether to visualize the results in real-time on screen'
)
parser.add_argument(
    '--no-boxes',
    action='store_true',
    help='do not show bounding boxes, only show segmentation map'
)
args = parser.parse_args()

OUT_DIR = os.path.join('outputs', 'inference_2')
os.makedirs(OUT_DIR, exist_ok=True)

model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
    pretrained=False, num_classes=91
)

model.roi_heads.box_predictor.cls_score = nn.Linear(in_features=1024, out_features=len(class_names), bias=True)
model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features=1024, out_features=len(class_names)*4, bias=True)
model.roi_heads.mask_predictor.mask_fcn_logits = nn.Conv2d(256, len(class_names), kernel_size=(1, 1), stride=(1, 1))

# initialize the model
ckpt = torch.load(args.weights)
model.load_state_dict(ckpt['model'])

# set the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load the modle on to the computation device and set to eval mode
model.to(device).eval()
print(model)

# transform to convert the image to tensor
transform = transforms.Compose([
    transforms.ToTensor()
])

image_paths = sorted(glob.glob(os.path.join(args.input, '*.jpg')))
Final_Dict = {}
for image_path in image_paths:
    print(image_path)
    image = Image.open(image_path)
    # keep a copy of the original image for OpenCV functions and applying masks
    orig_image = image.copy()
    
    # transform the image
    image = transform(image)
    # add a batch dimension
    image = image.unsqueeze(0).to(device)
    
    masks, boxes, labels = get_outputs(image, model, args.threshold)
    
    Lane_Line_Colors = Get_Lane_Line_Colors(masks, boxes, labels, orig_image)
    Int_Dict = {}
    Arrow_Direction = 'None'
    for i in range(len(boxes)):
        if labels[i] == 'road-sign-line':
            Arrow_Direction = Check_Arrow_Direction(masks[i], boxes[i], orig_image)
    
    # Form Dictionary for json
    
    
    for box in boxes:
        Dict = {}
        Dict['box'] = box
        Dict['label'] = labels[boxes.index(box)]
        Dict['color'] = Lane_Line_Colors[boxes.index(box)]
        if labels[boxes.index(box)] == 'road-sign-line':
            Arrow_Direction = Check_Arrow_Direction(masks[i], boxes[i], orig_image)
            Dict['arrow'] = Arrow_Direction
        else:
            Dict['arrow'] = 'None'
        Int_Dict[boxes.index(box)] = Dict
    Final_Dict['frame'+str(image_paths.index(image_path))] = Int_Dict
    result = draw_segmentation_map(orig_image, masks, boxes, labels, args)
    
    # visualize the image
    if args.show:
        cv2.imshow('Segmented image', np.array(result))
        cv2.waitKey(1)
    
    # set the save path
    save_path = f"{OUT_DIR}/{image_path.split(os.path.sep)[-1].split('.')[0]}.jpg"
    cv2.imwrite(save_path, result)

f = open("/media/storage/lost+found/CV/P3/P3Data/Sequences/scene4/Undist/Scene4_Lanes_2D.json","w")
json.dump(Final_Dict, indent=4, fp=f)
f.close()