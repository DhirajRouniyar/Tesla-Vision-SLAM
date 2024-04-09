## Usage
1. Copy images and replace folder paths in all files.
2. Run Detic python notebook for json file of all objects.
3. Run inference_images.py in Lane to get json file of all Lanes.
4. Run Wrapper_Depth to update depth in the JSON files
5. Update paths for json files in blender code and run to simulate.

For Blender
1. Open Blender file 'Objects_spawn.blender' given in the folder
2. In the 'Fresh_all_detic_S5.py' change the json file location in line 105,106,107 to load Object, Lane and Angles (Vehicle Orientation)
3. Create folder Blender_Ouput and change render path in the line 10 of 'Objects_spaw.py'



## Packages used:
1. YOLO V8 model pretrained with COCO dataset. This model is used to detect cars, pedestrians, stop signs and traffic lights. Git: https://github.com/ultralytics/ultralytics

2. ZoeDepth - This pretrained model is used to extract the depth from monocular camera. Git: https://github.com/isl-org/ZoeDepth

3. Mask-RCNN - This is used to detect the bounding boxes for lane markers. Each lane marker has a bounding box. We use these bounding boxes and apply classical methods to determine the color of the detected lane. With the obtained bounding box of the lane marker, we extract the contour of the lane. We then dilate the obtained contour and subtract the original contour from it to identify the color of the region(road) surrounding the lane and check if the lane color has more saturation than the surrounding region. If the saturation is higher, the lane is yellow or else it is white.
 This model also detects the presence of a road sign. We use classical methods to identify whether the detected sign is an arrow (Up, Down, Left, Right) where we identify the contour of the sign present and compare it with previously saved contours. Source: https://debuggercafe.com/lane-detection-using-mask-rcnn/


4. YOLO3D - This pretrained model is used to detect the pose of the car. We assume the roll and pitch to be fixed and only collect the yaw information from this model. Git: https://github.com/ruhyadi/YOLO3D


5. I2L-MeshNet: This pretrained model is used to determine the pedestrian pose. Official implementation can be found in the following link: https://github.com/mks0601/I2L-MeshNet_RELEASE?tab=readme-ov-file

6. Detic: Facebook Research: https://github.com/facebookresearch/Detic


## Results:
![](https://github.com/DhirajRouniyar/Assets/blob/main/Videos/OutputVisualizationVideoSeq2.mp4)

