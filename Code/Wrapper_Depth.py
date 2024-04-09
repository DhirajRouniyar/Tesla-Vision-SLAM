import numpy as np
import json
import torch
import PIL
import cv2
from glob import glob
# from ultralytics import YOLO



Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

repo = "isl-org/ZoeDepth"
# Zoe_N
model_zoe_n = torch.hub.load(repo, "ZoeD_N", pretrained=True)

# model_zoe_n = torch.hub.load(".", "ZoeD_N", source="local", pretrained=True)
zoe = model_zoe_n.to(Device)

# model = YOLO('yolov8n.pt').to('cuda')

# im_path = "/home/dhrumil/Downloads/2.jpeg"
Images = "/media/storage/lost+found/CV/P3/P3Data/Sequences/scene1/Undist/Front_Frames/frame*.jpg"
# video_path = "/media/storage/lost+found/CV/P3/P3Data/Sequences/scene5/Undist/2023-02-14_11-56-56-front_undistort.mp4"
# im = cv2.imread(im_path)
print(Images)

Depths = []
Loaded_Images = []
count = 0
for im_path in sorted(glob(Images)):
    # if count != 142:
    #     count += 1
    #     continue
    print(count)
    im = PIL.Image.open(im_path).convert("RGB")
    # im.save("/media/storage/lost+found/CV/P3/P3Data/Sequences/scene6/Undist/Depth_Frames/frame"+str(count)+".jpg")
    Loaded_Images.append(im)
    depth = zoe.infer_pil(im)
    # Save_Depth = np.array(depth)
    # Max = np.max(Save_Depth)
    # Normalised_Depth = ((Max-Save_Depth)/Max)*255
    # cv2.imwrite("/media/storage/lost+found/CV/P3/P3Data/Sequences/scene6/Undist/Depth_Frames/frame"+str(count)+".jpg", Normalised_Depth)
    Depths.append(depth)
    count += 1


##################### Objects #####################

#Open json files to read and write
f = open("/media/storage/lost+found/CV/P3/P3Data/Sequences/scene1/Undist/Scene1_Objects_Detic_2D_Optical_Flow.json", "r")
w = open("/media/storage/lost+found/CV/P3/P3Data/Sequences/scene1/Undist/Scene1_Objects_Detic_3D_Optical_Flow.json", "w")

# Load Traffic Sign Images and create contours
Traffic_UP = cv2.imread("/media/storage/lost+found/CV/P3/P3Data/Traffic_Sign/Traffic_UP.png")
Traffic_Down = cv2.imread("/media/storage/lost+found/CV/P3/P3Data/Traffic_Sign/Traffic_DOWN.png")
UP_Contour, _ = cv2.findContours(cv2.cvtColor(Traffic_UP, cv2.COLOR_BGR2GRAY), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
Down_Contour, _ = cv2.findContours(cv2.cvtColor(Traffic_Down, cv2.COLOR_BGR2GRAY), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

Vehicle_Classes = [5, 34, 49, 65, 87, 199]

print(f)
print(w)
kernel = np.ones((3, 3), np.uint8)
data = json.load(f)
c = 0
for key in data.keys():
    # if key == "frame142":
    #     pass
    depth = Depths[c]
    # depth = Depths[0]
    im = np.array(Loaded_Images[c])
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    # print(im.shape)
    c += 1
    Frame_data = data[key]
    for Object in Frame_data:
        Center = Frame_data[Object]["Center"]
        Center_depth = float(depth[Center[1]][Center[0]])
        Frame_data[Object]["Center"] = [Center[0], Center[1], Center_depth]
        if Frame_data[Object]["Class"] == 40:
            Box = Frame_data[Object]["Box"]
            Cropped_Image = im[int(Box[1]):int(Box[3]), int(Box[0]):int(Box[2])]
            cv2.imwrite("/media/storage/lost+found/CV/P3/Cropped.png", Cropped_Image)
            Cropped_Image = cv2.cvtColor(Cropped_Image, cv2.COLOR_BGR2HSV)
            # Gray = cv2.cvtColor(Cropped_Image, cv2.COLOR_BGR2GRAY)
            # Contour, _ = cv2.findContours(Gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # cv2.drawContours(Cropped_Image, Contour, -1, (0, 255, 0), 3)
            # Green_mask = cv2.inRange(Cropped_Image, (80, 0, 120), (200, 255, 255))
            Green_mask = cv2.inRange(Cropped_Image, (80, 30, 140), (200, 255, 255))
            lower1 = np.array([0, 20, 120])
            upper1 = np.array([25, 255, 255])
            lower2 = np.array([170,20,20])
            upper2 = np.array([179,255,255])
            lower_mask = cv2.inRange(Cropped_Image, lower1, upper1)
            upper_mask = cv2.inRange(Cropped_Image, lower2, upper2)
            Red_mask = lower_mask + upper_mask
            
            mask = Green_mask
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=2)
            BGR_Image = cv2.cvtColor(Cropped_Image, cv2.COLOR_HSV2BGR)
            Color = cv2.bitwise_and(BGR_Image, BGR_Image, mask = mask)
            cv2.imwrite("/media/storage/lost+found/CV/P3/Green_Masked.png", Color)
            Gray = cv2.cvtColor(Color, cv2.COLOR_BGR2GRAY)
            cv2.imwrite("/media/storage/lost+found/CV/P3/Gray.png", Gray)
            Green_Contour, _ = cv2.findContours(Gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)                
            
            mask = Red_mask
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=2)
            Color = cv2.bitwise_and(BGR_Image, BGR_Image, mask = mask)
            cv2.imwrite("/media/storage/lost+found/CV/P3/Red_Masked.png", Color)
            Gray = cv2.cvtColor(Color, cv2.COLOR_BGR2GRAY)
            Red_Contour, _ = cv2.findContours(Gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            Frame_data[Object]["Color"] = "None"  
            if len(Red_Contour) == 0 and len(Green_Contour) != 0:
                Frame_data[Object]["Color"] = "Green"
                UP_Score = cv2.matchShapes(UP_Contour[0], Green_Contour[0], 1, 0.0)
                # Left_Score = cv2.matchShapes(Left_Contour[0], Green_Contour[0], 1, 0.0)
                # Right_Score = cv2.matchShapes(Right_Contour[0], Green_Contour[0], 1, 0.0)
                Down_Score = cv2.matchShapes(Down_Contour[0], Green_Contour[0], 1, 0.0)
                Scores = [UP_Score, Down_Score]
                min_score = min(Scores)
                if min_score < 0.25:
                    if UP_Score == min_score:
                        Frame_data[Object]["Arrow"] = "UP"
                    elif Down_Score == min_score:
                        Frame_data[Object]["Arrow"] = "DOWN"
                else:
                    Frame_data[Object]["Arrow"] = "None"
            elif len(Green_Contour) == 0 and len(Red_Contour) != 0:
                Red_Area = cv2.contourArea(Red_Contour[0])
                if Red_Area < 70:
                    Frame_data[Object]["Color"] = "Red"

            elif len(Green_Contour) != 0 and len(Red_Contour) != 0:
                Green_Area = cv2.contourArea(Green_Contour[0])
                Red_Area = cv2.contourArea(Red_Contour[0])
                if Red_Area > Green_Area and Red_Area < 70:
                    Frame_data[Object]["Color"] = "Red"
                elif Green_Area > Red_Area:
                    Frame_data[Object]["Color"] = "Green"
                    UP_Score = cv2.matchShapes(UP_Contour[0], Green_Contour[0], 1, 0.0)
                    # Left_Score = cv2.matchShapes(Left_Contour[0], Green_Contour[0], 1, 0.0)
                    # Right_Score = cv2.matchShapes(Right_Contour[0], Green_Contour[0], 1, 0.0)
                    Down_Score = cv2.matchShapes(Down_Contour[0], Green_Contour[0], 1, 0.0)
                    print(UP_Score)
                    Scores = [UP_Score, Down_Score]
                    min_score = min(Scores)
                    if min_score < 0.25:
                        if UP_Score < 0.25:
                            Frame_data[Object]["Arrow"] = "UP"
                        elif Down_Score == min_score:
                            Frame_data[Object]["Arrow"] = "DOWN"
                    else:
                        Frame_data[Object]["Arrow"] = "None"
                    
            
            # Contour = np.concatenate(Contour)
            # Rect = cv2.boundingRect(Contour)
            # Recropped = mask[Rect[1]:Rect[1]+Rect[3], Rect[0]:Rect[0]+Rect[2]]
            # cv2.imwrite("/media/storage/lost+found/CV/P3/Recropped.png", Recropped)
        elif Frame_data[Object]["Class"] in Vehicle_Classes:
            if key == "frame594":
                continue
            NCenter = Frame_data[Object]["NCenter"]
            if NCenter[0] >1279:
                NCenter[0] = 1279
            if NCenter[0] < 0:
                NCenter[0] = 0
            if NCenter[1] > 959:
                NCenter[1] = 959
            if NCenter[1] < 0:
                NCenter[1] = 0
            NCenter_depth = float(depth[NCenter[1]][NCenter[0]])
            Frame_data[Object]["NCenter"] = [NCenter[0], NCenter[1], NCenter_depth]
            Center_Angle_Point1 = [Center[0], Center_depth]
            Center_Angle_Point2 = [NCenter[0], NCenter_depth]
            Direction = np.arctan2(Center_Angle_Point2[1] - Center_Angle_Point1[1], Center_Angle_Point2[0] - Center_Angle_Point1[0])
            if Frame_data[Object]["Status"] == "Moving":
                Frame_data[Object]["Direction"] = Direction
            distance = NCenter_depth
            if distance < 3:
                horizontal_distance = abs(640 - Center_Angle_Point2[0])
                if horizontal_distance < 300:
                    Frame_data[Object]["Collision"] = 1
            else:
                Frame_data[Object]["Collision"] = 0

json.dump(data, indent=4, fp = w)
w.close()
f.close()
#################################################



# ################# Lane #######################

# f = open("/media/storage/lost+found/CV/P3/P3Data/Sequences/scene4/Undist/Scene4_Lanes_2D.json", "r")
# w = open("/media/storage/lost+found/CV/P3/P3Data/Sequences/scene4/Undist/Scene4_Lanes_3D.json", "w")


# data = json.load(f)
# c = 0
# for key in data.keys():
#     depth = Depths[c]
#     c += 1
#     Frame_data = data[key]
#     Updated_lane = {}
#     for lane in Frame_data:
#         [x1, y1], [x2, y2] = Frame_data[lane]["box"]
#         mean_x = (x1 + x2) // 2
#         mean_y = (y1 + y2) // 2
#         mean_depth = float(depth[mean_y][mean_x])
#         # depth1 = float(depth[y1-1][x1-1])
#         # depth2 = float(depth[y2-1][x2-1])
#         Frame_data[lane]["box_mid"] = [[mean_x, mean_y, mean_depth]]
#         # Frame_data[lane]["label"] = Frame_data[lane]["label"]
#         # Frame_data[lane]["color"] = Frame_data[lane]["color"]
#         # Frame_data[lane]["arrow"] = Frame_data[lane]["arrow"]
#         # Frame_data[lane] = Updated_lane
#     # data[key] = Frame_data

# json.dump(data, indent=4, fp = w)
# w.close()
# f.close()
# ################################################
    

##################### YOLOv8 ######################
# All_Results = []

# Final_Dict = {}
# Depths = []
# for im_path in sorted(glob(Images)):
#     im = PIL.Image.open(im_path).convert("RGB")
#     depth = zoe.infer_pil(im)
#     Depths.append(depth)
#     results = model(im, stream=False, conf = 0.3)
#     All_Results.append(results)

# for results in All_Results:
#     Int_Dict = {}
#     for result in results:
#         Boxes = result.boxes.xyxy.to('cpu').numpy().astype(int)
#         Classes = result.boxes.cls.to('cpu').numpy().astype(int)
#         # result.show()
#         Boxes = Boxes.tolist()
#         Classes = Classes.tolist()

#         for box in Boxes:
#             Dict = {}
#             Dict["Class"] = Classes[Boxes.index(box)]
#             x1, y1, x2, y2 = box
#             mean_x = (x1 + x2) // 2
#             mean_y = (y1 + y2) // 2
#             Dict["XYXY"] = [x1, y1, x2, y2]
#             Dict["Depth"] = float(depth[mean_y][mean_x])
#             print(depth[mean_y][mean_x])
#             Int_Dict[str(Boxes.index(box))] = Dict

#     Final_Dict['frame'+str(All_Results.index(results))] = Int_Dict


# file = open("/media/storage/lost+found/CV/P3/P3Data/Sequences/scene7/Scene7_Objects.json", "w")
# json.dump(Final_Dict, indent=4, fp = file)
# file.close()


# print(Boxes)
# print(Classes)
#################################################