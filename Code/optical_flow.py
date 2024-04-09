import numpy as np
import cv2
import json
import glob


def opticalflow(frame1, frame2, Center,num):

    Current = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    # cv2.imwrite('/media/storage/lost+found/CV/P3/Code/Optical_Flow/Frame1.png',frame1)
    # cv2.imwrite('/media/storage/lost+found/CV/P3/Code/Optical_Flow/Frame2.png',frame2)
    hsv[...,1] = 255
    Left_points = [[50,160],[100,160],[150,160],[200,160],[250,160],[300,160],[350,160],[400,160],[450,160],[500,160],[550,160],[600,160],[650,160],[700,160],[750,160],[800,160],[850,160],[900,160],[950,160],[50,320],[100,320],[150,320],[200,320],[250,320],[300,320],[350,320],[400,320],[450,320],[500,320],[550,320],[600,320],[650,320],[700,320],[750,320],[800,320],[850,320],[900,320],[950,320]]
    Right_points = [[50,1120],[100,1120],[150,1120],[200,1120],[250,1120],[300,1120],[350,1120],[400,1120],[450,1120],[500,1120],[550,1120],[600,1120],[650,1120],[700,1120],[750,1120],[800,1120],[850,1120],[900,1120],[950,1120],[50,960],[100,960],[150,960],[200,960],[250,960],[300,960],[350,960],[400,960],[450,960],[500,960],[550,960],[600,960],[650,960],[700,960],[750,960],[800,960],[850,960],[900,960],[950,960]]

    Next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(Current,Next,None, 0.5, 3, 40, 3, 20, 2, 0)
    magnitude, angle = cv2.cartToPolar(flow[...,0], flow[...,1])

    Threshold_List = []
    for i in range(len(Left_points)):
        y1, x1 = Left_points[i]
        disp = np.linalg.norm(flow[y1][x1])
        if disp > 2:
            Threshold_List.append(disp)
        y2, x2 = Right_points[i]
        disp = np.linalg.norm(flow[y2][x2])
        if disp > 2:
            Threshold_List.append(disp)
    Threshold_List = np.array(Threshold_List)
    Threshold = np.mean(Threshold_List)



    hsv[..., 0] = angle*180/np.pi/2
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite('/media/storage/lost+found/CV/P3/Code/Optical_Flow/Flow'+str(num)+'.png',bgr)
    x, y = Center[0],Center[1]
    # ipdb.set_trace()
    Car_Center_disp = flow[int(y)][int(x)]
    Car_flow_value = np.linalg.norm(Car_Center_disp)
    New_Car_Center = [int(x + Car_Center_disp[0]), int(y + Car_Center_disp[1])]
    
    if Car_flow_value - Threshold < 2:
        Status = 'Moving'
    else:
        Status = 'Parked'
    

    return Status, New_Car_Center


f = open("/media/storage/lost+found/CV/P3/P3Data/Sequences/scene1/Undist/Scene1_Objects_Detic_2D.json", "r")
w = open("/media/storage/lost+found/CV/P3/P3Data/Sequences/scene1/Undist/Scene1_Objects_Detic_2D_Optical_Flow.json", "w")
data = json.load(f)

Image_Paths = "/media/storage/lost+found/CV/P3/P3Data/Sequences/scene1/Undist/Front_Frames/frame*.jpg"
print(Image_Paths)
print(f)
print(w)
paths = sorted(glob.glob(Image_Paths))
Vehicle_Classes = [5, 34, 49, 65, 87, 199]
Steady_Classes = [40, 44, 53, 66, 127, 176, 267]
Images = []
for i in range(len(paths)):
    im = cv2.imread(paths[i])
    Images.append(im)

count = 0

for key in data.keys():
    if count == len(Images) - 1: 
        continue
    print(count)
    Im1 = Images[count]
    Im2 = Images[count+1]
    Frame_data = data[key]
    for object in Frame_data:
        if Frame_data[object]['Class'] in Vehicle_Classes:
            Center = Frame_data[object]['Center']
            Frame_data[object]['Status'],Frame_data[object]['NCenter'] = opticalflow(Im1, Im2, Center, count)
    count += 1


json.dump(data, indent = 4, fp = w)
w.close()
f.close()
