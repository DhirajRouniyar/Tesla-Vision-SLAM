# import Find_world_coord as fwc
import numpy as np
import json
import Find_world_coord_copy as fwc

# Object_data = json.load(open(r'D:\WPI\Computer Vision\Einstein Vision\Phase 1\Sub files\src\Json\Detic\Scene5_Objects_Detic_3D_Optical_Flow_3.json', 'r'))
# Object_data = json.load(open(r'D:\WPI\Computer Vision\Einstein Vision\Phase 1\Sub files\src\Json\Scene5_Objects.json', 'r'))
# Angle_data = json.load(open(r'D:\WPI\Computer Vision\Einstein Vision\Phase 1\Sub files\src\Json\Detic\Scene5_Cars_angle.json', 'r'))
# Lane_data = json.load(open(r'D:\WPI\Computer Vision\Einstein Vision\Phase 1\Sub files\src\Json\Detic\Scene5_Lanes_3D.json', 'r'))
# Trash_data = json.load(open(r'D:\WPI\Computer Vision\Einstein Vision\Phase 1\Sub files\src\Json\Scene5_Trash_3D.json', 'r'))
# S_data = json.load(open(r'D:\WPI\Computer Vision\Einstein Vision\Phase 1\Sub files\src\Json\Scene5_Speed_3D.json', 'r'))

def Get_Object_Data(data, Angle_data):
    Object_pd = {}
    for frame_key, frame_data in data.items():  
        print("Frame:", frame_key)
        Object_p = []

        # Access object data within the frame
        for obj_key, obj_data in frame_data.items():
        
            rec_coord = obj_data['XYXY']
            top_left = rec_coord[:2]
            bottom_right = rec_coord[2:]
            depth = obj_data['Depth']
            obj_class = obj_data['Class']
            
            u = (top_left[0] + bottom_right[0]) // 2
            v = (top_left[1] + bottom_right[1]) // 2
            if obj_class == 2 or obj_class == 7:
                Dists = []
                Index = []
                for i in range(0, len(Angle_data[str(frame_key)])):
                    dist = np.sqrt((u - Angle_data[str(frame_key)][str(i)]["Center"][0])**2 + (v - Angle_data[str(frame_key)][str(i)]["Center"][1])**2)
                    Dists.append(dist)
                    Index.append(i)
                if len(Dists) > 0: 
                    dist = min(Dists)
                    Index = Index[Dists.index(dist)]
                    Angle = Angle_data[str(frame_key)][str(Index)]["angle"]
                else:
                    Angle = 0
            X, Y, Z = fwc.main(u, v, depth)
        #     Object_p.append([u, v, depth, obj_class])
            Object_p.append([X, Y, Z, obj_class, Angle])
            print("Object_p:", Object_p)
        Object_pd[str(frame_key)] = Object_p
    return Object_pd

def Get_Object_Data_Detic(data, Angle_data):
    Object_pd = {}
    start_key = 0
    end_key = len(data.items())
    step = 5
    frame_index = 0
    for frame_key, frame_data in data.items():  
        # if frame_index % 5 == 0:    
            print("Frame:", frame_key)
            Object_p = []
         
            # Access object data within the frame
            for obj_key, obj_data in frame_data.items():
                obj_class = obj_data['Class']
                obj_center = obj_data['Center']
                # obj_tailight = obj_data['Tail_Light']
                # obj_color = obj_data['Color']
                

                u = obj_center[0]
                v = obj_center[1] 
                
                depth = obj_center[2]
                Angle = 0
                obj_tailight = 0
                obj_signalcolor = 0
                obj_status = 0
                obj_signalarrow = 0
                if obj_class == 40:
                    obj_signalcolor = obj_data['Color']  
                    # obj_signalarrow = obj_data['Arrow']

                if obj_class == 5 or obj_class == 87 or obj_class == 65:   #Car and Pickup Truck
                    obj_tailight = obj_data['Tail_Light']
                    if isinstance(obj_data, dict) and 'Status' in obj_data:
                        obj_status = obj_data['Status']
                        # print("Frame data", obj_statuss)
                    Dists = []
                    Index = []
                    for i in range(0, len(Angle_data[str(frame_key)])):
                        dist = np.sqrt((u - Angle_data[str(frame_key)][str(i)]["Center"][0])**2 + (v - Angle_data[str(frame_key)][str(i)]["Center"][1])**2)
                        Dists.append(dist)
                        Index.append(i)
                    if len(Dists) > 0: 
                        dist = min(Dists)
                        Index = Index[Dists.index(dist)]
                        Angle = Angle_data[str(frame_key)][str(Index)]["angle"]
                    else:
                        Angle = 0
                    if obj_class == 87 or obj_class == 65:
                        if isinstance(obj_data, dict) and 'Direction' in obj_data:
                            Angle = obj_data['Direction']
                    
                X, Y, Z = fwc.main(u, v, depth)
                
                Object_p.append([X, Y, Z, obj_class, Angle, obj_tailight, obj_signalcolor, obj_status])
                # print("Object_p:", Object_p)
            Object_pd[str(frame_key)] = Object_p
        # frame_index += 1  
    return Object_pd

def Get_Lane_Data(data):
    Lanes_pd = {}
    for frame_key, frame_data in data.items():
        
        # print("Frame:", frame_key)
        Lanes_p = []

        # Access object data within the frame
        for obj_key, obj_data in frame_data.items():

            [rec_coord] = obj_data['box_mid']
            u = rec_coord[0]
            v = rec_coord[1]
            depth = rec_coord[2]
            label = obj_data['label']
            Color = obj_data['color']
            arrow = obj_data['arrow']
            X, Y, Z = fwc.main(u, v, depth)
            # if Z < 6:
            #     Lanes_p.append([X, Y, -Z, label, arrow, Color])
            # else:
            Lanes_p.append([X, Y, Z, label, arrow, Color])
        Lanes_pd[str(frame_key)] = Lanes_p
    return Lanes_pd

# def Get_Angle_Data(data):
#     Angle_pd = {}
#     for frame_key, frame_data in data.items():
        
#         # print("Frame:", frame_key)
#         Angle_p = []

#         # Access object data within the frame
#         for obj_key, obj_data in frame_data.items():

#                 rec_coord = obj_data['Center']
#                 u = rec_coord[0]
#                 v = rec_coord[1]
#                 angle = obj_data['angle']
#                 angle = np.around(angle, decimals=4)
#                 Angle_p.append([u, v, angle])
                
#         Angle_pd['frame'+str(frame_key)] = Angle_p
#     return Angle_pd

def Get_Trash_Data(data):
        Trash_pd = {}
        for frame_key, frame_data in data.items():
                print("Frame:", frame_key)
                Trash_p = []
                # Access object data within the frame
                for obj_key, obj_data in frame_data.items():
                      
                    u = obj_data[0][0]
                    v = obj_data[0][1]
                    depth = obj_data[0][2]
                    if u is not None:
                        X, Y, Z = fwc.main(u, v, depth)
                #     Trash_p.append([u, v, depth, obj_class])
                        Trash_p.append([X, Y, Z])
                Trash_pd[str(frame_key)] = Trash_p
        return Trash_pd


def Speed_data(data):
        Speed_pd = {}
        for frame_key, frame_data in data.items():
                
                # print("Frame:", frame_key)
                Speed_p = []
                
                # Access object data within the frame
                for obj_key, obj_data in frame_data.items():
        
                        u = obj_data[0][0]
                        v = obj_data[0][1]
                        depth = obj_data[0][2] 
                        X, Y, Z = fwc.main(u, v, depth)
        #     Trash_p.append([u, v, depth, obj_class])
                        Speed_p.append([X, Y, Z])
                Speed_pd[str(frame_key)] = Speed_p
                        
        return Speed_pd

# Speed_pd = Speed_data(S_data)
# Object_pd = Get_Object_Data(Object_data, Angle_data)   
# Object_pd = Get_Object_Data_Detic(Object_data, Angle_data)  
# Lanes_pd = Get_Lane_Data(Lane_data)   
# Angles_pd = Get_Angle_Data(Angle_data)
# Trash_pd = Get_Trash_Data(Trash_data)
# print("Angles_pd:", Angles_pd['frame0'])
# print("Object_pd:", Object_pd['frame93'])
# print("Lanes_pd:", Lanes_pd['frame8'])
# print("Trash_pd:", Trash_pd['frame0'])
# print("Speed_pd:", Speed_pd['frame19'])
