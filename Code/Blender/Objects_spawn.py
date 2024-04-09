import os
import random
import bpy
from mathutils import Vector
import numpy as np
import Find_world_coord as fwc
import json
from Test import *

RENDER_PATH = r'D:\WPI\Computer Vision\Einstein Vision\Phase 1\Sub files\src\Blender output'

class Spawn_objs:

    def __init__(self):
        
        self.ped = bpy.data.objects['Pedestrian']  #Class 0
        self.car = bpy.data.objects['Car']         #Class 5
        self.car_R = bpy.data.objects['Car_R']     #Class 5
        self.car_P = bpy.data.objects['Car_P']     #Class 5
        self.suv = bpy.data.objects['Jeep']     #Class 34
        self.suv_r = bpy.data.objects['Jeep_R']     #Class 34
        self.suv_p = bpy.data.objects['Jeep_P']     #Class 34
        self.traflight_G = bpy.data.objects['Traffic_signalGreen'] #class 
        self.traflight_Y = bpy.data.objects['Traffic_signalYellow'] #class
        self.traflight_R = bpy.data.objects['Traffic_signalRed'] #class
        self.traflight = bpy.data.objects['Traffic_signal'] #class 40
        self.traflight_up = bpy.data.objects['Traffic_signalGreen_A']
        self.traflight_down = bpy.data.objects['Traffic_signalGreen_AD']
        self.trash = bpy.data.objects['Trashbox'] #class 44
        self.bicycle = bpy.data.objects['Bicycle'] #class 46
        # No Van class 49
        self.drum = bpy.data.objects['Drum'] #class 50
        self.truck = bpy.data.objects['Truck'] #class 65
        self.cone = bpy.data.objects['Cone'] #class 66
        self.pickuptruck = bpy.data.objects['PickupTruck']  #Class 87
        self.stopsign = bpy.data.objects['StopSign'] #class 127
        self.hydrant = bpy.data.objects['Hydrant'] #class 176
        
        # No Heavy truck class 199
        # self.speedboard = bpy.data.objects['Speed_board']
        self.speedboard = bpy.data.objects['Speed_board_H']
        self.speedhump = bpy.data.objects['Speed_Hump']
        
        # No crosswalk sign class 305

        self.shortlane_w = bpy.data.objects['Lane1_w']
        self.longlane_w = bpy.data.objects['Lane2_w']
        self.shortlane_y = bpy.data.objects['Lane1']
        self.longlane_y = bpy.data.objects['Lane2']
        self.jointlane = bpy.data.objects['Lane3']
        self.lanearrow = bpy.data.objects['LaneArrow']
        self.lanearrow_r = bpy.data.objects['LaneArrow_R']
        self.ped1 = bpy.data.objects['1751_1']
        self.ped2 = bpy.data.objects['1753_1']
        self.ped3 = bpy.data.objects['1755_1']
        self.ped4 = bpy.data.objects['1757_1']
        self.ped5 = bpy.data.objects['1759_1']
        self.ped6 = bpy.data.objects['1761_1']
        self.ped7 = bpy.data.objects['1763_1']
        self.ped8 = bpy.data.objects['1765_1']
        self.ped9 = bpy.data.objects['1767_1']
        self.ped10 = bpy.data.objects['1769_1']
        self.ped11 = bpy.data.objects['1771_1']
        self.ped12 = bpy.data.objects['1773_1']
        self.ped13 = bpy.data.objects['1775_1']
        self.ped14 = bpy.data.objects['1777_1']
        self.ped15 = bpy.data.objects['1779_1']
        
        self.road = bpy.data.objects['Base_Road']
        self.cam = bpy.data.objects['Camera']
        self.duplicated_objects = []
        

    def SetRenderSettings(self):
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'
        bpy.context.scene.cycles.taa_render_samples = 100
        bpy.context.scene.render.resolution_x = 1280
        bpy.context.scene.render.resolution_y = 960
        # Reset Frame to 0
        bpy.data.scenes['Scene'].frame_set(0)
        # Deselect all objects
        for obj_ in bpy.data.objects:
            obj_.select_set(False)
            obj_.animation_data_clear()

    def setLocKeyframeAndAdvance(self, obj, frame):
        obj.keyframe_insert(data_path="location", frame=frame)
        bpy.data.scenes['Scene'].frame_set(bpy.data.scenes['Scene'].frame_current + 1)
        # bpy.data.scenes['Scene'].frame_set(frame+1)

    def Render(self):
        path_dir = RENDER_PATH
        
        bpy.context.scene.camera = self.cam
        bpy.context.scene.render.filepath = os.path.join(path_dir, 'Frame%04d' % (bpy.data.scenes[0].frame_current))
        # bpy.context.scene.render.filepath = os.path.join(path_dir, 'Frame%04d' % frame)
        bpy.ops.render.render(write_still=True)
        
    

    def loop(self):
        # Object_data = json.load(open(r'D:\WPI\Computer Vision\Einstein Vision\Phase 1\Sub files\src\Json\Detic\Scene5_Objects_Detic_3D_Optical_Flow_3.json', 'r'))
        # Lane_data = json.load(open(r'D:\WPI\Computer Vision\Einstein Vision\Phase 1\Sub files\src\Json\Detic\Scene5_Lanes_3D.json', 'r'))
        # Angle_data = json.load(open(r'D:\WPI\Computer Vision\Einstein Vision\Phase 1\Sub files\src\Json\Detic\Scene5_Cars_angle.json', 'r'))
        Object_data = json.load(open(r'D:\WPI\Computer Vision\Einstein Vision\Phase 1\Sub files\src\Json\Detic\Scene13\Scene13_Objects_Detic_3D_Optical_Flow.json', 'r'))
        Lane_data = json.load(open(r'D:\WPI\Computer Vision\Einstein Vision\Phase 1\Sub files\src\Json\Detic\Scene13\Scene13_Lanes_3D.json', 'r'))
        Angle_data = json.load(open(r'D:\WPI\Computer Vision\Einstein Vision\Phase 1\Sub files\src\Json\Detic\Scene13\Scene13_Cars_Angle.json', 'r'))
        # Trash_data = json.load(open(r'D:\WPI\Computer Vision\Einstein Vision\Phase 1\Sub files\src\Json\Scene5_Trash_3D.json', 'r'))
        # S_data = json.load(open(r'D:\WPI\Computer Vision\Einstein Vision\Phase 1\Sub files\src\Json\Scene5_Speed_3D.json', 'r'))
        Object_pd = Get_Object_Data_Detic(Object_data, Angle_data)   
        Lanes_pd = Get_Lane_Data(Lane_data)
        # Speed_pd = Speed_data(S_data)
        # Trash_pd = Get_Trash_Data(Trash_data)
        # frame = 0
        
        frame_index = 0
        
        for frame_key, frame_data in Object_data.items():
            if frame_index % 4 == 0:  
                # frame_key = 'frame594'  
                print("Hello")   
                Objects = []
                # while bpy.data.scenes['Scene'].frame_current < MaxFrames:
                # frame += 1
                print("frame_current", bpy.data.scenes['Scene'].frame_current)
                # Delete duplicated objects
                for obj in self.duplicated_objects:
                    bpy.data.objects.remove(obj, do_unlink=True)
                self.duplicated_objects.clear()  # Clear the list for the next iteration
                # self.Render()
                # bpy.context.scene.frame_set(frame)

                for i in range(0, len(Lanes_pd[frame_key])):
                    Lane_label = Lanes_pd[frame_key][i][3]
                    Lane_Arrow = Lanes_pd[frame_key][i][4]
                    Lane_Color = Lanes_pd[frame_key][i][5]
                    X_lane = Lanes_pd[frame_key][i][0]
                    Y_lane = Lanes_pd[frame_key][i][1]
                    Z_lane = Lanes_pd[frame_key][i][2]
                    # self.Render()
                    if Lane_label == "dotted-line":
                        if Lane_Color == "white":
                            current_obj = self.shortlane_w
                        if Lane_Color == "yellow":
                            current_obj = self.shortlane_y
                    elif Lane_label == "solid-line":
                        if Lane_Color == "white":
                            current_obj = self.longlane_w
                        if Lane_Color == "yellow":
                            current_obj = self.longlane_y
                    elif Lane_label == "divider-line":
                        current_obj = self.jointlane

                    elif Lane_label == "road-sign-line":
                        if Lane_Arrow != "None":
                            if Lane_Arrow == "Up":
                                current_obj = self.lanearrow
                            elif Lane_Arrow == "Right":
                                current_obj = self.lanearrow_r
                    
                    current_obj.select_set(True)
                    bpy.context.view_layer.objects.active = current_obj

                    # Duplicate the selected object
                    bpy.ops.object.duplicate()

                    # The duplicated object is now the active object
                    new_obj = bpy.context.active_object
                    new_obj.location = Vector((X_lane, 0.02, -Z_lane))
                    print("new_obj.location", new_obj.location)
                    self.duplicated_objects.append(new_obj)
                    # Optionally, clear the selection
                    bpy.ops.object.select_all(action='DESELECT')


                for i in range(0, len(Object_pd[frame_key])):
                    
                    X_obj = Object_pd[str(frame_key)][i][0]
                    Y_obj = Object_pd[str(frame_key)][i][1]
                    Z_obj = Object_pd[str(frame_key)][i][2]
                    obj_class = Object_pd[str(frame_key)][i][3]
                    Angle_obj = Object_pd[str(frame_key)][i][4]
                    Tail_light = Object_pd[str(frame_key)][i][5]
                    Signal_clr = Object_pd[str(frame_key)][i][6]
                    Obj_status = Object_pd[str(frame_key)][i][7]
                    # obj_signalarrow = Object_pd[str(frame_key)][i][8] 
                    print('Tail Light', Tail_light)
                    print("Object Status", Obj_status)
                    if obj_class == 0:
                        if frame_key == "frame79":
                           current_obj = self.ped1
                        if frame_key == "frame480":
                            current_obj = self.ped2
                        if frame_key == "frame481":
                            current_obj = self.ped3
                        if frame_key == "frame482":
                            current_obj = self.ped4
                        if frame_key == "frame483":
                            current_obj = self.ped5
                        if frame_key == "frame484":
                            current_obj = self.ped6
                        if frame_key == "frame485":
                            current_obj = self.ped7
                        if frame_key == "frame486":
                            current_obj = self.ped8
                        if frame_key == "frame487":
                            current_obj = self.ped9
                        if frame_key == "frame488":
                            current_obj = self.ped10
                        if frame_key == "frame489":
                            current_obj = self.ped11
                        if frame_key == "frame490":
                            current_obj = self.ped12
                        if frame_key == "frame491":
                            current_obj = self.ped13
                        if frame_key == "frame492":
                            current_obj = self.ped14
                        if frame_key == "frame493":
                            current_obj = self.ped15
                        
                    
                    if obj_class == 5:
                        if Tail_light == 1 and Obj_status == 'Moving':
                            current_obj = self.car_R
                        elif Tail_light == 0 and Obj_status == 'Moving':
                            current_obj = self.car
                        elif Tail_light == 0 and Obj_status == 'Parked':
                            current_obj = self.car_P

                    if obj_class == 34:
                        if Tail_light == 1 and Obj_status == 'Moving':
                            current_obj = self.suv_r
                        elif Tail_light == 0 and Obj_status == 'Moving':
                            current_obj = self.suv
                        elif Tail_light == 0 and Obj_status == 'Parked':
                            current_obj = self.suv_p
                  
                    
                    if obj_class == 40:
                        if Signal_clr == 'Green':
                            current_obj = self.traflight_down
                        elif Signal_clr == 'Red':
                            current_obj = self.traflight_R
                        else:
                            current_obj = self.traflight

                    if obj_class == 44:
                        current_obj = self.trash

                    if obj_class == 46:
                        current_obj = self.bicycle

                    if obj_class == 53:
                        current_obj = self.drum

                    if obj_class == 65:
                        current_obj = self.truck    

                    if obj_class == 66:
                        current_obj = self.cone

                    if obj_class == 87:
                        current_obj = self.pickuptruck

                    if obj_class == 127:
                        current_obj = self.stopsign

                    if obj_class == 176:
                        current_obj = self.hydrant

                    if obj_class == 267:
                        current_obj = self.speedboard
                        
                                       
                    current_obj.select_set(True)
                    bpy.context.view_layer.objects.active = current_obj
                    bpy.ops.object.duplicate()
                    # The duplicated object is now the active object
                    new_obj = bpy.context.active_object
                    # new_obj.location = Vector((X_obj, 0, -Z_obj))
                    if obj_class == 5 or obj_class == 34 or obj_class == 46 or obj_class == 53 or obj_class == 65 or obj_class == 66 or obj_class == 176 or obj_class == 267:
                        if 0 < Z_obj < 3.5:
                            new_obj.location = Vector((X_obj, 0, -Z_obj))
                        elif 3.5 < Z_obj < 7:
                            new_obj.location = Vector((X_obj, 0, -Z_obj))
                        # elif 4 < Z_obj < 8:
                        #     new_obj.location = Vector((X_obj, 0, -Z_obj*3.5))
                        else: 
                            new_obj.location = Vector((X_obj, 0, Z_obj)) 

                    if obj_class == 267:
                        new_obj.location = Vector((X_obj, abs(Y_obj) + 0.5, -Z_obj))
                   
                    if obj_class == 5:    
                        new_obj.rotation_euler = Vector((1.57, Angle_obj + 1.65, 3.14))
                    
                    if obj_class == 44 :
                        if X_obj < 0:
                            X_obj = X_obj - 0.2
                        if 0 < Z_obj < 5:
                           new_obj.location = Vector((X_obj, 0, -Z_obj))
                        elif 5 < Z_obj < 8:
                            new_obj.location = Vector((X_obj, 0, -Z_obj*3.5))
                        # elif 4 < Z_obj < 8:
                        #     new_obj.location = Vector((X_obj, 0.04, -Z_obj*3.5))
                        else: 
                           new_obj.location = Vector((X_obj, 0, Z_obj)) 

                    if obj_class == 66 :
                       new_obj.rotation_euler = Vector((1.57, -1.57, -1.57))
                    if obj_class == 87 :
                        if 0 < Z_obj < 1.5:
                           new_obj.location = Vector((X_obj, 0.2, -Z_obj))
                        elif 1.5 < Z_obj < 7:
                            new_obj.location = Vector((X_obj, 0.2, -Z_obj))
                        # elif 4 < Z_obj < 8:
                        #     new_obj.location = Vector((X_obj, 0.04, -Z_obj*3.5))
                        else: 
                           new_obj.location = Vector((X_obj, 0.2, Z_obj)) 

                        new_obj.rotation_euler = Vector((0, Angle_obj + 1.57, 0))
                    
                    if obj_class == 127 or obj_class == 40:
                        if 0 < Z_obj < 3.5:
                           new_obj.location = Vector((X_obj, abs(Y_obj)+0.1, -Z_obj))
                        elif 3.5 < Z_obj < 7:
                            new_obj.location = Vector((X_obj, abs(Y_obj)+0.1, -Z_obj))
                        # elif 4 < Z_obj < 8:
                        #     new_obj.location = Vector((X_obj, abs(Y_obj), -Z_obj*3.5))
                        else: 
                           new_obj.location = Vector((X_obj, abs(Y_obj)+0.1, Z_obj)) 
                    
                    if obj_class == 0:
                        if 0 < Z_obj < 3.5:
                           new_obj.location = Vector((X_obj-0.1, abs(Y_obj) + 0.26, -Z_obj))
                        elif 3.5 < Z_obj < 7:
                            new_obj.location = Vector((X_obj-0.1, abs(Y_obj) + 0.26, -Z_obj*1.5))
                        # elif 4 < Z_obj < 8:
                        #     new_obj.location = Vector((X_obj, abs(Y_obj) + 0.26, -Z_obj*3.5))
                        else: 
                           new_obj.location = Vector((X_obj-0.1, abs(Y_obj) + 0.26, Z_obj)) 

                    print("Obj_class", obj_class)
                    print("new_obj.location", new_obj.location)
                    self.duplicated_objects.append(new_obj)
                    # Optionally, clear the selection
                    bpy.ops.object.select_all(action='DESELECT')

                Speed_Hump = True        
                if Speed_Hump == True:
                    if frame_key == "frame587":
                        self.speedhump.location = Vector((0, 0.1, -3)) 
                    if frame_key == "frame588":
                        self.speedhump.location = Vector((0, 0.1, -2.8)) 
                    if frame_key == "frame589":
                        self.speedhump.location = Vector((0, 0.1, -2.5)) 
                    if frame_key == "frame590":
                        self.speedhump.location = Vector((0, 0.1, -2.3)) 
                    if frame_key == "frame591":
                        self.speedhump.location = Vector((0, 0.1, -2)) 
                    if frame_key == "frame592":
                        self.speedhump.location = Vector((0, 0.1, -1.8)) 
                    if frame_key == "frame593":
                        self.speedhump.location = Vector((0, 0.1, -1.5)) 
                    if frame_key == "frame594":
                        self.speedhump.location = Vector((0, 0.1, -1.2)) 
                    if frame_key == "frame595":
                        self.speedhump.location = Vector((0, 0.1, -1)) 
                    if frame_key == "frame596":
                        self.speedhump.location = Vector((0, 0.1, -0.8)) 
                    if frame_key == "frame597":   
                        self.speedhump.location = Vector((0, 0.1, -0.3))
                    # new_obj =  self.speedhump     
                self.setLocKeyframeAndAdvance(new_obj, bpy.data.scenes['Scene'].frame_current)

                    # self.setLocKeyframeAndAdvance(new_obj, frame)
                self.Render()
            frame_index += 1
        
       

def main():
    obsInstance = Spawn_objs()
    obsInstance.SetRenderSettings()
    obsInstance.loop()
    

if __name__ == "__main__":
    main()

 
        