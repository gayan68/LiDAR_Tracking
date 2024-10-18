import pickle
import open3d as o3d
from filterpy.kalman import KalmanFilter
import numpy as np
import math
import random
import json
import yaml

class BoxTracker:
    def __init__(self, bbox_3d):
        bbox_3d = np.array(bbox_3d)
        # Initialize Kalman Filter for tracking in 3D space (x, y, z positions + velocities)
        self.kf = KalmanFilter(dim_x=6, dim_z=3)
        
        # State Transition matrix
        dt = 1.0  # Time step (assumed to be 1 for simplicity)
        self.kf.F = np.array([[1, 0, 0, dt, 0, 0], 
                              [0, 1, 0, 0, dt, 0], 
                              [0, 0, 1, 0, 0, dt], 
                              [0, 0, 0, 1, 0, 0], 
                              [0, 0, 0, 0, 1, 0], 
                              [0, 0, 0, 0, 0, 1]])

        # Measurement Matrix
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0], 
                              [0, 1, 0, 0, 0, 0], 
                              [0, 0, 1, 0, 0, 0]])

        # Measurement Uncertainty (R matrix)
        self.kf.R = np.eye(3) * 0.1

        # Process noise (Q matrix)
        self.kf.Q = np.eye(6) * 0.1

        # Initialize state (x, y, z, vx, vy, vz)
        self.kf.x[:3, 0] = bbox_3d[:3]  # x, y, z

        # Initialize state covariance matrix (P matrix)
        self.kf.P = np.eye(6) * 10

    def predict(self):
        self.kf.predict()

    def update(self, bbox_3d):
        self.kf.update(bbox_3d[:3])  # Update with new 3D measurement

    def get_state(self):
        return self.kf.x[:3, 0]  # Return only the position part of the state

class bb_traking():
    def __init__(self, config) -> None:

        self.lidar_frequency = config['lidar_frequency']
        self.threshold = config['threshold'] #distance threshold
        self.countdown = config['countdown'] #kalman_filter prediction countdown
        self.velocity_threshold = config['velocity_threshold']
        self.ignore_velocity= config['ignore_velocity']
        self.box_colors= config['box_colors']

    # Function to create a 3x3 rotation matrix from a yaw angle
    def create_rotation_matrix(self, yaw):
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        # Rotation matrix for a yaw rotation (around z-axis)
        rotation_matrix = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])
        return rotation_matrix
    
    def calculate_distance(self, point1, point2):
        return math.sqrt((point2[0] - point1[0]) ** 2 +
                        (point2[1] - point1[1]) ** 2)
    
    def calculate_velocity(self, P1, P2):
        t = 1/self.lidar_frequency
        P1 = np.array(P1[:2])
        P2 = np.array(P2[:2])
        velocity = (P2 - P1) / (t)
        return velocity
    
    def generate_traking_id(self, vehicle_ids, box, velocity):
        new_velocity = True
        if len(vehicle_ids)==0:
            new_id = 0
        else:
            new_id = max(vehicle_ids.keys())+1
        tracker = BoxTracker(box)
        vehicle_ids[new_id] = [0,0,0,0,float('inf') ,box ,tracker ,self.countdown ,velocity, new_velocity] #x,y,z,mapped,distance,box,tracker,coutdown,velocity
        tracker.predict()
        predicted_bbox_center = tracker.get_state().tolist()
        vehicle_ids[new_id][:3] = predicted_bbox_center
        return vehicle_ids, new_id
    
    def create_bb_box(self, box, key):
        center = np.array(box[:3])  # First 3 values are the center
        extent = np.array(box[3:6])  # Next 3 values are the extent
        yaw = box[6]  # Last value is the yaw angle
        rotation = self.create_rotation_matrix(yaw) # Create a rotation matrix from yaw
        bbox_3d = o3d.geometry.OrientedBoundingBox(center, rotation, extent)
        color = self.box_colors[key % len(self.box_colors)] #best_match_key % len(self.box_colors)
        bbox_3d.color = color

        res_box = {'uuid': '0b80a037-a160-46cc-baf7-adc7654783ae', 'label': 'Car', 'position': {'x': center[0], 'y': center[1], 'z': center[2]}, 'dimensions': {'x': extent[1], 'y': extent[0], 'z': extent[2]}, 'yaw': yaw, 'stationary': True, 'camera_used': None, 'attributes': {'state': 'Parked'}, 'points_count': 0, 'key':key, 'color':color }
        
        return bbox_3d, res_box
    
    def visualize_bounding_boxes(self, frames):
        vehicle_ids = {0: [0,0,0,0,float('inf'),[],0,self.countdown,[0,0]]} #x,y,z,mapped,distance,box,tracker,coutdown,velocity
        o3d_boxes = []
        frame_results = []
        #tracker = BoxTracker(bbox_detections[0])
        for f, frame_boxs in enumerate(frames):
            
            #reset mapped count to 0
            for key in vehicle_ids:
                vehicle_ids[key][3] = 0

            box_results = []
            for b, box in enumerate(frame_boxs): #If you want to see few sample boxes, limit from here ex: frame_boxs[:5]
                
                center = np.array(box[:3])  # First 3 values are the center

                #Calculate Velocity
                """
                velocity = 0
                if (f > 0) and (len(frames[f-1]) >= len(frames[f])):
                    previous_center = np.array(frames[f-1][b][:3])
                    velocity = self.calculate_velocity(center, previous_center)
                """

                #Search closest mapping
                best_match_key = 0
                best_match_distance = float('inf')
                force_add=False
                if f>0:  #Ignore traking for the 1st frame
                    for key, value in vehicle_ids.items():
                        distance = self.calculate_distance(center, value[:3])
                        #print(f"Frame: {f} | Key: {key} | Box: {b} | DISTANCE: {distance}")
                        if (distance<best_match_distance) and (distance<self.threshold):

                            velocity = [0,0]
                            if f>0:
                                previous_center = np.array(value[5][:3])
                                velocity = self.calculate_velocity(center, previous_center)
                            velocity_diff = np.linalg.norm(np.array(value[8]) - np.array(velocity))   

                            velocity_diff = np.linalg.norm(np.array(vehicle_ids[key][8]) - np.array(velocity))
                            #print(f"velocity_diff: {velocity_diff} | Old:{vehicle_ids[key][8]} | New:{velocity} | distance:{distance}")
                            if np.array_equal(vehicle_ids[key][8], [0, 0]) or (self.ignore_velocity) or (velocity_diff<self.velocity_threshold):
                                #print(f"Frame: {f} | Key: {key} | Box: {b} | DISTANCE: {distance}")
                                #Map existing Traking ID if the ID is not claimed by any other car or the distance is lower than 
                                if (vehicle_ids[key][3] == 0) or ((vehicle_ids[key][3] > 0) and (distance<vehicle_ids[key][4])) :   
                                    if (vehicle_ids[key][3] > 0) and (distance<vehicle_ids[key][4]): #Found a better box for traking ID. So reassign.
                                        force_add = True
                                        tmp = vehicle_ids[key][5]
                                                                
                                    best_match_distance = distance
                                    best_match_key = key
                                    vehicle_ids[key][3] += 1
                                    vehicle_ids[key][4] = distance
                                    tmp = vehicle_ids[key][5]
                                    vehicle_ids[key][5] = box

                                    tracker = vehicle_ids[key][6]
                                    tracker.update(box)
                                    tracker.predict()
                                    predicted_bbox_center = tracker.get_state().tolist()
                                    real_pos = vehicle_ids[key][:3]

                                    vehicle_ids[key][:3] = predicted_bbox_center
                                    #vehicle_ids[key][6] = tracker
                                    vehicle_ids[key][8] = velocity
                                    #print(f"f: {f} | Current: {center}")
                                    #print(f"f: {f} | Predict: {predicted_bbox_center}")
                                    #print(f"f: {f} | B: {b} | velocity: {velocity}")

                    if f>0:
                        #print(f"Real-Position: {real_pos}")
                        #print(f"Pred-Position: {predicted_bbox_center:}")
                        #print()
                        if best_match_key != 0:
                            vehicle_ids[best_match_key][6] = tracker
                        bb, bb_res = self.create_bb_box(box,best_match_key)
                        o3d_boxes.append(bb)
                        box_results.append(bb_res)

                    if force_add:
                        box = tmp
                        #print(f"Frame: {f} | Key: {key} | Box: {b} | DISTANCE: {distance}")
                    #print(f"Frame: {f} | BOX: {b} | DISTANCE: {best_match_distance} | KEY: {key} | colorID: {best_match_key}")
                    #print(f"Real-Position: {vehicle_ids[key][:3]} | Pred-Position: {predicted_bbox_center}")       

                if (best_match_distance == float('inf')) or (force_add) or (f == 0):
                    vehicle_ids, new_id = self.generate_traking_id(vehicle_ids, box, [0,0])
                    #print(f"CREATE NEW BOX: {new_id}")
                    bb, bb_res = self.create_bb_box(box, new_id)
                    o3d_boxes.append(bb)
                    box_results.append(bb_res)
            
            #Count down missing traking
            if f>0:
                for key in list(vehicle_ids.keys()): 
                    if (vehicle_ids[key][3] == 0) and (len(vehicle_ids)>0):
                        #print(f"Key: {key} | {vehicle_ids[key]}")
                        bb, bb_res = self.create_bb_box(vehicle_ids[key][5],key)
                        o3d_boxes.append(bb)
                        box_results.append(bb_res)
                        #print(f"AAAAAAAAAAAAXXX: {vehicle_ids[key][7]}")
                        vehicle_ids[key][7] -= 1
                        #print(f"BBBBBBBBBBBBBBB: {vehicle_ids[key][7]}")
                        if vehicle_ids[key][7] == -1:
                            del vehicle_ids[key]
            else:
                del vehicle_ids[0]
            
            #if f==1:
            #    frame_results.append({'cuboids':box_results})
            if f>-1:
                frame_results.append({'cuboids':box_results})
            if f==1:
                print(f"len(frame_results)1 : {len(frame_results)}")

            #if f ==100:
            #    break

        #print(vehicle_ids)

        # Visualize all bounding boxes
        o3d.visualization.draw_geometries(o3d_boxes)
        print(f"len(frame_results)2 : {len(frame_results)}")
        return frame_results


def load_bounding_boxes(file_path):
    with open(file_path, 'rb') as f:
        box_list = pickle.load(f)
    return box_list

def convert_to_python_types(data):
    if isinstance(data, dict):
        return {key: convert_to_python_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_python_types(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.float32, np.float64)):
        return float(data)
    elif isinstance(data, (np.int32, np.int64)):
        return int(data)
    else:
        return data

box_list = load_bounding_boxes("box_list.pkl")
config = yaml.safe_load(open("settings.yaml", 'r'))
tracker = bb_traking(config)
frame_results = tracker.visualize_bounding_boxes(box_list)

print(f"len(frame_results)3: {len(frame_results)}")
print(f"len(frame_results[0]): {len(frame_results[0])}")
print(f"len(frame_results[0]['cuboids']): {len(frame_results[0]['cuboids'])}")
print()


converted_results = convert_to_python_types(frame_results)

with open('3d_ann.json', 'w') as json_file:
    json.dump(converted_results, json_file, indent=4)
