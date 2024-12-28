import os
import pickle5 as pickle
import random
import pdb
import numpy as np
import pandas as pd
import warnings
import fnmatch
warnings.filterwarnings("ignore")



class CustomDataLayer():
    def __init__(self, seed):
        self.base_dir = '/home/aghiya'
        self.data_dict_keys = ['image', 'resolution', 'pid', 'bbox', 'center', 'skeleton_unnormed', 'skeleton', 'angle']
        self.seed = seed
        
    def rotate_and_flip_points(self, points, flip_flag=False):
    
        # Convert points to a numpy array with shape [23, 2] for transformations
        points = np.array(points)
        
        # Step 1: Calculate the center point for the current image
        center_x = np.mean(points[:, 0])  # Mean of x-coordinates
        center_y = np.mean(points[:, 1])  # Mean of y-coordinates
        center = np.array([center_x, center_y])
        
        # Step 2: Translate points to the origin by subtracting the center coordinates
        translated_points = points - center  # Shape [23, 2]
        
        # Note: following comments are wrong
        # Step 3: Rotate each point
        if not flip_flag:
            # Counter-clockwise rotation for flipped data
            rotated_points = np.array([[-y, x] for x, y in translated_points])  # Shape [23, 2]
        else:
            # Clockwise rotation for non-flipped data
            rotated_points = np.array([[y, -x] for x, y in translated_points])  # Shape [23, 2]
        
        # Step 4: Translate back to the original center
        rotated_points += center
        
        # Step 5: Flip horizontally around center_y (inverting x-coordinates relative to center_y)
        flipped_points = rotated_points.copy()
        flipped_points[:, 0] = center_x - (flipped_points[:, 0] - center_x)  # Flip x-coordinates around center_x
    
        return flipped_points.tolist()
        
    def norm_coords(self, skeleton_dict):
    
        # Convert the list of lists to a NumPy array
        arr = np.array(skeleton_dict)
        
        # pdb.set_trace()

        # Separate the first and second columns
        x_col = arr[:, 0]
        y_col = arr[:, 1]

        # Get the min and max values for each column
        min_x = np.min(x_col)
        max_x = np.max(x_col)

        min_y = np.min(y_col)
        max_y = np.max(y_col)

        # Normalize each column separately
        normalized_x_col = (x_col - min_x) / (max_x - min_x)
        normalized_y_col = (y_col - min_y) / (max_y - min_y)

        # Combine the normalized columns back into a 2D array
        normalized_arr = np.column_stack((normalized_x_col, normalized_y_col))

        # Convert the normalized NumPy array back to a list of lists
        skeleton_dict_norm = normalized_arr.tolist()
        
        return skeleton_dict_norm
        
        
    def augment_y(self, data_dict):
        # Flips on x-axis
    
        augment_dict = {k: [] for k in self.data_dict_keys}
    
        # Iterate over all sequences
        for i in range(len(data_dict['image'])):
            
            # seq_dict = {k: None for k in self.data_dict_keys}
        
            for k in self.data_dict_keys:
                
                # Flip xtl, xbr values (bbox format is [xtl, ytl, xbr, ybr]
                if k == 'bbox':
                    augmented_list = [[1080 - val if j % 2 == 1 else val for j, val in enumerate(bbox)] for bbox in data_dict[k][i]]
                # Flip x coord for unnormed skeleton points (format is list of skeletons per frame, skeletons are list of 23 points, each point is list of 2 elements [x, y])
                elif k == 'skeleton_unnormed':
                    augmented_list = [[[x, 1080 - y] for [x, y] in skeleton_list] for skeleton_list in data_dict['skeleton_unnormed'][i]]
                    
                # Normalize skeleton coordinates
                elif k == 'skeleton':
                    augmented_list = [self.norm_coords(skeleton_dict) for skeleton_dict in augment_dict['skeleton_unnormed'][i]]
                    
                # "skeleton" processed before angle, so this should work
                elif k == 'angle':
                    augmented_list = [np.nan_to_num(self.calc_angles(skeleton_list[:23]), nan=.5) for skeleton_list in augment_dict['skeleton'][i]]
                
                # flip center
                elif k == 'center':
                    augmented_list = [[1 - bbox[0], bbox[1]] for bbox in data_dict[k][i]]
                    
                # Note: "skeleton_unnormed" should be flipped, but not relevant
                else:
                    augmented_list = data_dict[k][i]
                    
                augment_dict[k].append(augmented_list)
                
        return augment_dict
        
    def augment_x(self, data_dict):
    
        # Flips on y-axis for the time being
    
        augment_dict = {k: [] for k in self.data_dict_keys}
    
        # Iterate over all sequences
        for i in range(len(data_dict['image'])):
            
            # seq_dict = {k: None for k in self.data_dict_keys}
        
            for k in self.data_dict_keys:
                
                # Flip xtl, xbr values (bbox format is [xtl, ytl, xbr, ybr]
                if k == 'bbox':
                    augmented_list = [[1920 - val if j % 2 == 0 else val for j, val in enumerate(bbox)] for bbox in data_dict[k][i]]
                # Flip x coord for unnormed skeleton points (format is list of skeletons per frame, skeletons are list of 23 points, each point is list of 2 elements [x, y])
                elif k == 'skeleton_unnormed':
                    augmented_list = [[[1920 - x, y] for [x, y] in skeleton_list] for skeleton_list in data_dict['skeleton_unnormed'][i]]
                    
                # Normalize skeleton coordinates
                elif k == 'skeleton':
                    augmented_list = [self.norm_coords(skeleton_dict) for skeleton_dict in augment_dict['skeleton_unnormed'][i]]
                    
                # "skeleton" processed before angle, so this should work
                elif k == 'angle':
                    augmented_list = [np.nan_to_num(self.calc_angles(skeleton_list[:23]), nan=.5) for skeleton_list in augment_dict['skeleton'][i]]
                
                # flip center
                elif k == 'center':
                    augmented_list = [[1 - bbox[0], bbox[1]] for bbox in data_dict[k][i]]
                    
                # Note: "skeleton_unnormed" should be flipped, but not relevant
                else:
                    augmented_list = data_dict[k][i]
                    
                augment_dict[k].append(augmented_list)
                
        return augment_dict
        
    def compute_angle(self, a, b):
        # Ensure vectors are numpy arrays
        a = np.array(a)
        b = np.array(b)
        
        # Calculate the dot product
        dot_product = np.dot(a, b)
        
        # Calculate the magnitudes (norms) of the vectors
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        cos_theta = dot_product / (norm_a * norm_b)
        
        # cos_theta = 1
        
        # try:
            # # Calculate the cosine of the angle
            # cos_theta = dot_product / (norm_a * norm_b)
        # except RuntimeWarning:
            # print(a, b, dot_product, norm_a, norm_b)
            # exit
        
        # Clip the cosine value to avoid possible floating-point precision issues
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        
        # Calculate the angle in radians
        angle_radians = np.arccos(cos_theta)
        
        # Optionally, convert to degrees and normalize
        angle_degrees = np.degrees(angle_radians)
        
        # Normalize degrees
        angle_degrees = angle_degrees / 180
        
        return angle_degrees

    def calc_angles(self, skeleton_dict):
        
        skeleton_dict = np.asarray(skeleton_dict)
        
        # Shoulder-neck (using nose)
        shoulder_midpoint = [(skeleton_dict[1][0] + skeleton_dict[2][0])/2,\
                             (skeleton_dict[1][1] + skeleton_dict[2][1])/2]
        
        left_shoulder_nose = self.compute_angle(skeleton_dict[1] - shoulder_midpoint, skeleton_dict[0] - shoulder_midpoint)
        right_shoulder_nose = self.compute_angle(skeleton_dict[2] - shoulder_midpoint, skeleton_dict[0] - shoulder_midpoint)
        
        # Armpit
        left_armpit = self.compute_angle(skeleton_dict[3] - skeleton_dict[1], skeleton_dict[2] - skeleton_dict[1])
        right_armpit = self.compute_angle(skeleton_dict[4] - skeleton_dict[2], skeleton_dict[2] - skeleton_dict[1])
        
        # Elbow
        left_elbow = self.compute_angle(skeleton_dict[5] - skeleton_dict[3], skeleton_dict[1] - skeleton_dict[3])
        right_elbow = self.compute_angle(skeleton_dict[6] - skeleton_dict[4], skeleton_dict[2] - skeleton_dict[4])
        
        # Hip
        left_hip = self.compute_angle(skeleton_dict[9] - skeleton_dict[7], skeleton_dict[7] - skeleton_dict[1])
        right_hip = self.compute_angle(skeleton_dict[10] - skeleton_dict[8], skeleton_dict[8] - skeleton_dict[2])
        
        # Thigh
        left_thigh = self.compute_angle(skeleton_dict[9] - skeleton_dict[7], skeleton_dict[7] - skeleton_dict[8])
        right_thigh = self.compute_angle(skeleton_dict[10] - skeleton_dict[8], skeleton_dict[7] - skeleton_dict[8])
        
        # Knee
        left_knee = self.compute_angle(skeleton_dict[11] - skeleton_dict[9], skeleton_dict[7] - skeleton_dict[9])
        right_knee = self.compute_angle(skeleton_dict[12] - skeleton_dict[10], skeleton_dict[8] - skeleton_dict[10])
        
        angles = [left_shoulder_nose,
                  right_shoulder_nose,
                  left_armpit,
                  right_armpit,
                  left_elbow,
                  right_elbow,
                  left_hip,
                  right_hip,
                  left_thigh,
                  right_thigh,
                  left_knee,
                  right_knee
                 ]
        # print(angles)
        return angles
        
    def get_data(self, pickle_dir, min_consecutive, augment_x, augment_y):

        data_dict = {}

        for ddk in self.data_dict_keys:
            data_dict[ddk] = []

        # Iterate over all files in the directory
        for pickle_file in os.listdir(pickle_dir):
            # Check if the file is prefixed with "video_" or "set" and is a pickle file
            if (pickle_file.startswith("video_") or pickle_file.startswith("set")) and pickle_file.endswith(".pkl"):
                # Full path to the pickle file
                pickle_path = os.path.join(pickle_dir, pickle_file)
                
                # Pie skeleton data is split into multiple files
                if self.is_pie:
                
                    skeleton_pkl_fps = [
                        os.path.join(self.skeleton_dir, filename)
                        for filename in os.listdir(self.skeleton_dir)
                        if fnmatch.fnmatch(filename, f"{pickle_file.split('.')[0]}*")
                    ]
                    
                    skeleton_dict = {}
                    for fp in skeleton_pkl_fps:
                        with open(fp, 'rb') as f:
                            skeleton_dict.update(pickle.load(f))
                
                else:
                
                    skeleton_pkl_fp = self.skeleton_dir + "/" + pickle_file
                
                    with open(skeleton_pkl_fp, 'rb') as f:
                        skeleton_dict = pickle.load(f)
                        
                # pdb.set_trace()
                
                # Open the pickle file
                with open(pickle_path, 'rb') as f:
                    data = pickle.load(f)
                    
                    # Create list of lists for all peds data
                    image_list = []
                    res_list = []
                    pid_list = []
                    bbox_list = []
                    center_list = []
                        
                    for i in range(len(data[0]['ped_data'])):
                        image_list.append([])
                        res_list.append([])
                        pid_list.append([])
                        bbox_list.append([])
                        center_list.append([])
                    
                    # Iterate over the top-level list
                    for top_element in data:
                        # Get the list from the "ped_data" key
                        ped_data_list = top_element.get("ped_data", [])
                        
                        # Iterate over the "ped_data" list
                        for i in range(len(ped_data_list)):
                        
                            ped_element = ped_data_list[i]
                        
                            # Ped has valid data
                            if ped_element.get("actions") != -10:
                            
                                # Get the values for "pid" and "bbox"
                                image = top_element.get("path")
                                resolution = [1920, 1080]
                                pid = ped_element.get("pid")
                                bbox = ped_element.get("bbox")
                                # normalize centroid
                                center = [((bbox[0] + bbox[2])/2)/1920, ((bbox[1] + bbox[3])/2)/1080]
                                    
                                image_list[i].append(image)
                                res_list[i].append(resolution)
                                pid_list[i].append(pid)
                                bbox_list[i].append(bbox)
                                center_list[i].append(center)

                    # Filtering to only retain frames in consecutive sets of 60 frames or more
                    filtered_pid_list = []
                    filtered_image_list = []
                    filtered_bbox_list = []
                    filtered_center_list = []
                    filtered_res_list = []
                    skeleton_list = []
                    skeleton_norm_list = []
                    angles_list = []
                    
                    for i, pid_data in enumerate(pid_list):
                    
                        # get image names and convert to integer
                        image_fn_list = [int(os.path.splitext(os.path.split(x)[-1])[0]) for x in image_list[i]]
                    
                        if pid_data:
                            pid_skeleton_dict = skeleton_dict[pid_data[0]]
                            
                            sorted_frames = sorted([frame for frame in pid_skeleton_dict.keys() if frame in image_fn_list])
                            # pdb.set_trace()
                            ranges = []
                            current_start = sorted_frames[0]

                            for idx in range(1, len(sorted_frames)):
                                # Check if the current frame is consecutive to the previous one
                                if sorted_frames[idx] == sorted_frames[idx - 1] + 1:
                                    continue  # Stay in the same range
                                else:
                                    # We have reached the end of a consecutive range
                                    if (sorted_frames[idx - 1] - current_start + 1) >= min_consecutive:
                                        ranges.append((current_start, sorted_frames[idx - 1]))  # Append the range
                                    current_start = sorted_frames[idx]  # Start a new range

                            # Check the last group after the loop ends
                            if (sorted_frames[-1] - current_start + 1) >= min_consecutive:
                                ranges.append((current_start, sorted_frames[-1]))  # Append the last range
                        # pdb.set_trace()
                        if ranges:
                            for range_ in ranges:
                                start_idx, end_idx = range_
                                
                                # Append filtered data for the current group
                                
                                # with PIE, we have cases where the end_idx is not in the file list (presumably the skeleton detection found someone despite the official data not tracking that person at the point
                                # if self.is_pie:
                                    # end_idx = min(end_idx, image_fn_list[-1])

                                try:
                                
                                    # start_idx, end_idx are frame values, not direct idx values, 
                                    # so we need to find the corresponding indices by looking at the image files
                                    image_start_idx = image_fn_list.index(start_idx)
                                    image_end_idx = image_fn_list.index(end_idx)
                                    
                                except Exception:
                                    exit()
                                    # print(f"{pid_data[0]}")
                            
                                    # sorted_frames = sorted(pid_skeleton_dict.keys())
                                    # ranges = []
                                    # current_start = sorted_frames[0]
                                    # print(f"Now at frame {current_start}, ", end="")
                                    # for idx in range(1, len(sorted_frames)):
                                        
                                        # # Check if the current frame is consecutive to the previous one
                                        # if sorted_frames[idx] == sorted_frames[idx - 1] + 1:
                                            # print(f"{sorted_frames[idx]}, ", end="")
                                            # continue  # Stay in the same range
                                        # else:
                                            # print("...sequence finished")
                                            # # We have reached the end of a consecutive range
                                            # if (sorted_frames[idx - 1] - current_start + 1) >= min_consecutive:
                                                # ranges.append((current_start, sorted_frames[idx - 1]))  # Append the range
                                            # current_start = sorted_frames[idx]  # Start a new range
                                            # print(f"New sequence starting frame is {current_start}")

                                    # # Check the last group after the loop ends
                                    # if (sorted_frames[-1] - current_start + 1) >= min_consecutive:
                                        # ranges.append((current_start, sorted_frames[-1]))  # Append the last range
                                        
                                    # print(ranges)
                                    # print(image_fn_list)
                                    # exit()

                                
                                filtered_pid_list.append(pid_list[i][image_start_idx:image_end_idx + 1])
                                filtered_image_list.append(image_list[i][image_start_idx:image_end_idx + 1])
                                filtered_bbox_list.append(bbox_list[i][image_start_idx:image_end_idx + 1])
                                filtered_center_list.append(center_list[i][image_start_idx:image_end_idx + 1])
                                filtered_res_list.append(res_list[i][image_start_idx:image_end_idx + 1])

                                range_skeleton_list = []
                                range_skeleton_norm_list = []
                                range_angles_list = []
                                for j in range(start_idx, end_idx + 1):
                                    
                                    only_coords = []
                                    
                                    for k in range(17):
                                        # Don't take eyes and ears
                                        if k not in set([1, 2, 3, 4]):
                                            only_coords.append(pid_skeleton_dict[j][k][0:2])
                                        # only_coords.append([pid_skeleton_dict[j][k][1], pid_skeleton_dict[j][k][0]])
                                        
                                    
                                        
                                    only_coords = self.rotate_and_flip_points(only_coords)
                                    
                                    # pdb.set_trace()
                                    
                                    # Convert the list of lists to a NumPy array
                                    arr = np.array(only_coords)

                                    # Separate the first and second columns
                                    x_col = arr[:, 0]
                                    y_col = arr[:, 1]

                                    # Get the min and max values for each column
                                    min_x = np.min(x_col)
                                    max_x = np.max(x_col)

                                    min_y = np.min(y_col)
                                    max_y = np.max(y_col)

                                    # Normalize each column separately
                                    normalized_x_col = (x_col - min_x) / (max_x - min_x)
                                    normalized_y_col = (y_col - min_y) / (max_y - min_y)

                                    # Combine the normalized columns back into a 2D array
                                    normalized_arr = np.column_stack((normalized_x_col, normalized_y_col))

                                    # Convert the normalized NumPy array back to a list of lists
                                    only_coords_norm = normalized_arr.tolist()
                                
                                    range_skeleton_list.append(only_coords)
                                    range_skeleton_norm_list.append(only_coords_norm)
                                    
                                    # Compute angles (don't need normalized coordinates), replaces NaN with .5
                                    range_angles_list.append(np.nan_to_num(self.calc_angles(only_coords), nan=.5))
                                    # range_angles_list.append(self.calc_angles(only_coords))
                                    
                                # print(len(range_angles_list))
                                skeleton_list.append(range_skeleton_list)
                                skeleton_norm_list.append(range_skeleton_norm_list)
                                angles_list.append(range_angles_list)
                            # pdb.set_trace()

                    
                    for i in range(len(filtered_image_list)):
                        # if len(filtered_image_list[i]) != len(skeleton_list[i]):
                            # pdb.set_trace()
                        data_dict['image'].append(filtered_image_list[i])
                        data_dict['resolution'].append(filtered_res_list[i])
                        data_dict['pid'].append(filtered_pid_list[i])
                        data_dict['bbox'].append(filtered_bbox_list[i])
                        data_dict['center'].append(filtered_center_list[i])
                        data_dict['skeleton_unnormed'].append(skeleton_list[i])
                        data_dict['skeleton'].append(skeleton_norm_list[i])
                        data_dict['angle'].append(angles_list[i])
                    
        # Zip together all the lists from the dictionary to ensure synchronized shuffling
        combined = list(zip(*[data_dict[key] for key in data_dict]))

        random.seed(self.seed)
        # Shuffle the combined data
        random.shuffle(combined)

        # Unpack the shuffled data back into separate lists and update the original dictionary
        unzipped_data = list(zip(*combined))
        for i, key in enumerate(data_dict.keys()):
            data_dict[key] = list(unzipped_data[i])
            
        # augment data by flipping on horizontal
        if augment_x:
            aug_dict = self.augment_x(data_dict)
            
            for k in self.data_dict_keys:
                data_dict[k] += aug_dict[k]
                
        if augment_y:
            aug_dict = self.augment_y(data_dict)
            
            for k in self.data_dict_keys:
                data_dict[k] += aug_dict[k]
        
        return data_dict
        
    def get_split(self, split, min_consecutive, augment_x, augment_y):
        
        if split == 'train':
            return self.get_data(self.train_path, min_consecutive, augment_x, augment_y)
        elif split == 'test':
            return self.get_data(self.test_path, min_consecutive, augment_x, augment_y)
        elif split == 'val':
            return self.get_data(self.val_path, min_consecutive, augment_x, augment_y)

            
class JAAD(CustomDataLayer):
    def __init__(self, seed):
        super().__init__(seed)
        self.train_path = self.base_dir + '/jaadpie_data/sequences/jaad_all_all/train/combined'
        self.test_path = self.base_dir + '/jaadpie_data/sequences/jaad_all_all/test/combined'
        self.val_path = self.base_dir + '/jaadpie_data/sequences/jaad_all_all/val/combined'
        self.skeleton_dir = self.base_dir + '/jaadpie_data/skeleton/jaad'
        self.is_pie = False

class PIE(CustomDataLayer):
    def __init__(self, seed):
        super().__init__(seed)
        self.train_path = self.base_dir + '/jaadpie_data/sequences/pie/train/combined'
        self.test_path = self.base_dir + '/jaadpie_data/sequences/pie/test/combined'
        self.val_path = self.base_dir + '/jaadpie_data/sequences/pie/val/combined'
        self.skeleton_dir = self.base_dir + '/jaadpie_data/skeleton/pie'
        self.is_pie = True