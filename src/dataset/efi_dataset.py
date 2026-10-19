import os
import csv
import pickle
import warnings
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from .data_utils import estimate_normals, sample_points
from dataset.ocnn_utils import CustomTransform

class EfiDataset(Dataset):
    def __init__(self, root, csv_path, label_col, model_name, split='train', 
                 num_points=8192, task_type='classification', classes_list=None,
                 process_data=False, use_normals=False, use_fps=True):
        
        self.root = root
        self.csv_path = csv_path
        self.label_col = label_col
        self.model_name = model_name
        self.task_type = task_type
        self.use_normals = use_normals
        self.use_fps = use_fps
        self.split = split
        self.list_of_points = None
        self.list_of_labels = None
        self.list_of_pids = None

        if model_name == 'ocnn':
            self.npoints = 'all'
        else:
            self.npoints = num_points


        # 1. Parse CSV  -> meta_data
        self.meta_data = [] # Stores (plot_id, path, raw_label_value)
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('split', '').strip().lower() == self.split:
                    # PID fetch
                    pid = row.get('plot_id') or row.get('plot') or row.get('id')
                    # Label fetch
                    raw_val = row.get(self.label_col)

                    if pid and raw_val:
                        pc_path = os.path.join(self.root, f"{pid}.npy")
                        if os.path.exists(pc_path):
                            self.meta_data.append({
                                'plot_id': pid,
                                'path': pc_path,
                                'raw_label': raw_val
                            })

        # 2. Label Mapping
        self.cls_to_idx = {}
        self.class_names = []
        if self.task_type == 'classification':
            self.class_names = classes_list
            self.cls_to_idx = {name: i for i, name in enumerate(self.class_names)}
            self.num_classes = len(self.class_names)
            print(f"[{split}] Using class list from config: {self.class_names}")
        else:
            self.num_classes = 1 # Regression output dim
            print(f"[{self.split}] Regression task on '{label_col}'")

        # 3. Cache path
        cache_name = f"{self.split}_{self.npoints}_{self.label_col}"
        if self.use_normals: cache_name += "_normals"
        cache_name += "_fps" if self.use_fps else "_uniform"
        self.cache_path = os.path.join(self.root, f"cache_{cache_name}.pkl")
        
        # 4. Mode selection
        cache_exists = os.path.exists(self.cache_path)
        print(f"[{self.split}] process_data={process_data} cache_exists={cache_exists} "
              f"use_fps={self.use_fps} use_normals={self.use_normals}")
        if process_data:
            self._process_and_cache() # always (re)build
        elif cache_exists:
            self._load_cache()  # use cache
        else:
            print("No cache found. Loading data on-the-fly (slow).")
    
        # Specify octree build function
        if self.model_name == 'ocnn':

            self.load_octree_sample = CustomTransform(depth=6,
                                        full_depth=2,
                                        augment= True if self.split=='train' else False,
                                        angle=(0, 0, 40),
                                        interval=(1, 1, 1),
                                        jitter=0.125,
                                        downsample_prop=0.1)
                        
        else:
            self.load_octree_sample = None


    # --- Processing Function ---
    def _process_and_cache(self):
        print(f"[{self.split}] Processing data to {self.cache_path}...")
        processed_points_list = []
        processed_labels_list = []
        processed_pid_list = []

        for item in tqdm(self.meta_data):
            # Load and Sample Point Cloud
            points = np.load(item['path']).astype(np.float32)

            # OCNN uses all points to build octree
            if self.model_name != 'ocnn':
                points = sample_points(points, self.npoints, use_fps=self.use_fps)

            # Normals
            if self.use_normals:
                xyz = points[:, :3]
                normals = estimate_normals(xyz)
                points = np.concatenate([points, normals], axis=1)

            processed_points_list.append(points)
            processed_pid_list.append(item['plot_id'])

            # Process Label based on Task
            if self.task_type == 'classification':
                label_idx = self.cls_to_idx[item['raw_label']] # String -> Integer Index
                processed_labels_list.append(label_idx)
            else:
                processed_labels_list.append(float(item['raw_label']))

        # Save to disk
        data = {
            'points': processed_points_list, 
            'labels': processed_labels_list,
            'pids': processed_pid_list,
            'meta': {'cls_map': self.cls_to_idx, 
                    'use_fps': self.use_fps,
                    'use_normals': self.use_normals,
                    'npoints': self.npoints,
                    'label_col': self.label_col,
                    'split': self.split}
        }
        with open(self.cache_path, 'wb') as f:
            pickle.dump(data, f)
            
        # Load into memory
        self.list_of_points = processed_points_list
        self.list_of_labels = processed_labels_list
        self.list_of_pids = processed_pid_list
    
    def _load_cache(self):
        print(f"Loading cached data from {self.cache_path}")
        with open(self.cache_path, 'rb') as f:
            data = pickle.load(f)
        
        self.list_of_points = data['points']
        self.list_of_labels = data['labels']
        self.list_of_pids = data.get('pids', None)
        self.cls_to_idx = data['meta'].get('cls_map', {})
        
        if len(self.meta_data) != len(self.list_of_points):
            warnings.warn(
                f"[{self.split}] Cache/meta mismatch: meta={len(self.meta_data)} "
                f"cached={len(self.list_of_points)}. Using cached length."
            )

    def __len__(self):
        if self.list_of_points is not None:
            return len(self.list_of_points)
        return len(self.meta_data)

    def __getitem__(self, index):
        # 1. Get Points
        if self.list_of_points is not None:
            points = self.list_of_points[index]
        else:
            # On-the-fly fallback
            item = self.meta_data[index]
            points = np.load(item['path']).astype(np.float32)

            # OCNN uses all points to build octree
            if self.model_name != 'ocnn':
                points = sample_points(points, self.npoints, use_fps=self.use_fps)

            if self.use_normals:
                normals = estimate_normals(points[:,:3])
                points = np.concatenate([points, normals], axis=1)

        # 2. Get Labels
        if self.list_of_points is not None:
            label = self.list_of_labels[index]
        else:
            # On-the-fly label parsing
            raw = self.meta_data[index]['raw_label']
            if self.task_type == 'classification':
                label = self.cls_to_idx[raw]
            else:
                label = float(raw)
        
        # 3. Get pid
        if self.list_of_pids is not None:
            pid = self.list_of_pids[index]
        else:
            # On-the-fly
            pid = self.meta_data[index]['plot_id']

        # 4. Return Correct Types
        if self.task_type == 'classification':
            return points, torch.tensor(label, dtype=torch.long), pid
        else:
            if self.model_name == 'ocnn':
                sample = self.load_octree_sample(points, index)
                sample['target'] = torch.tensor(label, dtype=torch.float32)
                sample['pid'] = pid
                return sample
            else:
                # For regression, return float32 (usually shape [1] or scalar)
                return points, torch.tensor(label, dtype=torch.float32), pid
        