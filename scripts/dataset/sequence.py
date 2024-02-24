from pathlib import Path
import weakref

import os
import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import imageio
from PIL import Image
import random

from dataset.representations import VoxelGrid
from utils.eventslicer import EventSlicer

import ipdb
from tqdm import tqdm

class Sequence(Dataset):
    # NOTE: This is just an EXAMPLE class for convenience. Adapt it to your case.
    # In this example, we use the voxel grid representation.
    #
    # This class assumes the following structure in a sequence directory:
    #
    # seq_name (e.g. zurich_city_11_a)
    # ├── disparity
    # │   ├── event
    # │   │   ├── 000000.png
    # │   │   └── ...
    # │   └── timestamps.txt
    # └── events
    #     ├── left
    #     │   ├── events.h5
    #     │   └── rectify_map.h5
    #     └── right
    #         ├── events.h5
    #         └── rectify_map.h5

    def __init__(self, flow_path: Path, seg_path: Path, mode: str='train', delta_t_ms: int=50, num_bins: int=15):
        assert num_bins >= 1
        assert delta_t_ms <= 100, 'adapt this code, if duration is higher than 100 ms'
        # assert flow_path.is_dir()
        # assert seg_path.is_dir()

        # NOTE: Adapt this code according to the present mode (e.g. train, val or test).
        self.mode = mode

        # Save output dimensions
        self.height = 480
        self.width = 640
        self.num_bins = num_bins

        # Set event representation
        self.voxel_grid = VoxelGrid(self.num_bins, self.height, self.width, normalize=True)

        self.locations = ['left', 'right']

        # Save delta timestamp in ms
        self.delta_t_us = delta_t_ms * 1000

        # # load disparity timestamps
        # disp_dir = seq_path / 'disparity'
        # assert disp_dir.is_dir()
        # self.timestamps = np.loadtxt(disp_dir / 'timestamps.txt', dtype='int64')
        
        # load flow timestamps
        if flow_path is not None:
            flow_dir = flow_path / 'flow' 
            assert flow_dir.is_dir()
            self.timestamps = np.loadtxt(flow_dir / 'forward_timestamps.txt', delimiter = ',', dtype='int64')

            flow_dir_ = flow_dir / 'forward'
            flow_gt_pathstrings = list()
            for entry in flow_dir_.iterdir():
                assert str(entry.name).endswith('.png')
                flow_gt_pathstrings.append(str(entry))
            flow_gt_pathstrings.sort()
            self.flow_gt_pathstrings = flow_gt_pathstrings
            
            assert len(self.flow_gt_pathstrings) == self.timestamps.shape[0]
        seg_dir = seg_path / 'semantic'
        seg_dir_ = seg_dir / 'left' / '11classes'
        seg_gt_pathstrings = list()
        for entry in seg_dir_.iterdir():
            assert str(entry.name).endswith('.png')
            seg_gt_pathstrings.append(str(entry))
        seg_gt_pathstrings.sort()
        self.seg_gt_pathstrings = seg_gt_pathstrings
        self.seg_timestamps = np.loadtxt(seg_dir / 'timestamps.txt', dtype='int64')
        remove_time_window=250
        self.seg_timestamps = self.seg_timestamps[(remove_time_window // 100 + 1) * 2:]
        del self.seg_gt_pathstrings[:(remove_time_window // 100 + 1) * 2]

        assert len(self.seg_gt_pathstrings) == self.seg_timestamps.shape[0]

        self.h5f = dict()
        self.rectify_ev_maps = dict()
        self.event_slicers = dict()

        ev_dir = seg_path / 'events'
        for location in self.locations:
            ev_dir_location = ev_dir / location
            ev_data_file = ev_dir_location / 'events.h5'
            ev_rect_file = ev_dir_location / 'rectify_map.h5'

            h5f_location = h5py.File(str(ev_data_file), 'r')
            self.h5f[location] = h5f_location
            self.event_slicers[location] = EventSlicer(h5f_location)
            with h5py.File(str(ev_rect_file), 'r') as h5_rect:
                self.rectify_ev_maps[location] = h5_rect['rectify_map'][()]


        self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)

    def events_to_voxel_grid(self, x, y, p, t, device: str='cpu'):
        t = (t - t[0]).astype('float32')
        t = (t/t[-1])
        x = x.astype('float32')
        y = y.astype('float32')
        pol = p.astype('float32')
        return self.voxel_grid.convert(
                torch.from_numpy(x),
                torch.from_numpy(y),
                torch.from_numpy(pol),
                torch.from_numpy(t))

    def getHeightAndWidth(self):
        return self.height, self.width

    @staticmethod
    def get_disparity_map(filepath: Path):
        assert filepath.is_file()
        disp_16bit = cv2.imread(str(filepath), cv2.IMREAD_ANYDEPTH)
        return disp_16bit.astype('float32')/256

    @staticmethod
    def get_flow_map(filepath: Path):
        assert filepath.is_file()
        flow_16bit = imageio.imread(filepath, format='PNG-FI')
        flow_x = (flow_16bit[:, :, 0].astype('float32') - 2**15) / 128
        flow_y = (flow_16bit[:, :, 1].astype('float32') - 2**15) / 128
        valid_pixels = flow_16bit[:, :, 2].astype(bool)

        flow_x = np.expand_dims(flow_x, axis=0)  # shape (H, W) --> (1, H, W)
        flow_y = np.expand_dims(flow_y, axis=0)

        flow_map = np.concatenate((flow_x, flow_y), axis = 0).astype(np.float32)

        return flow_map, valid_pixels
    
    @staticmethod
    def get_seg_label(filepath: Path):
        assert filepath.is_file()
        label = Image.open(str(filepath))
        label = np.array(label)
        return label
    
    @staticmethod
    def close_callback(h5f_dict):
        for k, h5f in h5f_dict.items():
            h5f.close()

    def __len__(self):
        # return len(self.seg_gt_pathstrings)
        return (self.seg_timestamps.shape[0] + 1) // 2

    def rectify_events(self, x: np.ndarray, y: np.ndarray, location: str):
        assert location in self.locations
        # From distorted to undistorted
        rectify_map = self.rectify_ev_maps[location]
        assert rectify_map.shape == (self.height, self.width, 2), rectify_map.shape
        assert x.max() < self.width
        assert y.max() < self.height
        return rectify_map[y, x]

    def __getitem__(self, index, get_flow=False):
        # _, ts_end = self.timestamps[index]
        # seg_index = np.argwhere(self.seg_timestamps == ts_end)[0][0]
        # # ts_start should be fine (within the window as we removed the first disparity map)
        # ts_start = ts_end - self.delta_t_us

        # if get_flow:
        #     flow_gt_path = Path(self.flow_gt_pathstrings[index])
        #     file_index = int(flow_gt_path.stem)
        #     flow_maps, valid_pixels = self.get_flow_map(flow_gt_path)
        #     # remove 40 bottom rows
        #     flow_maps = torch.from_numpy(flow_maps[:, :-40, :])     # 2 H W
        #     valid_pixels = torch.from_numpy(valid_pixels[:-40, :])  #   H W
        #     output['flow_gt'] = flow_maps
        #     output['flow_mask'] = valid_pixels
        #     output['file_index'] = file_index

        # seg_path = Path(self.seg_gt_pathstrings[seg_index])
        # seg_label = self.get_seg_label(seg_path)
        # seg_label = torch.from_numpy(seg_label).long()          #   H 
        
        seg_path = Path(self.seg_gt_pathstrings[index * 2])
        seg_label = self.get_seg_label(seg_path)
        seg_label = torch.from_numpy(seg_label).long()          #   H W

        ts_end = self.seg_timestamps[index * 2]
        ts_start = ts_end - self.delta_t_us
        output = {}
        for location in self.locations:
            event_data = self.event_slicers[location].get_events(ts_start, ts_end)

            p = event_data['p']
            t = event_data['t']
            x = event_data['x']
            y = event_data['y']

            xy_rect = self.rectify_events(x, y, location)
            x_rect = xy_rect[:, 0]
            y_rect = xy_rect[:, 1]

            event_representation = self.events_to_voxel_grid(x_rect, y_rect, p, t)

             # remove 40 bottom rows
            event_representation = event_representation[:, :-40, :] # T=10 H W

            # if self.augmentation:
            #     value_flip = round(random.random())
            #     if value_flip > 0.5:
            #         event_representation = torch.flip(event_representation, [2])
            #         seg_label = torch.flip(seg_label, [1])
            if 'representation' not in output:
                output['representation'] = dict()
            output['representation'][location] = event_representation
        
        # if self.augmentation:
        #     value_flip = round(random.random())
        #     if value_flip > 0.5:
        #         event_tensor = torch.flip(event_tensor, [2])    # C=T H W
        #         seg_label = torch.flip(seg_label, [1])          #     H W
        #         flow_maps = torch.flip(flow_maps, [2])          # C=2 H W
        #         valid_pixels = torch.flip(valid_pixels, [1])    #     H W

        output['seg_gt'] = seg_label
        
        return output
    
    def save_data(self, root, sequence):
        for index in tqdm(range(len(self))):
            output = self.__getitem__(index)
            filename = '{}_{}.npy'.format(sequence, str(index+1).zfill(4))
            np.save(os.path.join(root, 'seg_dataset_large', filename), output)
