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
import yaml

from dataset.representations import VoxelGrid
from utils.eventslicer import EventSlicer

import ipdb
from tqdm import tqdm

from dataset.visualization import *

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
        remove_time_window=250
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

        # segmentation gt
        # load segmentation timestamps
        if self.mode == 'train' or self.mode=='seg_val':
            seg_dir = seg_path / 'semantic'
            assert seg_dir.is_dir()
            self.seg_timestamps = np.loadtxt(seg_dir / 'timestamps.txt', dtype='int64')

            # load segmentation paths
            ev_seg_dir = seg_dir / 'left' / '11classes'
            assert ev_seg_dir.is_dir()
            seg_gt_pathstrings = list()
            for entry in ev_seg_dir.iterdir():
                assert str(entry.name).endswith('.png')
                seg_gt_pathstrings.append(str(entry))
            seg_gt_pathstrings.sort()
            self.seg_gt_pathstrings = seg_gt_pathstrings

            self.seg_timestamps = self.seg_timestamps[(remove_time_window // 100 + 1) * 2:]
            del self.seg_gt_pathstrings[:(remove_time_window // 100 + 1) * 2]
            assert len(self.seg_gt_pathstrings) == self.seg_timestamps.shape[0]

        if self.mode == 'train' or self.mode=='depth_val':
            # cam gt
            cam_yaml = seg_path / 'calibration' / 'cam_to_cam.yaml'
            with open(cam_yaml,"r") as file:
                parameter=yaml.load(file.read(),Loader=yaml.Loader)
                mtx=parameter['disparity_to_depth']['cams_03']
                self.disp2depth_mtx=np.array(mtx)
            # disparity gt
            # load disparity timestamps
            disp_dir = seg_path / 'disparity'
            assert disp_dir.is_dir()
            self.disp_timestamps = np.loadtxt(disp_dir / 'timestamps.txt', dtype='int64')

            # load disparity paths
            ev_disp_dir = disp_dir / 'event'
            assert ev_disp_dir.is_dir()
            disp_gt_pathstrings = list()
            for entry in ev_disp_dir.iterdir():
                assert str(entry.name).endswith('.png')
                disp_gt_pathstrings.append(str(entry))
            disp_gt_pathstrings.sort()
            self.disp_gt_pathstrings = disp_gt_pathstrings

            self.disp_timestamps = self.disp_timestamps[(remove_time_window // 100 + 1):]
            del self.disp_gt_pathstrings[:(remove_time_window // 100 + 1)]
            assert len(self.disp_gt_pathstrings) == self.disp_timestamps.size
        
        if self.mode == 'train':
            assert (self.seg_timestamps.size + 1) // 2 == self.disp_timestamps.size

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
        if self.mode == 'train' or self.mode=='seg_val':
            return (self.seg_timestamps.size + 1) // 2
        elif self.mode=='depth_val':
            return (self.disp_timestamps.size)

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
        if self.mode == 'train':
            assert self.seg_timestamps[index * 2] == self.disp_timestamps[index]
        if self.mode == 'seg_val':
            ts_end = self.seg_timestamps[index * 2]
        if self.mode == 'depth_val':
            ts_end = self.disp_timestamps[index]
        ts_start = ts_end - self.delta_t_us

        output = dict()

        if self.mode == 'train' or self.mode == 'seg_val':
            seg_path = Path(self.seg_gt_pathstrings[index * 2])
            seg_label = self.get_seg_label(seg_path)
            seg_label = torch.from_numpy(seg_label).long()          #   H W
            output['seg_gt'] = seg_label                            #   H W torch.int64

        if self.mode == 'train' or self.mode == 'depth_val':
            disp_path = Path(self.disp_gt_pathstrings[index])
            file_index = int(disp_path.stem)
            disp_gt = self.get_disparity_map(disp_path)

            _3dImage = cv2.reprojectImageTo3D(disp_gt, self.disp2depth_mtx)
            depth_gt = _3dImage[:, :, 2]                            #  H W 3 → H W

            output['disp_gt'] = torch.from_numpy(disp_gt)           #   H W torch.unit8
            output['depth_gt'] = torch.from_numpy(depth_gt)         #   H W torch.float32
            output['file_index'] = file_index

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

            # # remove 40 bottom rows
            # event_representation = event_representation[:, :-40, :] # T=10 H W

            if 'representation' not in output:
                output['representation'] = dict()
            output['representation'][location] = event_representation   # T=10 H W torch.float32

        # # visualization
        # overlay = True
        # disp_img = disp_img_to_rgb_img(output['disp_gt'])       # H W 3 unit8
        # depth_img = output['depth_gt'].numpy()                  # H W unit8
        # seg_img = output['seg_gt'].numpy()                      # H W float32
        # ev_img = torch.sum(output['representation']['left'], axis=0).numpy()
        # ev_img = (ev_img/ev_img.max()*256).astype('uint8')      # H W int64

        # # disp_ev_overlay_img = get_disp_overlay(ev_img, disp_img, height=480, width=640)
        # # ev_rbg_img, depth_rgb_image, depth_ev_overlay_img = get_depth_overlay(ev_img, depth_img, height=480, width=640)
        # # ev_rbg_img, seg_rgb_image, seg_ev_overlay_img = get_seg_overlay(ev_img, depth_img, height=480, width=640)
        # depth_rgb_image, seg_rgb_image, overlay = get_depth_seg_overlay(depth_img, seg_img, height=480, width=640)


        # path = '/home/xiaoshan/work/datasets/ESS_DSEC/train/seg_depth_dataset/overlay/'
        # idx = str(index+1).zfill(4)

        # ev_path = path + 'ev_{}.png'.format(idx)
        # disp_path = path + 'disp_{}.png'.format(idx)
        # depth_path = path + 'depth_{}.png'.format(idx)
        # seg_path = path + 'seg_{}.png'.format(idx)

        # disp_ev_overlay_path = path + 'disp_ev_overlay_{}.png'.format(idx)
        # depth_ev_overlay_path = path + 'depth_ev_overlay_{}.png'.format(idx)
        # seg_ev_overlay_path = path + 'seg_ev_overlay_{}.png'.format(idx)
        # seg_depth_overlay_path = path + 'seg_depth_overlay_{}.png'.format(idx)

        # # cv2.imwrite(disp_path, disp_img)
        # # cv2.imwrite(ev_path, ev_img)
        # # cv2.imwrite(depth_path, depth_img)
        # # cv2.imwrite(seg_path, seg_img)

        # # cv2.imwrite(disp_ev_overlay_path, disp_ev_overlay_img)
        # cv2.imwrite(depth_path, depth_rgb_image)
        # cv2.imwrite(seg_path, seg_rgb_image)
        # cv2.imwrite(seg_depth_overlay_path, overlay)
        # exit(0)
        return output
    
    def save_data(self, root, sequence):
        for index in tqdm(range(len(self))):
            output = self.__getitem__(index)
            filename = '{}_{}.npy'.format(sequence, str(index+1).zfill(4))
            np.save(os.path.join(root, 'depth_dataset_z10az11a', filename), output)
