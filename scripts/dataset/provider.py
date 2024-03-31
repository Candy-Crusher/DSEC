from pathlib import Path

import torch

from dataset.sequence import Sequence

import numpy as np
import ipdb

class DatasetProvider:
    def __init__(self, flow_dataset_path: Path, seg_dataset_path: Path, delta_t_ms: int=50, num_bins=15):
        # flow_train_path = flow_dataset_path / 'train'
        seg_train_path = seg_dataset_path / 'train'
        seg_test_path = seg_dataset_path / 'test'

        # assert flow_dataset_path.is_dir(), str(flow_dataset_path)
        # assert flow_train_path.is_dir(), str(flow_train_path)

        flow_seg_seqs = ['zurich_city_01_a', 'zurich_city_02_a',
                         'zurich_city_05_a', 'zurich_city_06_a', 
                         'zurich_city_07_a', 'zurich_city_08_a']
        
        seg_train_seqs = flow_seg_seqs + ['zurich_city_00_a', 'zurich_city_04_a']
        seg_test_seqs = ['zurich_city_13_a', 'zurich_city_14_c', 'zurich_city_15_a']
        disp_val_sequences = ['zurich_city_10_a', 'zurich_city_11_a']
        # train_sequences = list()

        # for seq_name in flow_seg_seqs:
        #     # train_sequences.append(Sequence(child, 'train', delta_t_ms, num_bins))
        #     print("Start processing sequence: ", seq_name)
        #     seq = Sequence(flow_train_path / seq_name, seg_train_path / seq_name, 'train', delta_t_ms, num_bins)
        #     seq.save_data(saved_path, seq_name)
        #     # use file name child to save the data
        #     # np.save(saved_path, seq)
        # # self.train_dataset = torch.utils.data.ConcatDataset(train_sequences)

        # for seq_name in seg_train_seqs:
        #     print("Start processing sequence: ", seq_name)
        #     seq = Sequence(None, seg_train_path / seq_name, 'train', delta_t_ms, num_bins)
        #     seq.save_data(seg_train_path, seq_name)

        # for seq_name in seg_test_seqs:
        #     print("Start processing sequence: ", seq_name)
        #     seq = Sequence(None, seg_test_path / seq_name, 'seg_val', delta_t_ms, num_bins)
        #     seq.save_data(seg_test_path, seq_name)

        for seq_name in disp_val_sequences:
            print("Start processing sequence: ", seq_name)
            seq = Sequence(None, seg_test_path / seq_name, 'depth_val', delta_t_ms, num_bins)
            seq.save_data(seg_test_path, seq_name)


    def get_train_dataset(self):
        return self.train_dataset

    def get_val_dataset(self):
        # Implement this according to your needs.
        raise NotImplementedError

    def get_test_dataset(self):
        # Implement this according to your needs.
        raise NotImplementedError
