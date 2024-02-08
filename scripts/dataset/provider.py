from pathlib import Path

import torch

from dataset.sequence import Sequence

import numpy as np
import ipdb

class DatasetProvider:
    def __init__(self, flow_dataset_path: Path, seg_dataset_path: Path, delta_t_ms: int=50, num_bins=15):
        flow_train_path = flow_dataset_path / 'train'
        seg_train_path = seg_dataset_path / 'train'
        saved_path = flow_dataset_path / 'flow_seg'
        assert flow_dataset_path.is_dir(), str(flow_dataset_path)
        assert flow_train_path.is_dir(), str(flow_train_path)

        flow_seg_seqs = ['zurich_city_01_a', 'zurich_city_02_a',
                         'zurich_city_05_a', 'zurich_city_06_a', 
                         'zurich_city_07_a', 'zurich_city_08_a']
        
        train_sequences = list()

        for seq_name in flow_seg_seqs:
            # train_sequences.append(Sequence(child, 'train', delta_t_ms, num_bins))
            print("Start processing sequence: ", seq_name)
            seq = Sequence(flow_train_path / seq_name, seg_train_path / seq_name, 'train', delta_t_ms, num_bins)
            seq.save_data(saved_path, seq_name)
            # use file name child to save the data
            # np.save(saved_path, seq)
        # self.train_dataset = torch.utils.data.ConcatDataset(train_sequences)

    def get_train_dataset(self):
        return self.train_dataset

    def get_val_dataset(self):
        # Implement this according to your needs.
        raise NotImplementedError

    def get_test_dataset(self):
        # Implement this according to your needs.
        raise NotImplementedError
