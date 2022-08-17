# -*- coding: utf-8 -*-
# @Time         : 2022/7/22 20:15
# @Author       : Yufan Liu
# @Description  : Graph data for peptide binding sites

import pickle
from torch_geometric.data import Data, InMemoryDataset
from typing import AnyStr
import torch
import numpy as np
from tqdm import tqdm
import re
from scipy.spatial import distance




class ProteinGraph(InMemoryDataset):

    def __init__(self, data_name: AnyStr, root, dist_cutoff=8, transform=None, pre_trainsform=None):
        self.data_name = data_name
        self.dist_cutoff = dist_cutoff
        super(ProteinGraph, self).__init__(root, transform, pre_trainsform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return self.data_name + '.pkl'

    @property
    def processed_file_names(self):
        return self.data_name + '.pt'

    def process(self):
        total = []
        # data_dict[pdb_id]: sequence, target, coordinat
        data_dict = pickle.load(open(self.raw_paths[0], 'rb'))
        for pdb_id, contents in tqdm(data_dict.items()):
            node_feat, target, coordinates = contents[0], contents[1], contents[2]
            target = list(map(lambda x: int(x), list(target)))
            dist = distance.cdist(coordinates, coordinates)
            edge_ids = np.array(np.where(dist < self.dist_cutoff))
            graph_data = Data()

            graph_data.x = node_feat
            graph_data.y = torch.tensor(target, dtype=torch.long)
            graph_data.edge_index = torch.tensor(edge_ids, dtype=torch.long)

            total.append(graph_data)

        data, slices = self.collate(total)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == '__main__':
    graph_data = ProteinGraph(data_name='TE125', root='F:/protein_peptide/Dataset_pkl')
    print("")
