import os, sys, h5py
import numpy as np
import torch
import torch.utils.data as data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

class shapenet4096(data.Dataset):
    def __init__(self, phase, data_root, data_type, if_4096):
        super().__init__()
        self.folder = data_type + '/'
        if phase == 'train':
            self.data_list_file = data_root + data_type + '_train.npy'
        else:
            self.data_list_file = data_root + data_type + '_test.npy'
        self.data_dir = data_root + self.folder
        self.data_list = np.load(self.data_list_file)
        
    def __getitem__(self, idx):
        cur_name = self.data_list[idx].split('.')[0]
        cur_data = torch.from_numpy(np.load(self.data_dir + self.data_list[idx])).float()
        cur_points = cur_data[:,0:3]
        cur_normals = cur_data[:,3:]
        cur_points_num = 4096
        cur_values = -1
        return cur_points, cur_normals, cur_points_num, cur_values, cur_name
        
    def __len__(self):
        return self.data_list.shape[0]

class loadnp(data.Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.data_list_file = data_dir
        self.data_list = np.load(self.data_list_file)
        print(type(self.data_list))

class loadobj(data.Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.data_list_file = data_dir
        data_d=list()
        v_num = 0
        vn_num = 0
        file_object = open(self.data_list_file, 'r')
        lines = file_object.readlines()
        for line in lines:
            if line:
                line = line.rstrip('\n')
                strs = line.split(' ')
                if strs[0] == 'v':
                    v_d = [float(strs[1]),float(strs[2]),float(strs[3]),0,0,0]
                    print(v_d)
                    data_d.append(v_d)
                    v_num+=1

                if strs[0] == 'vn' and vn_num <v_num:
                    data_d[vn_num][3]=float(strs[1])
                    data_d[vn_num][4]=float(strs[2])
                    data_d[vn_num][5]=float(strs[3])
                    vn_num+=1
        print(data_d)
        print(type(data_d))
