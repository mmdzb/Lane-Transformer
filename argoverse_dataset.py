import os
import numpy as np
import pickle
import copy
import math
import pandas as pd
import torch

from tqdm import tqdm
from torch.utils.data import Dataset
from utils import rotate, densen, increase_or_decrease, merge_arrys, get_angle
from parser import get_training_parser
from argoverse.map_representation.map_api import ArgoverseMap

TIMESTAMP = 0
TRACK_ID = 1
OBJECT_TYPE = 2
X = 3
Y = 4
CITY_NAME = 5

angle_interval = 2
angle_points = 6

VECTOR_PRE_X = 0
VECTOR_PRE_Y = 1
VECTOR_X = 2
VECTOR_Y = 3

def collate_fn(batch):
    actor_input = []
    actor_input_length = []

    map_input = []
    map_input_length = []

    actor_polyline_input_length = []
    map_polyline_input_length = []

    GT = []
    GT_rel = []
    target = []
    target_rel = []

    cent_x = []
    cent_y = []
    angle = []

    file_names = []
    city_names = []

    batch_size = len(batch)

    for i in range(len(batch)):
        case = batch[i]

        file_names.append(case['file_name'])
        city_names.append(case['city_name'])

        GT.append(case['GT'])
        GT_rel.append(case['GT_rel'])
        target.append(case['target'])
        target_rel.append(case['target_rel'])

        cent_x.append(case['cent_x'])
        cent_y.append(case['cent_y'])
        angle.append(case['angle'])

        actor_input.append(case['actor_input_list'])
        actor_input_length.append(case['actor_input_lengths'])

        map_input.append(case['map_input_list'])
        map_input_length.append(case['map_input_lengths'])

        actor_polyline_input_length.append(case['actor_input_list'].shape[0])
        map_polyline_input_length.append(case['map_input_list'].shape[0])
    
    actor_input_length = np.hstack(actor_input_length)
    actor_polyline_num = actor_input_length.shape[0]
    actor_max_vec_num = np.max(actor_input_length)
    actor_vec_len = len(actor_input[0][0][0])

    actor_max_polyline_num = np.max(actor_polyline_input_length)
    actor_polyline_padding = np.zeros((batch_size, actor_max_polyline_num, 128))
    actor_polyline_mask = np.ones((batch_size, actor_max_polyline_num))

    actor_input_padding = np.zeros((actor_polyline_num, actor_max_vec_num, actor_vec_len))
    actor_input_mask = np.ones((batch_size * actor_max_polyline_num, actor_max_vec_num))   

    idx = 0
    idx_no_padding = 0
    for i, arrys in enumerate(actor_input):
        for j, arry in enumerate(arrys):
            actor_input_padding[idx_no_padding, :arry.shape[0]] = arry
            actor_input_mask[idx, :actor_input_length[idx_no_padding]].fill(0)
            idx += 1
            idx_no_padding += 1

        idx = (i+1) * actor_max_polyline_num


    map_input_length = np.hstack(map_input_length)
    map_polyline_num = map_input_length.shape[0]
    map_max_vec_num = np.max(map_input_length)
    map_vec_len = len(map_input[0][0][0])

    map_max_polyline_num = np.max(map_polyline_input_length)
    map_polyline_padding = np.zeros((batch_size, map_max_polyline_num, 128))
    map_polyline_mask = np.ones((batch_size, map_max_polyline_num))

    map_input_padding = np.zeros((map_polyline_num, map_max_vec_num, map_vec_len))
    map_input_mask = np.ones((batch_size * map_max_polyline_num, map_max_vec_num))

    idx = 0
    idx_no_padding = 0
    for i, arrys in enumerate(map_input):
        for j, arry in enumerate(arrys):
            map_input_padding[idx_no_padding, :arry.shape[0]] = arry
            map_input_mask[idx, :map_input_length[idx_no_padding]].fill(0)
            idx += 1
            idx_no_padding += 1

        idx = (i+1) * map_max_polyline_num

    global_graph_mask = np.ones((batch_size, actor_max_polyline_num))

    for i in range(batch_size):
        actor_polyline_mask[i, :actor_polyline_input_length[i]].fill(0)
        map_polyline_mask[i, :map_polyline_input_length[i]].fill(0)
        global_graph_mask[i, :actor_polyline_input_length[i]].fill(0)
    
    
    data = {}
    data['map_input_padding'] = torch.tensor(map_input_padding, dtype=torch.float32)
    data['map_input_mask'] = torch.tensor(map_input_mask, dtype=torch.float32)

    data['actor_input_padding'] = torch.tensor(actor_input_padding, dtype=torch.float32)
    data['actor_input_mask'] = torch.tensor(actor_input_mask, dtype=torch.float32)

    data['actor_polyline_padding'] = torch.tensor(actor_polyline_padding, dtype=torch.float32)
    data['actor_polyline_mask'] = torch.tensor(actor_polyline_mask, dtype=torch.float32)

    data['map_polyline_padding'] = torch.tensor(map_polyline_padding, dtype=torch.float32)
    data['map_polyline_mask'] = torch.tensor(map_polyline_mask, dtype=torch.float32)

    data['actor_polyline_input_length'] = actor_polyline_input_length
    data['map_polyline_input_length'] = map_polyline_input_length

    data['global_graph_mask'] = torch.tensor(global_graph_mask, dtype=torch.float32)

    data['actor_max_polyline_num'], data['actor_max_vec_num'], data['actor_vec_len'] = actor_max_polyline_num, actor_max_vec_num, actor_vec_len
    data['map_max_polyline_num'], data['map_max_vec_num'], data['map_vec_len'] = map_max_polyline_num, map_max_vec_num, map_vec_len

    actor_total_input_padding = torch.zeros((batch_size, actor_max_polyline_num, actor_max_vec_num, actor_vec_len), dtype=torch.float32)
    map_total_input_padding = torch.zeros((batch_size, map_max_polyline_num, map_max_vec_num, map_vec_len), dtype=torch.float32)
    actor_idx = 0
    map_idx = 0
    for i in range(batch_size):
        actor_total_input_padding[i, :actor_polyline_input_length[i]] = torch.clone(data['actor_input_padding'][actor_idx: actor_idx+actor_polyline_input_length[i]])
        # actor_idx += actor_polyline_length[i]
        actor_idx = actor_idx + actor_polyline_input_length[i]

        map_total_input_padding[i, :map_polyline_input_length[i]] = torch.clone(data['map_input_padding'][map_idx: map_idx+map_polyline_input_length[i]])
        # map_idx += map_polyline_length[i]
        map_idx = map_idx + map_polyline_input_length[i]
    data['actor_total_input_padding'] = actor_total_input_padding
    data['map_total_input_padding'] = map_total_input_padding

    valid_actor_polyline = 1 - actor_polyline_mask
    valid_actor_polyline = np.array(valid_actor_polyline, dtype=bool)
    data['valid_actor_polyline'] = valid_actor_polyline.flatten()

    data['GT'] = GT
    data['GT_rel'] = GT_rel
    data['target'] = target
    data['target_rel'] = target_rel

    data['cent_x'] = cent_x
    data['cent_y'] = cent_y
    data['angle'] = angle

    data['file_names'] = file_names
    data['city_names'] = city_names

    # data['actor_input_mask'] = data['actor_input_mask'].bool()
    # data['map_input_mask'] = data['map_input_mask'].bool()
    # data['actor_polyline_mask'] = data['actor_polyline_mask'].bool()
    # data['map_polyline_mask'] = data['map_polyline_mask'].bool()
    # data['global_graph_mask'] = data['global_graph_mask'].bool()

    return data

def preprocess(args, lines, file):
    global max_total_vec_num
    max_total_vec_num = 0
    total_vec_num = 0
    data_for_id = {}
    data_dict = {}
    data_dict['file_name'] = file
    data_dict['city_name'] = lines[0].strip().split(',')[CITY_NAME]
    data_dict['start_time'] = float(lines[0].strip().split(',')[TIMESTAMP])

    for i, line in enumerate(lines):
        line = line.strip().split(',')
        line[X], line[Y], line[TIMESTAMP] = float(line[X]), float(line[Y]), float(line[TIMESTAMP])-data_dict['start_time']
        id = line[TRACK_ID]

        if line[OBJECT_TYPE] == 'AV' or line[OBJECT_TYPE] == 'AGENT':
            line[TRACK_ID] = line[OBJECT_TYPE]
        
        if line[TRACK_ID] in data_for_id:
            data_for_id[line[TRACK_ID]].append(line)
            total_vec_num += 1
        else:
            data_for_id[line[TRACK_ID]] = [line]

        if line[OBJECT_TYPE] == 'AGENT' and len(data_for_id['AGENT']) == 20:
            data_dict['cent_x'] = line[X]
            data_dict['cent_y'] = line[Y]
            data_dict['agent_pred_index'] = 20
            data_dict['two_seconds'] = line[TIMESTAMP]

            angles = []
            agent_lines = data_for_id['AGENT']
            for j in range(angle_points):
                if j + angle_interval < angle_points:
                    dis_x, dis_y = agent_lines[angle_interval+j-angle_points][X] - agent_lines[j-angle_points][X], agent_lines[angle_interval+j-angle_points][Y] - agent_lines[j-angle_points][Y]
                    angles.append([dis_x, dis_y])
            
    if total_vec_num > max_total_vec_num:
        max_total_vec_num = total_vec_num

    target = []
    GT = []
    for line in data_for_id['AGENT']:
        if line[TIMESTAMP] > data_dict['two_seconds']:
            GT.append([line[X], line[Y]])
        else:
            target.append([line[X], line[Y]])
    
    data_dict['target'] = [np.array(target)]
    data_dict['GT'] = [np.array(GT)]

    angles = np.array(angles)
    dis_x, dis_y = np.mean(angles, axis=0)
    angle = -get_angle(dis_x, dis_y) + math.radians(90)
    data_dict['angle'] = angle

    for id in data_for_id:
        id_data = data_for_id[id]
        for line in id_data:
            line[X], line[Y] = rotate(line[X] - data_dict['cent_x'], line[Y] - data_dict['cent_y'], angle)

    ids = list(data_for_id.keys())
    if ids.index('AGENT') != 0:
        ids[ids.index('AGENT')] = ids[0]
        ids[0] = 'AGENT'
    if ids.index('AV') != 1:
        ids[ids.index('AV')] = ids[1]
        ids[1] = 'AV'

    vectors = []
    vectors_indexs = []
    end_time = data_dict['two_seconds']
    data_dict['trajs'] = []
    data_dict['agents'] = []

    ## 运动物体的向量
    for i, id in enumerate(ids):
        start = len(vectors)
        id_data = data_for_id[id]


        for j in range(len(id_data)):
            point = id_data[j]

            if point[TIMESTAMP] > end_time:
                break

            if j > 0:
                now_X = point[3]
                now_Y = point[4]
                pre_X = point_pre[3]
                pre_Y = point_pre[4]
                vector = [now_X, now_Y, pre_X, pre_Y, now_X-pre_X, now_Y-pre_Y, j, i] # 改动
                vectors.append(vector)
            point_pre = point
        
        end = len(vectors)
        if end - start != 0:
            vectors_indexs.append([start, end])
    
    data_dict['map_start_polyline_index'] = len(vectors_indexs)

    ## 地图的获取
    lane_ids = argo_map.get_lane_ids_in_xy_bbox(data_dict['cent_x'], data_dict['cent_y'], city_name=data_dict['city_name'], query_search_range_manhattan=50.)
    semantic_lane_centerlines = [argo_map.get_lane_segment_centerline(lane_id, data_dict['city_name']) for lane_id in lane_ids]
    lane_centerlines = [semantic_lane_centerline[:, :2].copy() for semantic_lane_centerline in semantic_lane_centerlines]

    for lane_centerline in lane_centerlines:
        for point in lane_centerline:
            point[0], point[1] = rotate(point[0] - data_dict['cent_x'], point[1] - data_dict['cent_y'], angle)
    data_dict['polygons'] = lane_centerlines

    for i, lane_centerline in enumerate(lane_centerlines):
        start = len(vectors)
        
        lane_id = lane_ids[i]

        for j, point in enumerate(lane_centerline):
            if j>0:
                now_X = point[0]
                now_Y = point[1]
                pre_X = point_pre[0]
                pre_Y = point_pre[1]
                vector = [now_X, now_Y, pre_X, pre_Y, now_X-pre_X, now_Y-pre_Y, j, i+data_dict['map_start_polyline_index']] # 改动
                vectors.append(vector)

            point_pre = point
        
        end = len(vectors)
        vectors_indexs.append([start, end])

    matrix = np.array(vectors)

    ## 获取GT
    target_rel = []
    GT_rel = []
    for line in data_for_id['AGENT']:
        if line[TIMESTAMP] > data_dict['two_seconds']:
            GT_rel.append([line[X], line[Y]])
        else:
            target_rel.append([line[X], line[Y]])
    
    data_dict['target_rel'] = [np.array(target_rel)]
    data_dict['GT_rel'] = [np.array(GT_rel)]
    
    data_dict['vectors'] = matrix
    data_dict['vectors_indexs'] = vectors_indexs

    return data_dict


class Argoverse_Dataset(Dataset):
    def __init__(self, args, datas_dir) -> None:
        super().__init__()
        self.data = []
        self.files_list = []
        self.args = args
        self.data_path = ''
        self.datas_dir = datas_dir   
        self.type = args.pkl_save_dir.split('/')[-1]
        

        resume_dir = args.pkl_save_dir

        if args.resume and os.path.exists(resume_dir):
            
            files = []
            _, _, file_names = os.walk(resume_dir).__next__()
            files.extend([os.path.join(resume_dir, file_name) for file_name in file_names if file_name.endswith("pkl") and not file_name.startswith('.')])
            # files = files[:2000]
            files = np.sort(files)


            self.files_list.append(files)

            with tqdm(total=len(files)) as t:
                for file in files:    
                    self.data.append(file)
                    t.update(1)

        else:
            global argo_map
            argo_map = ArgoverseMap()

            if not os.path.exists(resume_dir):
                os.makedirs(resume_dir)

            cases = []
            # _, _, file_names = os.walk(self.data_path).__next__()
            cases = [case for case in os.listdir(self.datas_dir)]
            cases.sort()
            # cases = cases[:20000]
            # files.extend([os.path.join(self.data_path, file_name) for file_name in file_names if file_name.endswith("pkl")])

            with tqdm(total=(len(cases))) as t:
                for index, case in enumerate(cases):
                    case_path = os.path.join(self.datas_dir, case)

                    with open(case_path, "r", encoding='utf-8') as f:
                        case_lines = f.readlines()[1:]
                    
                    case_data_temp = preprocess(args, case_lines, case)
                    if args.resume:
                        name = f"case{index}.pkl"
                        self.data.append(os.path.join(resume_dir, name))
                        with open(os.path.join(resume_dir, name), "wb") as f:
                            pickle.dump(case_data_temp, f, protocol=pickle.HIGHEST_PROTOCOL)
                        f.close()
                    t.update(1)

    def __len__(self) -> int:
        if self.args.data_augment == 0 or self.type == 'val':
            return len(self.data)
        else:
            return 2 * len(self.data)
        # return 1000
    
    def __getitem__(self, index: int):
        if self.args.data_augment and self.type == 'train':
            aug_index = index
            index = index // 2

        file = self.data[index]
        temp = open(file, 'rb')
        data = pickle.load(temp)
        temp.close()

        if self.args.data_augment and self.type == 'train':
            if aug_index % 2 == 0:
                data['cent_x'] = -data['cent_x'] 
                data['target'][0][:, 0] = -data['target'][0][:, 0]
                data['target_rel'][0][:, 0] = -data['target_rel'][0][:, 0]
                data['GT'][0][:, 0] = -data['GT'][0][:, 0]
                data['GT_rel'][0][:, 0] = -data['GT_rel'][0][:, 0]

                data['vectors'] = data['vectors'] * [-1, 1, -1, 1, -1, 1, 1, 1]

        actor_input_list = []
        map_input_list = []
        map_start_polyline_idx = data['map_start_polyline_index']
        for j, polyline_span in enumerate(data['vectors_indexs']):
            tensor = data['vectors'][polyline_span[0]:polyline_span[1]]
            if j >= map_start_polyline_idx:
                map_input_list.append(tensor)
            else:
                actor_input_list.append(tensor)
        
        data['map_input_list'], data['map_input_lengths'] = merge_arrys(map_input_list)
        data['actor_input_list'], data['actor_input_lengths'] = merge_arrys(actor_input_list)
              
        return data


if __name__ == '__main__':
    args = get_training_parser().parse_args()

    train_path = '/home/wzb/Datasets/Argoverse/train/data/'
    val_path = '/home/wzb/Datasets/Argoverse/val/data/'
    test_path = '/workspace1/wzb/Datasets/Argoverse/test_obs/data/'


    args.pkl_save_dir = os.path.join(args.pkl_save_dir, 'train')
    Train_Dataset = Argoverse_Dataset(args, train_path)
    args.pkl_save_dir = "/".join(args.pkl_save_dir.split('/')[:-1])

    args.pkl_save_dir = os.path.join(args.pkl_save_dir, 'val')
    Val_Dataset = Argoverse_Dataset(args, val_path)
    args.pkl_save_dir = "/".join(args.pkl_save_dir.split('/')[:-1])

    args.pkl_save_dir = os.path.join(args.pkl_save_dir, 'test')
    Test_Dataset = Argoverse_Dataset(args, test_path)
    args.pkl_save_dir = "/".join(args.pkl_save_dir.split('/')[:-1])

