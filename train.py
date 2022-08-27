from base64 import b16encode
from numpy import float32
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from tqdm import tqdm

from parser import get_training_parser
from utils import *

from argoverse_dataset import Argoverse_Dataset, collate_fn
from model.LaneTransformer import LaneTransformer
from matplotlib import pyplot as plt
from argoverse.evaluation import eval_forecasting
from argoverse.evaluation import competition_util


import os
import gc
import copy
import torch
import time
import pickle
import time
import csv

global hidden_size
train_path = '/home/wzb/Datasets/Argoverse/train/data/'
val_path = '/home/wzb/Datasets/Argoverse/val/data/'
test_path = '/home/wzb/Datasets/Argoverse/test_obs/data/'

os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
os.environ['CUDA_VISIBLE_DEVICE']='1'



def train_one_epoch(args, epoch, model, optimizer, loader, train_writer):
    for j, data in loader:

        output, score = model(data, device)
        loss = my_criterion(args, output, score, data, device)

        train_writer.add_scalar('Loss', loss.item(), j)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if j % 100 == 0:
            now = time.localtime()
            nowt = time.strftime("%Y-%m-%d-%H_%M_%S", now)

            log_dir = './log/' + args.lab_name
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            with open(log_dir + '/train.txt', 'a') as f:   
                f.write(f"{nowt}  Epoch: {epoch}, Training Step: {j}, Loss: {loss.item()}, Learning Rate: {optimizer.state_dict()['param_groups'][0]['lr']}\n")
                f.close()
        

def my_criterion(args, outputs, score, data, device):
    if True:
        pred_fut_trajs = []
        for output in outputs:
            pred_fut_traj = output
            pred_fut_trajs.append(pred_fut_traj.unsqueeze(0))
        pred_fut_trajs = torch.vstack(pred_fut_trajs)
        pred_fut_trajs = pred_fut_trajs.permute(1, 0, 2, 3)

        fut_traj = np.array(data['GT_rel'])[:, 0, :, :2]
        fut_traj = torch.tensor(fut_traj, dtype=torch.float32).to(device)

    loss = torch.zeros(1, dtype=torch.float32).to(device)

    fut_trajs = fut_traj.unsqueeze(1).repeat(1, pred_fut_trajs.shape[1], 1, 1)
    fde_k = torch.sqrt((pred_fut_trajs[:, :, -1, 0] - fut_trajs[:, :, -1, 0]) ** 2 + (pred_fut_trajs[:, :, -1, 1] - fut_trajs[:, :, -1, 1]) ** 2)
    k_hat = torch.argmin(fde_k, dim=1)
    index = torch.tensor(range(pred_fut_trajs.shape[0])).to(device)
    pred_fut_traj = pred_fut_trajs[index, k_hat]

    M = pred_fut_traj.shape[0]
    T = pred_fut_traj.shape[1]

    mse_loss = F.mse_loss(pred_fut_traj, fut_traj, reduction='none')
    mse_loss = mse_loss.sum(dim=2)
    mse_loss = torch.sqrt(mse_loss)
    mse_loss = mse_loss.mean(dim=1)

    fde_loss = fde_k[index, k_hat]

    K = 6
    score_hat = score[index, k_hat].unsqueeze(-1)
    score_hat = score_hat.repeat(1, K)
    cls_loss = score + 0.2 - score_hat
    cls_loss[cls_loss < 0] = 0
    cls_loss = cls_loss.sum(dim=-1).sum(dim=-1)
    cls_loss = cls_loss /((K-1) * M)

    loss = mse_loss * 0.5 + fde_loss * 0.5
    loss = loss.mean()
    loss = loss + cls_loss

    return loss

def validation(args, model, loader, device):
    model.eval()
    with torch.no_grad():
        minADE = 10000
        minFDE = 10000
        sum_true = 0
        sum_data = 0
        
        minADE = torch.tensor(minADE, device=device)
        minFDE = torch.tensor(minFDE, device=device)
        sum_true = torch.tensor(sum_true, device=device)
        sum_data = torch.tensor(sum_data, device=device)

        minADE_k1 = 10000
        minFDE_k1 = 10000
        sum_true_k1 = 0
        sum_data_k1 = 0
        
        minADE_k1 = torch.tensor(minADE_k1, device=device)
        minFDE_k1 = torch.tensor(minFDE_k1, device=device)
        sum_true_k1 = torch.tensor(sum_true_k1, device=device)
        sum_data_k1 = torch.tensor(sum_data_k1, device=device)

        ADE = []
        FDE = []

        FDE_k1 = []
        ADE_k1 = []

        flag = 1
        for j, data in loader:

            outputs, _ = model(data, device)
            obs_traj = np.array(data['target'])[:, 0, :, :2]

            origin_point = np.zeros([len(outputs[0]), 2])
            origin_angle = np.zeros([len(outputs[0])])
            for i in range(len(outputs[0])):
                origin_point[i][0], origin_point[i][1] = rotate(0 - data['cent_x'][i], 0 - data['cent_y'][i], data['angle'][i])
                origin_angle[i] = -data['angle'][i] 

            pred_fut_trajs = []
            pred_fut_trajs_rel = []
            for idx, output in enumerate(outputs):                                       
                pred_fut_traj = copy.deepcopy(output.cpu().detach().numpy())
                pred_fut_traj_rel = copy.deepcopy(output.cpu().detach().numpy())
                for batch_id in range(len(pred_fut_traj)):
                    to_origin_coordinate(pred_fut_traj[batch_id], batch_id, origin_point, origin_angle)
                pred_fut_trajs.append(pred_fut_traj)
                pred_fut_trajs_rel.append(pred_fut_traj_rel)

            fut_traj = np.array(data['GT'])[:, 0, :, :2]

            fut_traj_rel = np.array(data['GT_rel'])[:, 0, :, :2]
            obs_traj_rel = np.array(data['target_rel'])[:, 0, :, :2]

            pred_fut_trajs = np.array(pred_fut_trajs)
            pred_fut_trajs = pred_fut_trajs.transpose((1,0,2,3))
            for idx, pred_fut_traj in enumerate(pred_fut_trajs):
                temp_list_FDE = []
                temp_list_ADE = []

                for k in range(len(outputs)):
                    temp_list_FDE.append(np.sqrt((pred_fut_traj[k][-1][0] - fut_traj[idx][-1][0]) ** 2 + (pred_fut_traj[k][-1][1] - fut_traj[idx][-1][1]) ** 2))
                    DisErr = 0.
                    for l in range(30):
                        DisErr += np.sqrt((pred_fut_traj[k][l][0] - fut_traj[idx][l][0]) ** 2 + (pred_fut_traj[k][l][1] - fut_traj[idx][l][1]) ** 2)
                    temp_list_ADE.append(DisErr / 30.)

                min_temp_list_FDE = np.min(temp_list_FDE)
                min_temp_list_ADE = temp_list_ADE[np.argmin(temp_list_FDE)]
                
                FDE.append(min_temp_list_FDE)
                ADE.append(min_temp_list_ADE)

                temp_fde_k1 = np.sqrt((pred_fut_traj[0][-1][0] - fut_traj[idx][-1][0]) ** 2 + (pred_fut_traj[0][-1][1] - fut_traj[idx][-1][1]) ** 2)
                FDE_k1.append(temp_fde_k1)
                de = 0.
                for m in range(30):
                    de += np.sqrt((pred_fut_traj[0][m][0] - fut_traj[idx][m][0]) ** 2 + (pred_fut_traj[0][m][1] - fut_traj[idx][m][1]) ** 2)
                ADE_k1.append(de / 30.)

                if min_temp_list_FDE < 2.0:
                    sum_true += 1
                sum_data += 1  

                if temp_fde_k1 < 2.0:
                    sum_true_k1 += 1

            if flag == 1:
                for idx, pred_fut_traj in enumerate(pred_fut_trajs):
                    plt.figure()
                    plt.cla()

                    if False:
                        for centerline in centerlines[idx]:
                            plt.plot(centerline[:, 0], centerline[:, 1], color='orange')
                    
                    root = '/home/wzb/Datasets/Argoverse/val/data'
                    
                    if True:
                        min = 10000.
                        min_id = 0
                        for k in range(len(outputs)):
                            if min > np.sqrt((pred_fut_traj[k][-1][0] - fut_traj[idx][-1][0]) ** 2 + (pred_fut_traj[k][-1][1] - fut_traj[idx][-1][1]) ** 2):
                                min = np.sqrt((pred_fut_traj[k][-1][0] - fut_traj[idx][-1][0]) ** 2 + (pred_fut_traj[k][-1][1] - fut_traj[idx][-1][1]) ** 2)
                                min_id = k
                        plt.plot(pred_fut_traj[min_id, :, 0], pred_fut_traj[min_id, :, 1], color='red')
                        plt.scatter(pred_fut_traj[min_id, -1, 0], pred_fut_traj[min_id, -1, 1], color='red', marker='o')

                        for k in range(len(outputs)):
                            if k == min_id:
                                continue
                            plt.plot(pred_fut_traj[k, :, 0], pred_fut_traj[k, :, 1], color='purple', alpha=0.7)
                            plt.scatter(pred_fut_traj[k, -1, 0], pred_fut_traj[k, -1, 1], color='purple', marker='+')

                        plt.plot(obs_traj[idx, :, 0], obs_traj[idx, :, 1], color='orange')
                        plt.plot(fut_traj[idx, :, 0], fut_traj[idx, :, 1], color='blue')
                    
                    save_dir = './pic/' + args.lab_name
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    plt.savefig(f'./pic/{args.lab_name}/case{idx}_{min<2}.jpg') 
                    plt.close()
                flag = 0        

        FDE_sum = torch.tensor(np.sum(FDE), device=device)
        ADE_sum = torch.tensor(np.sum(ADE), device=device)

        FDE_sum_k1 = torch.tensor(np.sum(FDE_k1), device=device)
        ADE_sum_k1 = torch.tensor(np.sum(ADE_k1), device=device)

    if args.distributed_training:
        dist.barrier()
        min_FDE = reduce_sum(FDE_sum)
        min_ADE = reduce_sum(ADE_sum)
        a = reduce_sum(sum_true)
        b = reduce_sum(sum_data)

        min_FDE_k1 = reduce_sum(FDE_sum_k1)
        min_ADE_k1 = reduce_sum(ADE_sum_k1)
        a_k1 = reduce_sum(sum_true_k1)
    else:
        min_FDE = FDE_sum
        min_ADE = ADE_sum
        a = sum_true
        b = sum_data

        min_FDE_k1 = FDE_sum_k1
        min_ADE_k1 = ADE_sum_k1
        a_k1 = sum_true_k1

    a = a.cpu().detach().numpy()
    b = b.cpu().detach().numpy()
    min_FDE = min_FDE.cpu().detach().numpy() / b
    min_ADE = min_ADE.cpu().detach().numpy() / b
    miss_rate =  1 - a / b 

    a_k1 = a_k1.cpu().detach().numpy()
    min_FDE_k1 = min_FDE_k1.cpu().detach().numpy() / b
    min_ADE_k1 = min_ADE_k1.cpu().detach().numpy() / b
    miss_rate_k1 =  1 - a_k1 / b 

    if args.distributed_training: 
        print(f"rank # : {dist.get_rank()}, MR(k=6): {miss_rate}, min ADE(k=6): {min_ADE}, min FDE(k=6): {min_FDE}, sum true: {a}, sum data: {b}")
    else:
        print(f"MR(k=6): {miss_rate}, min ADE(k=6): {min_ADE}, min FDE(k=6): {min_FDE}, sum true(k=6): {a}, sum data: {b}, MR(k=1): {miss_rate_k1}, min ADE(k=1): {min_ADE_k1}, min FDE(k=1): {min_FDE_k1}, sum true(k=1): {a_k1},")
    return miss_rate, min_ADE, min_FDE

def test(args, model, loader, device):
    final_result = []
    file_name = []
    model.eval()
    with torch.no_grad():
        flag = 1
        for j, data in loader:
            
            file_name.append(data['file_names'])

            outputs, _ = model(data, device)

            obs_traj = np.array(data['target'])[:, 0, :, :2]

            origin_point = np.zeros([len(outputs[0]), 2])
            origin_angle = np.zeros([len(outputs[0])])
            for i in range(len(outputs[0])):
                origin_point[i][0], origin_point[i][1] = rotate(0 - data['cent_x'][i], 0 - data['cent_y'][i], data['angle'][i])
                origin_angle[i] = -data['angle'][i] 

            pred_fut_trajs = []
            pred_fut_trajs_rel = []
            for idx, output in enumerate(outputs):                                       
                pred_fut_traj = copy.deepcopy(output.cpu().detach().numpy())
                pred_fut_traj_rel = copy.deepcopy(output.cpu().detach().numpy())
                for batch_id in range(len(pred_fut_traj)):
                    if args.data_type == 'lane_gcn':
                        pred_fut_traj[batch_id] = lane_gcn_to_origin_coordinate(pred_fut_traj, data, batch_id)
                    else:
                        to_origin_coordinate(pred_fut_traj[batch_id], batch_id, origin_point, origin_angle)
                pred_fut_trajs.append(pred_fut_traj)
                pred_fut_trajs_rel.append(pred_fut_traj_rel)

            pred_fut_trajs = np.array(pred_fut_trajs)
            pred_fut_trajs = pred_fut_trajs.transpose((1,0,2,3))

            final_result.append(pred_fut_trajs)

            if flag == 1:
                for idx, pred_fut_traj in enumerate(pred_fut_trajs):
                    plt.figure()
                    plt.cla()

                    if False:
                        for centerline in centerlines[idx]:
                            plt.plot(centerline[:, 0], centerline[:, 1], color='orange')
                    
                    if True:
                        min = 10000.
                        min_id = 0
                        plt.scatter(pred_fut_traj[min_id, :, 0], pred_fut_traj[min_id, :, 1], color='red')
                        plt.plot(obs_traj[idx, :, 0], obs_traj[idx, :, 1], color='blue')
                    
                    save_dir = './pic/' + args.lab_name
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    plt.savefig(f'./pic/{args.lab_name}/case{idx}_{min<2}.jpg') 
                    plt.close()
                flag = 0        

    final_result = np.concatenate(final_result)
    file_name = np.concatenate(file_name)

    result = {}
    for i in range(len(final_result)):
        key = int(file_name[i].split('.')[0])
        result[key] = final_result[i]
    return result


def main(local_rank, args):
    args.local_rank = local_rank

    set_seed_globally(args.seed)

    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.distributed_training:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:23456', world_size=args.nprocs, rank=local_rank)
        device = torch.device('cuda', args.local_rank)

    if args.distributed_training == 0 or dist.get_rank() == 0:
        print("1. Initializing DataSet")
    

    if args.distributed_training == 0 or dist.get_rank() == 0:
        print("\n(1) Loading Training Dataset:\n")

    if args.mode != 'val':
        args.pkl_save_dir = os.path.join(args.pkl_save_dir, 'train')
        train_dataset = Argoverse_Dataset(args, train_path)
        args.pkl_save_dir = "/".join(args.pkl_save_dir.split('/')[:-1])

        if args.distributed_training:
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size // dist.get_world_size(), collate_fn=collate_fn, num_workers=8,sampler=train_sampler)
        else:
            train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=args.loader_num_workers,collate_fn=collate_fn,pin_memory=False,drop_last=False)
    
    if args.distributed_training == 0 or dist.get_rank() == 0:
        print("\n(2) Loading Validation Dataset:\n")
    
    args.pkl_save_dir = os.path.join(args.pkl_save_dir, 'val')
    val_dataset = Argoverse_Dataset(args, val_path)
    args.pkl_save_dir = "/".join(args.pkl_save_dir.split('/')[:-1])

    if args.distributed_training:
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size // dist.get_world_size(), collate_fn=collate_fn, num_workers=8,sampler=val_sampler)
    else: 
        val_dataloader = DataLoader(val_dataset,batch_size=args.batch_size,shuffle=False,num_workers=args.loader_num_workers,collate_fn=collate_fn,pin_memory=False,drop_last=False)
    
    if args.distributed_training == 0 or dist.get_rank() == 0:
        print("\n(3) Loading Test Dataset:\n")
    
    args.pkl_save_dir = os.path.join(args.pkl_save_dir, 'test')
    test_dataset = Argoverse_Dataset(args, test_path)
    args.pkl_save_dir = "/".join(args.pkl_save_dir.split('/')[:-1])

    if args.distributed_training:
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size // dist.get_world_size(), collate_fn=collate_fn, num_workers=8,sampler=val_sampler)
    else: 
        test_dataloader = DataLoader(test_dataset,batch_size=args.batch_size,shuffle=False,num_workers=args.loader_num_workers,collate_fn=collate_fn,pin_memory=False,drop_last=False)

    model = LaneTransformer(args).to(device)

    if args.distributed_training:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)

    if args.mode == "train":
        log_dir = './log/' + args.lab_name
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        train_writer = SummaryWriter(log_dir=log_dir)
        val_writer = SummaryWriter(log_dir=log_dir)

        if args.distributed_training == 0 or dist.get_rank() == 0:
            print("2. Training and Validation")

        if args.continue_training:
            load_checkpoint('./saved/#load_model#', model, optimizer)

        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Total: {total_num}, Trainable: {trainable_num}')

        best_MR = 10.
        for epoch in range(args.num_epochs):
            
            if epoch < args.warm_up_epochs:
                warm_up_ratio = (epoch + 1) / args.warm_up_epochs
                for p in optimizer.param_groups:
                    p['lr'] = args.lr * warm_up_ratio
            else:
                if epoch % 10 == 0 and epoch != 10:
                    for p in optimizer.param_groups:
                        p['lr'] *= 0.5

            if args.distributed_training == 0 or dist.get_rank() == 0:
                print(f'\n# Epoch {epoch}:')
                print("\nTraining: ")
            model.train()

            if args.distributed_training:
                train_dataloader.sampler.set_epoch(epoch)

            train_loader = enumerate(train_dataloader)
            if args.distributed_training == 0 or dist.get_rank() == 0:
                train_loader = tqdm(train_loader, total=len(train_dataloader))

            train_one_epoch(args, epoch, model, optimizer, train_loader, train_writer)

            if args.distributed_training == 0 or dist.get_rank() == 0:
                print(f"\n# After epoch {epoch}:")
                print("\nValidating: ")
            
            val_loader = enumerate(val_dataloader)
            if args.distributed_training == 0 or dist.get_rank() == 0:
                val_loader = tqdm(val_loader, total=len(val_dataloader))

            mr, ade, fde = validation(args, model, val_loader, device)

            val_writer.add_scalar('MR', mr, epoch)         

            if mr < best_MR:
                best_MR = mr
                if args.distributed_training == 0 or dist.get_rank() == 0:
                    now = time.localtime()
                    date = time.strftime("%Y-%m-%d-%H_%M_%S", now)

                    save_dir = './saved/' + args.lab_name
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    if args.distributed_training == 0:
                        save_checkpoint(save_dir, model, optimizer, epoch, date, best_MR, args.lab_name)   
                    else:
                        save_checkpoint(save_dir, model, optimizer, epoch, date, best_MR, args.lab_name)   
                
                
            if args.distributed_training == 0 or dist.get_rank() == 0:
                now = time.localtime()
                nowt = time.strftime("%Y-%m-%d-%H_%M_%S", now)
                log_dir = './log/' + args.lab_name
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                with open(log_dir + '/val.txt', 'a') as f:   
                    f.write(f"{nowt} Validation Dataset: Epoch: {epoch}, MR: {mr}, min ADE: {ade}, min FDE: {fde}\n")
                    f.close()
    elif args.mode == "val":
        load_checkpoint('./saved/#load_model#', model, optimizer)
        print("\nValidating: ")
        val_loader = enumerate(val_dataloader)
        if args.distributed_training == 0 or dist.get_rank() == 0:
            val_loader = tqdm(val_loader, total=len(val_dataloader))

        mr, ade, fde = validation(args, model, val_loader, device)
        if args.distributed_training == 0 or dist.get_rank() == 0:
            now = time.localtime()
            nowt = time.strftime("%Y-%m-%d-%H_%M_%S", now)

            log_dir = './log/' + args.lab_name + '_val'
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            with open(log_dir + '/val.txt', 'a') as f:
                f.write(f"{nowt} Validation Dataset: MR: {mr}, min ADE: {ade}, min FDE: {fde}\n")
            f.close()
    elif args.mode == 'test':
        load_checkpoint('./saved/#load_model#', model, optimizer)
        print("\nTesting: ")
        test_loader = enumerate(test_dataloader)
        if args.distributed_training == 0 or dist.get_rank() == 0:
            test_loader = tqdm(test_loader, total=len(test_dataloader))
            
        result = test(args, model, test_loader, device )
        competition_util.generate_forecasting_h5(result, './submission', args.lab_name)


if __name__ == "__main__":
    print('Using GPU: ' + str(torch.cuda.is_available()))
    input_args = get_training_parser().parse_args()
    input_args.nprocs = torch.cuda.device_count()
    hidden_size = input_args.hidden_size
    # mp.spawn(main, nprocs=input_args.nprocs, args=(input_args,))
    main(0, input_args)