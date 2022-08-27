import numpy as np
import math
import random
import torch
import os
import torch.distributed as dist

def rotate(x, y, angle):
    res_x = x * math.cos(angle) - y * math.sin(angle)
    res_y = x * math.sin(angle) + y * math.cos(angle)
    return res_x, res_y

def densen(arr1, arr2):
    num_points = len(arr1)
    start,end = arr2[0].reshape(-1,2),arr2[-1].reshape(-1,2)
    num_iter = len(arr1) - len(arr2)
    pre = arr2
    for i in range(num_iter):
        next = []
        for j in range(len(pre)-1):
            next.append((pre[j]+pre[j+1])/2)
        next = np.stack(next)
        pre = np.vstack([start,next,end])
    dense_points = pre
    if isinstance(dense_points,list):
        raise ValueError
    
    if len(arr1) != 10:
        arr1 = length2ten(arr1)
    if len(dense_points) != 10:
        dense_points = length2ten(dense_points)
    return arr1, dense_points

def length2ten(points):
    if len(points) < 10:
        start,end = points[0].reshape(-1,2),points[-1].reshape(-1,2)
        num_iter = 10 - len(points)
        pre = points
        for i in range(num_iter):
            next = []
            for j in range(len(pre)-1):
                next.append((pre[j]+pre[j+1])/2)
            next = np.stack(next)
            pre = np.vstack([start,next,end])
        dense_points = pre
    if len(points) > 10:
        start,end = points[0].reshape(-1,2),points[-1].reshape(-1,2)
        num_iter = len(points) - 10
        pre = points
        for i in range(num_iter):
            next = []
            for j in range(len(pre)-1):
                next.append((pre[j]+pre[j+1])/2)
            next_2 = []
            for j in range(len(next)-1):
                next_2.append((next[j]+next[j+1])/2)
            next_3 = []
            for j in range(len(next_2)-1):
                next_3.append((next_2[j]+next_2[j+1])/2)
            next_3 = np.stack(next_3)
            pre = np.vstack([start,next_3,end])
        dense_points = pre
    return dense_points

def increase_or_decrease(arr):
    if all(x<y for x, y in zip(arr[:, 0], arr[1:, 0])):
        return True
    if all(x>y for x, y in zip(arr[:, 0], arr[1:, 0])):
        return True

    min_id = np.argmin(arr[:, 0])
    max_id = np.argmax(arr[:, 0])
    
    h = 0
    if min_id != 0 or len(arr):
        h = arr[min_id, 0]
        if arr[0, 0] - h < 1 or arr[-1, 0] - h < 1:
            return True
    else:
        h = arr[max_id, 0]
        if h - arr[0, 0] < 1 or h - arr[-1, 0] < 1:
            return True
    return False

def merge_arrys(arrys, hidden_size=None):
    lengths = []
    hidden_size = arrys[0].shape[1]
    for arry in arrys:
        lengths.append(arry.shape[0] if arry is not None else 0)
    max_lengths = max(lengths)
    res = np.zeros([len(arrys), max_lengths, hidden_size])
    for i, arry in enumerate(arrys):
        if arry is not None:
            res[i][:arry.shape[0]] = arry
    return res, np.array(lengths)

def set_seed_globally(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def int_tuple(s, delim=','):
    return tuple(int(i) for i in s.strip().split(delim))

def get_angle(x, y):
    return math.atan2(y, x)

def to_origin_coordinate(points, idx_in_batch, origin_points, origin_angles, scale=None):
    for point in points:
        point[0], point[1] = rotate(point[0] - origin_points[idx_in_batch][0],
                                    point[1] - origin_points[idx_in_batch][1], origin_angles[idx_in_batch])

        if scale is not None:
            point[0] *= scale
            point[1] *= scale

def lane_gcn_to_origin_coordinate(pred_tur_traj, data, batch_id):
    pred_fut_traj = torch.tensor(pred_tur_traj[batch_id])
    rot = torch.tensor(data['rot'][batch_id])
    orig = torch.tensor(data['orig'][batch_id])

    result = torch.matmul(pred_fut_traj, rot) + orig.view(1, 1, 1, -1)
    return result.numpy()

def save_checkpoint(checkpoint_dir, model, optimizer, end_epoch, date, best_MR, lab_name):
    state = {
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'end_epoch' : end_epoch,
        }
    checkpoint_path = os.path.join(checkpoint_dir, f'{lab_name}_epoch_{end_epoch}.{date}.MR_{best_MR}.pth')
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

def reduce_min(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.MIN)
    return rt

def reduce_sum(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt
    
def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    new_dict = {}
    # for key, value in state['state_dict'].items():
    #     new_dict[key[7:]] = value
    # model.load_state_dict(new_dict)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

class StepLR:
    def __init__(self, lr, lr_epochs):
        assert len(lr) - len(lr_epochs) == 1
        self.lr = lr
        self.lr_epochs = lr_epochs

    def __call__(self, epoch):
        idx = 0
        for lr_epoch in self.lr_epochs:
            if epoch < lr_epoch:
                break
            idx += 1
        return self.lr[idx]