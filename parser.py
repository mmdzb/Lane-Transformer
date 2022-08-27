import argparse

from numpy import False_
from utils import int_tuple
def get_training_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="./log/", help="Directory containing logging file")
    parser.add_argument("--model_dir", default="", help="Directory containing logging file")

    # vector net
    parser.add_argument("--sub_graph_depth", default=3, type=int)

    # lab setting
    parser.add_argument("--mode", default="train", type=str)
    parser.add_argument("--data_type", default="mine", type=str)
    parser.add_argument("--data_augment", default=0)

    # 文件的命名
    parser.add_argument("--lab_name", default="baseline", type=str)

    # dataset setting
    parser.add_argument("--pkl_save_dir", default="/workspace2/wzb/Datasets/Argoverse/pkl/", type=str)
    parser.add_argument("--resume", default=1)
    parser.add_argument("--loader_num_workers", default=1, type=int)

    # train setting
    parser.add_argument("--lr", default=0.0005)
    parser.add_argument("--batch_size", default=16)
    parser.add_argument("--warm_up_epochs", default=10)
    parser.add_argument("--num_epochs", default=60)
    parser.add_argument("--continue_training", default=0)
    parser.add_argument("--distributed_training", default=0)
    parser.add_argument("--k_num", default=6)
    parser.add_argument("--local_rank", default=0)
    parser.add_argument("--nprocs", default=0)
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--seed", default=72, type=int)

    return parser