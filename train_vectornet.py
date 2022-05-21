import os
import sys
from os.path import join as pjoin
from datetime import datetime
import torch
import random
import numpy as np

import wandb
import argparse

from core.dataloader.argoverse_loader import Argoverse, GraphData
from core.dataloader.argoverse_loader_v2 import ArgoverseInMem as ArgoverseInMemv2
from core.dataloader.argoverse_loader_low_memory import ArgoverseCustom, GRAPH_TYPE
from core.trainer.vectornet_trainer import VectorNetTrainer

sys.path.append("core/dataloader")


def fix_random_seed():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)

    magic_value = 42

    torch.manual_seed(magic_value)
    torch.cuda.manual_seed(magic_value)
    torch.cuda.manual_seed_all(magic_value)

    random.seed(magic_value)
    np.random.seed(magic_value)

    # This will slightly reduce performance of CUDA convolution
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # thats for dataloader reproducibility
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(magic_value)


def train(args):
    """
    script to train the vectornet
    :param args:
    :return:
    """

    fix_random_seed()

    #train_set = ArgoverseInMemv2(pjoin(args.data_root, "train_intermediate")) #.shuffle()
    #eval_set = ArgoverseInMemv2(pjoin(args.data_root, "val_intermediate"))

    train_set = ArgoverseCustom(pjoin(args.data_root, "train_intermediate"), 79817, graph_type=GRAPH_TYPE.DRIVABLE_AREA) # 79817 # 205942 #.shuffle()
    eval_set = ArgoverseCustom(pjoin(args.data_root, "val_intermediate"), 13839, graph_type=GRAPH_TYPE.DRIVABLE_AREA) # 13839 # 39472

    # init output dir
    time_stamp = datetime.now().strftime("%m-%d-%H-%M-%S")
    output_dir = pjoin(args.output_dir, time_stamp)
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        raise Exception("The output folder does exists and is not empty! Check the folder.")
    else:
        os.makedirs(output_dir)

    # init trainer
    trainer = VectorNetTrainer(
        trainset=train_set,
        evalset=eval_set,
        testset=eval_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        warmup_epoch=args.warmup_epoch,
        lr_update_freq=args.lr_update_freq,
        lr_decay_rate=args.lr_decay_rate,
        weight_decay=args.adam_weight_decay,
        betas=(args.adam_beta1, args.adam_beta2),
        num_global_graph_layer=args.num_glayer,
        aux_loss=args.aux_loss,
        with_cuda=args.with_cuda,
        cuda_device=args.cuda_device,
        save_folder=output_dir,
        log_freq=args.log_freq,
        ckpt_path=args.resume_checkpoint if hasattr(args, "resume_checkpoint") and args.resume_checkpoint else None,
        model_path=args.resume_model if hasattr(args, "resume_model") and args.resume_model else None
    )

    # wandb.init(project='VectorNet', entity='techtoker')

    # resume minimum eval loss
    min_eval_loss = trainer.min_eval_loss
    num_modes = trainer.model.num_of_modes

    # training
    for iter_epoch in range(args.n_epoch):
        train_loss = trainer.train(iter_epoch)
        eval_loss = trainer.eval(iter_epoch)

        # compute the metrics and save
        metric = trainer.compute_metric()
        #
        # wandb.log({'Train loss': train_loss,
        #            'Val loss': eval_loss,
        #            'Learning_rate': trainer.optim.param_groups[0]['lr'],
        #            f'MinADE_{num_modes}': metric[f'minADE_over_{num_modes}'],
        #            f'MinFDE_{num_modes}': metric[f'minFDE_over_{num_modes}'],
        #            f'MR_{num_modes}': metric[f'MR_over_{num_modes}'],
        #            'MinADE': metric['minADE'],
        #            'MinFDE': metric['minFDE'],
        #            'MR': metric[f'MR'],
        #            'Offroad_rate': metric['offroad_rate']
        #            })

        if not min_eval_loss:
            min_eval_loss = eval_loss
        elif eval_loss < min_eval_loss:
            # save the model when a lower eval_loss is found
            min_eval_loss = eval_loss
            trainer.save(iter_epoch, min_eval_loss)
            trainer.save_model(metric, "best")

    trainer.save_model(None, "final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data_root", required=False, type=str, default="/home/techtoker/projects/TNT-Trajectory-Predition/dataset/interm_data_drivarea_dap_norm_balanced",
                        help="root dir for datasets")
    parser.add_argument("-o", "--output_dir", required=False, type=str, default="run/vectornet/",
                        help="ex)dir to save checkpoint and model")

    parser.add_argument("-l", "--num_glayer", type=int, default=1,
                        help="number of global graph layers")
    parser.add_argument("-a", "--aux_loss", action="store_true", default=True,
                        help="Training with the auxiliary recovery loss")

    parser.add_argument("-b", "--batch_size", type=int, default=64, #64,
                        help="number of batch_size")
    parser.add_argument("-e", "--n_epoch", type=int, default=50,
                        help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=1, # 16,
                        help="dataloader worker size")

    parser.add_argument("-c", "--with_cuda", action="store_true", default=True,
                        help="training with CUDA: true, or false")
    parser.add_argument("-cd", "--cuda_device", type=int, default=[], nargs='+',
                        help="CUDA device ids")
    parser.add_argument("--log_freq", type=int, default=2,
                        help="printing loss every n iter: setting n")
    parser.add_argument("--on_memory", type=bool, default=True, help="Loading on memory: true or false")

    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate of adam") # 8e-5
    parser.add_argument("-we", "--warmup_epoch", type=int, default=10, # 10
                        help="The epoch to start the learning rate decay")
    parser.add_argument("-luf", "--lr_update_freq", type=int, default=5,
                        help="learning rate decay frequency for lr scheduler")
    parser.add_argument("-ldr", "--lr_decay_rate", type=float, default=0.8, help="lr scheduler decay rate")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

    parser.add_argument("-rc", "--resume_checkpoint", type=str, help="resume a checkpoint for fine-tune")
    # parser.add_argument("-rc", "--resume_checkpoint", type=str,
    #                     #default="/home/techtoker/projects/TNT-Trajectory-Predition/pretrained_models/05-05-00-30/checkpoint_iter27.ckpt",
    #                     #default="/home/techtoker/projects/TNT-Trajectory-Predition/pretrained_models/05-04-12-42/checkpoint_iter6.ckpt",
    #                     default="/home/techtoker/projects/TNT-Trajectory-Predition/pretrained_models/05-20-18-37-58/checkpoint_iter33.ckpt", # MTP MODEL
    #                     help="resume a checkpoint for fine-tune")

    parser.add_argument("-rm", "--resume_model", type=str, help="resume a model state for fine-tune")
    # parser.add_argument("-rm", "--resume_model", type=str,
    #                     default="/home/techtoker/projects/TNT-Trajectory-Predition/pretrained_models/05-17-23-14-57/final_VectorNet.pth",
    #                     help="resume a model state for fine-tune")

    args = parser.parse_args()
    train(args)
