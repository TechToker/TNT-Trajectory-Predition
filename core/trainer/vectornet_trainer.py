import copy
import time

from tqdm import tqdm

import wandb
import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import DataLoader
from torch_geometric.nn import DataParallel

import core.util.visualization as visual

from core.trainer.trainer import Trainer
from core.model.vectornet import VectorNet
from core.optim_schedule import ScheduledOptim
from core.drivable_area_loss import DrivableAreaLoss


class VectorNetTrainer(Trainer):
    """
    VectorNetTrainer, train the vectornet with specified hyperparameters and configurations
    """
    def __init__(self,
                 trainset,
                 evalset,
                 testset,
                 batch_size: int = 1,
                 num_workers: int = 1,
                 num_global_graph_layer=1,
                 horizon: int = 30,
                 lr: float = 1e-3,
                 betas=(0.9, 0.999),
                 weight_decay: float = 0.01,
                 warmup_epoch=15,
                 lr_update_freq=5,
                 lr_decay_rate=0.3,
                 aux_loss: bool = False,
                 with_cuda: bool = False,
                 cuda_device=None,
                 log_freq: int = 2,
                 save_folder: str = "",
                 model_path: str = None,
                 ckpt_path: str = None,
                 verbose: bool = True
                 ):
        """
        trainer class for vectornet
        :param train_loader: see parent class
        :param eval_loader: see parent class
        :param test_loader: see parent class
        :param lr: see parent class
        :param betas: see parent class
        :param weight_decay: see parent class
        :param warmup_steps: see parent class
        :param with_cuda: see parent class
        :param multi_gpu: see parent class
        :param log_freq: see parent class
        :param model_path: str, the path to a trained model
        :param ckpt_path: str, the path to a stored checkpoint to be resumed
        :param verbose: see parent class
        """
        super(VectorNetTrainer, self).__init__(
            trainset=trainset,
            evalset=evalset,
            testset=testset,
            batch_size=batch_size,
            num_workers=num_workers,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            warmup_epoch=warmup_epoch,
            with_cuda=with_cuda,
            cuda_device=cuda_device,
            log_freq=log_freq,
            save_folder=save_folder,
            verbose=verbose
        )

        # init or load model
        self.aux_loss = aux_loss
        # input dim: (20, 8); output dim: (30, 2)

        model_name = VectorNet

        self.model = model_name(
            self.trainset.num_features,
            horizon,
            num_global_graph_layer=num_global_graph_layer,
            with_aux=aux_loss,
            device=self.device
        )

        # torch.save(self.model.state_dict(), 'default_model.pth')
        # raise ValueError('A very specific bad thing happened.')

        # resume from model file or maintain the original
        if model_path:
            self.load(model_path, 'm')

        if self.multi_gpu:
            # self.model = DataParallel(self.model)
            if self.verbose:
                print("[TNTTrainer]: Train the mode with multiple GPUs: {}.".format(self.cuda_id))
        else:
            if self.verbose:
                print("[TNTTrainer]: Train the mode with single device on {}.".format(self.device))
        self.model = self.model.to(self.device)

        # init optimizer
        self.optim = Adam(self.model.parameters(), lr=self.lr, betas=self.betas, weight_decay=self.weight_decay)
        self.optm_schedule = ScheduledOptim(
            self.optim,
            self.lr,
            n_warmup_epoch=self.warmup_epoch,
            update_rate=lr_update_freq,
            decay_rate=lr_decay_rate
        )

        # load ckpt
        if ckpt_path:
            self.load(ckpt_path, 'c')

    def iteration(self, epoch, dataloader):
        training = self.model.training
        avg_loss = 0.0
        num_sample = 0

        data_iter = tqdm(
            enumerate(dataloader),
            desc="{}_Ep_{}: loss: {:.5e}; avg_loss: {:.5e}".format("train" if training else "eval", epoch, 0.0, avg_loss),
            total=len(dataloader),
            bar_format="{l_bar}{r_bar}"
        )

        #return 0

        for i, data in data_iter:
            n_graph = data.num_graphs
            if training:

                loss = self.model.loss_override_test(data.to(self.device))
                #loss = self.model.loss(data.to(self.device))

                self.optm_schedule.zero_grad()
                loss.backward()
                self.optim.step()
                self.write_log("Train Loss", loss.detach().item() / n_graph, i + epoch * len(dataloader))

            else:
                with torch.no_grad():

                    loss = self.model.loss_override_test(data.to(self.device))
                    #loss = self.model.loss(data.to(self.device))

                    self.write_log("Eval Loss", loss.item() / n_graph, i + epoch * len(dataloader))

            num_sample += n_graph
            avg_loss += loss.detach().item()

            # print log info
            desc_str = "[Info: {}_Ep_{}: loss: {:.5e}; avg_loss: {:.5e}]".format("train" if training else "eval", epoch, loss.item() / n_graph, avg_loss / num_sample)

            data_iter.set_description(desc=desc_str, refresh=True)

        if training:
            learning_rate = self.optm_schedule.step_and_update_lr()
            self.write_log("LR", learning_rate, epoch)

        return avg_loss / num_sample

    def epoch_ending(self, train_loss, val_loss, metric):
        # return 0
        wandb.log({'Train loss': train_loss,
                   'Val loss': val_loss,
                   'Learning_rate': self.optim.param_groups[0]['lr'],
                   'MinADE': metric["minADE"],
                   'MinFDE': metric["minFDE"],
                   'MR': metric["MR"],
                   'Offroad_rate': metric['offroad_rate']
                   })

    def test(self):
        # Visualization on map

        # batch size must be = 1, otherwise we can't separate nodes by scenes
        with torch.no_grad():
            for data in tqdm(self.test_loader):

                target_future_trajectory = data.y.numpy().reshape(-1, 2)
                nodes = data.x.numpy()
                cluster = data.cluster.numpy()

                model_prediction = self.model.inference(data.to(self.device)).cpu().numpy().reshape(-1, 2)

                polylines = []
                for cluster_idc in np.unique(cluster):
                    [indices] = np.where(cluster == cluster_idc)
                    polyline = nodes[indices]

                    polylines.append(polyline)

                visual.draw_scene(polylines, target_future_trajectory, model_prediction)
