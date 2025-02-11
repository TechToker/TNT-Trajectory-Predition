# trainner to train the models
import copy
import os
from tqdm import tqdm

import json
import torch

import time
import random
import numpy as np

# from torch.utils.data import DataLoader,
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader, DataListLoader
from argoverse.evaluation.eval_forecasting import get_displacement_errors_and_miss_rate
import core.util.drivable_area_polygons_helper as da_helper
import core.util.visualization as visual

import gc


class Trainer(object):
    """
    Parent class for all the trainer class
    """
    def __init__(self,
                 trainset,
                 evalset,
                 testset,
                 loader=DataLoader,
                 batch_size: int = 1,
                 num_workers: int = 1,
                 lr: float = 1e-4,
                 betas=(0.9, 0.999),
                 weight_decay: float = 0.01,
                 warmup_epoch=5,
                 with_cuda: bool = False,
                 cuda_device=None,
                 enable_log: bool = False,
                 log_freq: int = 2,
                 save_folder: str = "",
                 verbose: bool = True
                 ):
        """
        :param train_loader: train dataset
        :param eval_loader: eval dataset
        :param test_loader: dataset
        :param lr: initial learning rate
        :param betas: Adam optiimzer betas
        :param weight_decay: Adam optimizer weight decay param
        :param warmup_steps: optimizatioin scheduler param
        :param with_cuda: tag indicating whether using gpu for training
        :param multi_gpu: tag indicating whether multiple gpus are using
        :param log_freq: logging frequency in epoch
        :param verbose: whether printing debug messages
        """

        self.fix_random_seed()

        # determine cuda device id
        self.cuda_id = cuda_device if with_cuda and cuda_device else [0]
        self.device = torch.device("cuda:{}".format(self.cuda_id[0]) if torch.cuda.is_available() and with_cuda else "cpu")
        self.multi_gpu = False if len(self.cuda_id) == 1 else True


        # dataset
        self.trainset = trainset
        self.evalset = evalset
        self.testset = testset
        self.batch_size = batch_size
        # self.loader = loader if not self.multi_gpu else DataListLoader
        self.loader = loader
        # print("[Debug]: using {} to load data".format(self.loader))

        self.train_loader = self.loader(
            self.trainset,
            batch_size=self.batch_size,
            # num_workers=num_workers,
            # pin_memory=True,
            shuffle=False
        )
        # self.eval_loader = self.loader(self.evalset, batch_size=self.batch_size, num_workers=num_workers)
        self.eval_loader = self.loader(self.evalset, batch_size=self.batch_size)
        # self.test_loader = self.loader(self.testset, batch_size=self.batch_size, num_workers=num_workers)
        self.test_loader = self.loader(self.testset, batch_size=self.batch_size)

        # model
        self.model = None

        # optimizer params
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.warmup_epoch = warmup_epoch
        self.optim = None
        self.optm_schedule = None

        # criterion and metric
        self.criterion = None
        self.min_eval_loss = None
        self.best_metric = None

        # log
        self.enable_log = enable_log
        self.save_folder = save_folder
        self.logger = SummaryWriter(log_dir=os.path.join(self.save_folder, "log"))
        self.log_freq = log_freq
        self.verbose = verbose

        gc.enable()

    @staticmethod
    def fix_random_seed():
        magic_value = 42

        torch.manual_seed(magic_value)
        torch.cuda.manual_seed(magic_value)

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

    def train(self, epoch):
        gc.collect()

        self.model.train()
        return self.iteration(epoch, self.train_loader)

    def eval(self, epoch):
        gc.collect()

        self.model.eval()
        return self.iteration(epoch, self.eval_loader)

    def test(self):
        self.model.eval()

    def iteration(self, epoch, dataloader):
        raise NotImplementedError

    def write_log(self, name_str, data, epoch):
        if not self.enable_log:
            return
        self.logger.add_scalar(name_str, data, epoch)

    # todo: save the model and current training status
    def save(self, iter_epoch, loss):
        """
        save current state of the training and update the minimum loss value
        :param save_folder: str, the destination folder to store the ckpt
        :param iter_epoch: int, ith epoch of current saving checkpoint
        :param loss: float, the loss of current saving state
        :return:
        """
        self.min_eval_loss = loss
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder, exist_ok=True)
        torch.save({
            "epoch": iter_epoch,
            # "model_state_dict": self.model.state_dict() if not self.multi_gpu else self.model.module.state_dict(),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optim.state_dict(),
            "min_eval_loss": loss
        }, os.path.join(self.save_folder, "checkpoint_iter{}.ckpt".format(iter_epoch)))
        if self.verbose:
            print("[Trainer]: Saving checkpoint to {}...".format(self.save_folder))

    def save_model(self, metric, prefix=""):
        """
        save current state of the model
        :param prefix: str, the prefix to the model file
        :return:
        """
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder, exist_ok=True)

        # skip model saving if the minADE is not better
        if self.best_metric and isinstance(metric, dict):
            if metric["minADE"] >= self.best_metric["minADE"]:
                print("[Trainer]: Best minADE: {}; Current minADE: {}; Skip model saving...".format(self.best_metric["minADE"], metric["minADE"]))
                return

        # save best metric
        self.best_metric = metric
        metric_stored_file = os.path.join(self.save_folder, "{}_metrics.txt".format(prefix))
        with open(metric_stored_file, 'a+') as f:
            f.write(json.dumps(self.best_metric))
            f.write("\n")

        # save model
        torch.save(
            # self.model.state_dict() if not self.multi_gpu else self.model.module.state_dict(),
            self.model.state_dict(),
            os.path.join(self.save_folder, "{}_{}.pth".format(prefix, type(self.model).__name__))
        )
        if self.verbose:
            print("[Trainer]: Saving model to {}...".format(self.save_folder))

    def load(self, load_path, mode='c'):
        """
        loading function to load the ckpt or model
        :param mode: str, "c" for checkpoint, or "m" for model
        :param load_path: str, the path of the file to be load
        :return:
        """
        if mode == 'c':
            # load ckpt
            ckpt = torch.load(load_path, map_location=self.device)
            try:
                if self.multi_gpu:
                    self.model.load_state_dict(ckpt["model_state_dict"])
                else:
                    self.model.load_state_dict(ckpt["model_state_dict"])
                self.optim.load_state_dict(ckpt["optimizer_state_dict"])
                self.min_eval_loss = ckpt["min_eval_loss"]
            except:
                raise Exception("[Trainer]: Error in loading the checkpoint file {}".format(load_path))
        elif mode == 'm':
            try:
                self.model.load_state_dict(torch.load(load_path, map_location=self.device))
            except:
                raise Exception("[Trainer]: Error in loading the model file {}".format(load_path))
        else:
            raise NotImplementedError

    def compute_metric(self, miss_threshold=2.0):
        show_amount_examples = 0

        """
        compute metric for test dataset
        :param miss_threshold: float,
        :return:
        """
        assert self.model, "[Trainer]: No valid model, metrics can't be computed!"
        assert self.testset, "[Trainer]: No test dataset, metrics can't be computed!"

        forecasted_trajectories, forecasted_probabilities, gt_trajectories = {}, {}, {}
        seq_id = 0

        num_modes = self.model.num_of_modes  # self.model.k if not self.multi_gpu else self.model.module.k
        horizon = self.model.horizon  # self.model.k if not self.multi_gpu else self.model.module.k

        self.model.eval()
        with torch.no_grad():

            outside_da_counter = 0
            total_amount_of_samples = 0

            for data in tqdm(self.test_loader):
                data_copy = copy.deepcopy(data)  # We save data_copy because inference overwrites it

                batch_size = data.num_graphs
                gt = data.y.unsqueeze(1).view(batch_size, -1, 2).cumsum(axis=1).numpy()  # cumulative format

                # inference and transform dimension
                out = self.model.inference(data.to(self.device))
                pred_y, traj_probabilities = out # For CoverNet

                # for MTP
                # traj_probabilities = out[:, -num_modes:]
                # out = out[:, :-num_modes]
                # end MTP

                # dim_out = len(out.shape)
                #pred_y = out.unsqueeze(dim_out).view((batch_size, num_modes, horizon, 2)).cumsum(axis=2).cpu().numpy()  # cumulative format; FOR vanila and MTP



                # Calculate offroad rate
                # clamped_drivable_areas = data.clamped_drivable_area
                # outline_drivable_area_indexes = data.outline_drivable_area_index
                total_amount_of_samples += batch_size

                # record the prediction and ground truth
                for batch_id in range(batch_size):

                    forecasted_trajectories[seq_id] = [pred_y_k for pred_y_k in pred_y[batch_id]]
                    forecasted_probabilities[seq_id] = traj_probabilities[batch_id]

                    gt_trajectories[seq_id] = gt[batch_id]

                    # print(f'probabilities: {traj_probabilities[batch_id]}')

                    # # Calculate offroad rate
                    # drivable_area_polygons = clamped_drivable_areas[batch_id]
                    # outline_da_polygon_index = outline_drivable_area_indexes[batch_id]
                    #
                    current_pred = forecasted_trajectories[seq_id]
                    # outside_da_mask_pred = da_helper.out_of_drivable_area_check(drivable_area_polygons, outline_da_polygon_index, current_pred, show_visualization=False)
                    #
                    current_gt = gt_trajectories[seq_id]
                    # #outside_da_mask_gt = da_helper.out_of_drivable_area_check(drivable_area_polygons, outline_da_polygon_index, current_gt, show_visualization=False)
                    #
                    # # visualization work only if batch size = 1

                    if show_amount_examples > 0:
                        polylines = da_helper.get_da_polylines_from_data(data_copy)
                        visual.draw_scene(polylines, current_gt, current_pred) #, outside_da_mask_pred)
                        show_amount_examples -= 1
                    #
                    # outside_da_counter += np.count_nonzero(outside_da_mask_pred)
                    # # end

                    seq_id += 1

            # calc metric over TOP1 trajectory
            metric_results = get_displacement_errors_and_miss_rate(
                forecasted_trajectories,
                gt_trajectories,
                1,
                horizon,
                miss_threshold,
                forecasted_probabilities
            )

            # calc metric over TOP_k trajectories
            metric_results_over_k = get_displacement_errors_and_miss_rate(
                forecasted_trajectories,
                gt_trajectories,
                num_modes,
                horizon,
                miss_threshold,
                forecasted_probabilities
            )

            for key in metric_results_over_k.keys():
                metric_results[f'{key}_over_{num_modes}'] = metric_results_over_k[key]

            metric_results['offroad_rate'] = outside_da_counter / (total_amount_of_samples * 30)  # 30 - forecast horizon

        return metric_results
