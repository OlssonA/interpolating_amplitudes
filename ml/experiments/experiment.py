import numpy as np
import torch
import math

import os, time
from omegaconf import open_dict
from pathlib import Path
from hydra.utils import instantiate

from experiments.dataset import AmplitudeDataset
from experiments.preprocessing import (
    preprocess_particles_gatr,
    preprocess_particles_mlp,
    preprocess_amplitude_gggh,
    undo_preprocess_amplitude_gggh,
)

from experiments.misc import get_device, flatten_dict

from gatr.layers import MLPConfig, SelfAttentionConfig

TYPE_TOKEN_DICT = {
    "qq_tth_test": [0, 0, 1, 1, 2],
    "qq_tth_loop_test": [0, 1, 2, 3, 4],
    "gg_tth_test": [0, 0, 1, 1, 2],
    "gg_tth_loop_test": [0, 0, 1, 1, 2],
    "gggh_test": [0, 0, 1, 2]
}

class AmplitudeExperiment:

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self):
        # Initialize GPU backend, dtypes and run name
        self.device = get_device()
        print(f"Using device {self.device}")

        torch.backends.cuda.enable_flash_sdp(self.cfg.enable_flash_sdp)
        torch.backends.cuda.enable_math_sdp(self.cfg.enable_math_sdp)
        torch.backends.cuda.enable_mem_efficient_sdp(
            self.cfg.enable_mem_efficient_sdp
        )
        self.dtype = torch.float32
        run_name = self.cfg.run_name
        print(f"### Starting experiment {self.cfg.exp_name}/{run_name} ###")
        self.full_run()

    def full_run(self):
        t0 = time.time()

        self.init_physics()
        self.init_model()
        self.init_data()
        self._init_dataloader()
        self.evaluate()

        if self.device == torch.device("cuda"):
            max_used = torch.cuda.max_memory_allocated()
            max_total = torch.cuda.mem_get_info()[1]
            print(
                f"GPU RAM information: max_used = {max_used/1e9:.3} GB, max_total = {max_total/1e9:.3} GB"
            )
        dt = time.time() - t0
        print(
            f"Finished experiment {self.cfg.exp_name}/{self.cfg.run_name} after {dt/60:.2f}min = {dt/60**2:.2f}h"
        )

    def init_model(self):

        # initialize model
        self.model = instantiate(self.cfg.model)
        num_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(
            f"Instantiated model {type(self.model.net).__name__} with {num_parameters} learnable parameters"
        )

        # load existing model
        model_path = os.path.join(
            self.cfg.run_dir, "models", f"model_run0.pt"
        )
        try:
            state_dict = torch.load(model_path, map_location="cpu")["model"]
        except FileNotFoundError:
            raise ValueError(f"Cannot load model from {model_path}")
        print(f"Loading model from {model_path}")
        self.model.load_state_dict(state_dict)

        self.model.to(self.device, dtype=self.dtype)
        
    def init_physics(self):

        # create type_token list
        self.dataset = self.cfg.data.dataset[0]
        self.type_token = TYPE_TOKEN_DICT[self.dataset]

        token_size = max(self.type_token) + 1
        self.modelname = self.cfg.model.net._target_.rsplit(".", 1)[-1]

        with open_dict(self.cfg):
            # specify shape for type_token and MLPs
            if self.modelname == "GATr":
                self.cfg.model.net.in_s_channels = token_size
                self.cfg.model.token_size = token_size
            elif self.modelname == "MLP":
                self.cfg.model.net.in_shape = sum(range(1, len(TYPE_TOKEN_DICT[self.cfg.data.dataset[0]])))
            else:
                raise ValueError(f"model {self.modelname} not implemented")

            # reinsert_type_token
            if self.modelname == "GATr" and self.cfg.model.reinsert_type_token:
                self.cfg.model.net.reinsert_s_channels = list(range(n_type_tokens))

    def init_data(self):
        print(
            f"Working with dataset {self.cfg.data.dataset} "
            f"and type_token={self.type_token}"
        )

        # load data
        data_path = os.path.join(self.cfg.data.data_path, f"{self.dataset}.npy")
        
        assert os.path.exists(data_path), f"data_path {data_path} does not exist"
        data_raw = np.load(data_path)
        mask_zeros = data_raw[:, -1] != 0
        data_raw = data_raw[mask_zeros]
        print(f"Loaded data with shape {data_raw.shape} from {data_path}")

        # bring data into correct shape
        particles = data_raw[:, :4*len(TYPE_TOKEN_DICT[self.dataset])]
        particles = particles.reshape(
            particles.shape[0], particles.shape[1] // 4, 4
        )
        amp_tree = data_raw[:, [-2]]
        amplitudes = data_raw[:, [-1]]

        if self.cfg.data.kfac:
            amplitudes = amplitudes / amp_tree

        if self.cfg.data.dataset == ['gggh']:
            amplitudes_prepd = preprocess_amplitude_loop(amplitudes, self.cfg.data.prepd_mean, self.cfg.data.prepd_std)
        else:
            amplitudes_prepd = amplitudes

        if self.modelname == "MLP":
            particles_prepd = preprocess_particles_mlp(particles, self.cfg.data.particles_prepd_mean, self.cfg.data.particles_prepd_std)
        elif self.modelname == "GATr":
            particles_prepd = preprocess_particles_gatr(particles, self.cfg.data.particles_prepd_std)
        else:
            raise ValueError(f"model {self.modelname} not implemented")

        # collect everything
        self.particles = particles
        self.amplitudes = amplitudes
        self.particles_prepd = particles_prepd
        self.amplitudes_prepd = amplitudes_prepd
        self.amp_tree = amp_tree

    def _init_dataloader(self):
        self.loader = torch.utils.data.DataLoader(
            dataset=AmplitudeDataset(self.particles_prepd, self.amplitudes_prepd, self.amp_tree, dtype=self.dtype),
            batch_size=self.cfg.evaluation.batchsize,
            shuffle=False,
        )

        print(
            f"Constructed dataloader with batches={len(self.loader)}, batch_size={self.cfg.evaluation.batchsize}"
        )

    def evaluate(self):
        with torch.no_grad():
            ps_points_prepd, amp_truth_prepd, amp_pred_prepd, amp_sm = [], [], [], []
            self.model.eval()
            t0 = time.time()
            for data in self.loader:
                x, y, y_sm = data
                pred = self.model(
                    x.to(self.device),
                    type_token=self.type_token,
                    global_token=0,
                )

                y_pred = pred[..., 0]

                ps_points_prepd.append(x.cpu().float().numpy())
                amp_pred_prepd.append(y_pred.cpu().float().numpy())
                amp_truth_prepd.append(y.flatten().cpu().float().numpy())
                amp_sm.append(y_sm.flatten().cpu().float().numpy())

            ps_points_prepd = np.concatenate(ps_points_prepd)
            amp_pred_prepd = np.concatenate(amp_pred_prepd)
            amp_truth_prepd = np.concatenate(amp_truth_prepd)
            amp_sm = np.concatenate(amp_sm)

            dt = (
                (time.time() - t0)
                * 1e6
                / amp_truth_prepd.shape[0]
            )
            print(
                f"Evaluation time: {dt:.2f}s for {amp_truth_prepd.shape[0]} events "
                f"using batchsize {self.cfg.evaluation.batchsize}"
            )

            # undo preprocessing
            if self.cfg.data.dataset == ['gggh']:
                amp_truth = undo_preprocess_amplitude_gggh(amp_truth_prepd, self.cfg.data.prepd_mean, self.cfg.data.prepd_std)
                amp_pred = undo_preprocess_amplitude_gggh(amp_pred_prepd, self.cfg.data.prepd_mean, self.cfg.data.prepd_std)
            else:
                amp_truth = amp_truth_prepd
                amp_pred = amp_pred_prepd
            
            if self.cfg.data.kfac:
                amp_pred = amp_pred*amp_sm
                amp_truth = amp_truth*amp_sm

            # compute metrics over actual amplitudes
            amp_pred_metric = amp_pred / amp_sm
            amp_true_metric = amp_truth / amp_sm

            amp_pred_metric = amp_pred_metric[~np.isnan(amp_true_metric) & (amp_true_metric != 0)]
            amp_true_metric = amp_true_metric[~np.isnan(amp_true_metric) & (amp_true_metric != 0)]

            norm_1 = np.linalg.norm(amp_true_metric, 1)/len(amp_true_metric)
            delta_norm_1 = np.linalg.norm(amp_pred_metric - amp_true_metric, 1)/len(amp_pred_metric)
            rel_delta_norm_1 = delta_norm_1/norm_1

            results = {
                "truth": amp_truth,
                "prediction": amp_pred,
                "rel_delta_norm_1": rel_delta_norm_1,
            }
            
            print(results)

