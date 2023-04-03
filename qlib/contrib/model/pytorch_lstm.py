# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division, print_function

import copy
from typing import Text, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...log import get_module_logger
from ...model.base import Model
from ...utils import get_or_create_path
from ...workflow import R

from torch.profiler import profile, record_function, ProfilerActivity
from datetime import datetime
import viztracer
from wszlib.constants import PROJECT_ROOT


class LSTM(Model):
    """LSTM Model

    Parameters
    ----------
    d_feat : int
        input dimension for each time step
    metric: str
        the evaluation metric used in early stop
    optimizer : str
        optimizer name
    GPU : str
        the GPU ID(s) used for training
    """

    def __init__(
        self,
        d_feat=6,
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        n_epochs=200,
        lr=0.001,
        metric="ic",
        batch_size=2000,
        early_stop=20,
        loss="mse",
        optimizer="adam",
        GPU=0,
        seed=None,
        **kwargs,
    ):
        # Set logger.
        self.logger = get_module_logger("LSTM")
        self.logger.info("LSTM pytorch version...")

        # set hyper-parameters.
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        GPU = int(GPU)
        self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.seed = seed

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.logger.info(
            "LSTM parameters setting:"
            "\nd_feat : {}"
            "\nhidden_size : {}"
            "\nnum_layers : {}"
            "\ndropout : {}"
            "\nn_epochs : {}"
            "\nlr : {}"
            "\nmetric : {}"
            "\nbatch_size : {}"
            "\nearly_stop : {}"
            "\noptimizer : {}"
            "\nloss_type : {}"
            "\nvisible_GPU : {}"
            "\nuse_GPU : {}"
            "\nseed : {}".format(
                d_feat,
                hidden_size,
                num_layers,
                dropout,
                n_epochs,
                lr,
                metric,
                batch_size,
                early_stop,
                optimizer.lower(),
                loss,
                GPU,
                self.use_gpu,
                seed,
            )
        )

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.lstm_model = LSTMModel(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.lstm_model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.lstm_model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        self.fitted = False
        self.lstm_model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    @property
    def model(self) -> nn.Module:
        return self.lstm_model

    def mse(self, pred, label):
        loss = (pred - label) ** 2
        return torch.mean(loss)

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)

        if self.loss == "mse":
            return self.mse(pred[mask], label[mask])

        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label):

        mask = torch.isfinite(label)

        if self.metric == "ic":
            x = pred[mask]
            y = label[mask]

            vx = x - torch.mean(x)
            vy = y - torch.mean(y)
            return torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2)))

        if self.metric == ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask])

        raise ValueError("unknown metric `%s`" % self.metric)

    def train_epoch(self, x_train, y_train):

        x_train_values = x_train.values
        y_train_values = np.squeeze(y_train.values)

        self.lstm_model.train()

        indices = np.arange(len(x_train_values))
        np.random.shuffle(indices)

        if hasattr(self, "profiling") and getattr(self, "profiling"):
            torch_profiler = torch.profiler.profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=2, warmup=3, active=4, repeat=3),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    f'{PROJECT_ROOT}/profiling/torch_profiler/Baseline_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
                ),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_flops=True,
                with_modules=True,
                use_cuda=True,
            )
            viz_tracer = viztracer.VizTracer(
                output_file=f"{PROJECT_ROOT}/profiling/viztracer/trace_baseline_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
            )
            viz_tracer.enable_thread_tracing()
            torch_profiler.start()
            viz_tracer.start()
            viz_traced = False

        for i in range(len(indices))[:: self.batch_size]:
            with record_function("Get Batch and Move to GPU"):
                if len(indices) - i < self.batch_size:
                    target_idx = len(indices)
                else:
                    target_idx = i + self.batch_size
                feature_np = x_train_values[indices[i:target_idx]]
                label_np = y_train_values[indices[i:target_idx]]
                # input("Indexed ndarray! Press Enter to continue...")
                feature = torch.tensor(feature_np, dtype=torch.float32, device=self.device)
                label = torch.tensor(label_np, dtype=torch.float32, device=self.device)
                # (
                #     torch.from_numpy(feature_np)
                #     .float()
                #     .to(self.device)
                # )
                # label = (
                #     torch.from_numpy(label_np)
                #     .float()
                #     .to(self.device)
                # )
                # input("Converted to GPU tensor! Press Enter to continue...")
            with record_function("Forward"):
                pred = self.lstm_model(feature)
                # input("FOrward finished! Press Enter to continue...")
            with record_function("Loss"):
                loss = self.loss_fn(pred, label)
                self.train_optimizer.zero_grad()
                loss.backward()
                # input("Backward finished! Press Enter to continue...")
            with record_function("Optimize"):
                torch.nn.utils.clip_grad_value_(self.lstm_model.parameters(), 3.0)
                self.train_optimizer.step()
                # input("Optimize finished! Press Enter to continue...")
            # input("Press Enter to continue...")
            if hasattr(self, "profiling") and getattr(self, "profiling"):
                torch_profiler.step()
                if not viz_traced:
                    viz_traced = True
                    viz_tracer.stop()
                    viz_tracer.save()
        if hasattr(self, "profiling") and getattr(self, "profiling"):
            torch_profiler.stop()
            self.logger.info("Profiling finished. Program exitting")
            exit(0)

    def test_epoch(self, data_x, data_y):

        # prepare training data
        x_values = data_x.values
        y_values = np.squeeze(data_y.values)

        self.lstm_model.eval()

        scores = []
        losses = []

        indices = np.arange(len(x_values))

        for i in range(len(indices))[:: self.batch_size]:

            if len(indices) - i < self.batch_size:
                break

            feature = torch.from_numpy(x_values[indices[i : i + self.batch_size]]).float().to(self.device)
            label = torch.from_numpy(y_values[indices[i : i + self.batch_size]]).float().to(self.device)

            pred = self.lstm_model(feature)
            loss = self.loss_fn(pred, label)
            losses.append(loss.item())

            score = self.metric_fn(pred, label)
            scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def fit(
        self,
        dataset: DatasetH,
        evals_result=dict(),
        save_path=None,
    ):
        df_train, df_valid, df_test = dataset.prepare(
            ["train", "valid", "test"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        if df_train.empty or df_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]

        save_path = get_or_create_path(save_path)
        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []

        # train
        self.logger.info("training...")
        self.fitted = True

        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(x_train, y_train)
            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(x_train, y_train)
            val_loss, val_score = self.test_epoch(x_valid, y_valid)
            R.log_metrics(
                step=step,
                **{
                    "train_loss": train_loss,
                    "train_score": train_score,
                    "val_loss": val_loss,
                    "val_score": val_score,
                },
            )

            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.lstm_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.lstm_model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        index = x_test.index
        self.lstm_model.eval()
        x_values = x_test.values
        sample_num = x_values.shape[0]
        preds = []

        for begin in range(sample_num)[:: self.batch_size]:
            if sample_num - begin < self.batch_size:
                end = sample_num
            else:
                end = begin + self.batch_size
            x_batch = torch.from_numpy(x_values[begin:end]).float().to(self.device)
            with torch.no_grad():
                pred = self.lstm_model(x_batch).detach().cpu().numpy()
            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=index)


class LSTMModel(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()

        self.rnn = nn.LSTM(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_size, 1)

        self.d_feat = d_feat

    def forward(self, x):
        # x: [N, F*T]
        x = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x = x.permute(0, 2, 1)  # [N, T, F]
        out, _ = self.rnn(x)
        return self.fc_out(out[:, -1, :]).squeeze()
