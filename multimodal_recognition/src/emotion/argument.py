#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import json

import yaml


class Argument(object):
    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as reader:
            self.config = yaml.load(reader.read(), Loader=yaml.FullLoader)

        # environment
        self.device = str(self.config["environment"]["devices"])
        self.is_data_parallel = True if "," in self.device else False

        # model
        self.model = self.config["model"]["model"]
        self.pretrain_model = self.config["model"]["pretrain_model"]
        self.drop_prob = self.config["model"]["drop_prob"]
        self.hidden_size = self.config["model"]["hidden_size"]
        self.video_size = self.config["model"]["video_size"]
        self.audio_size = self.config["model"]["audio_size"]
        self.intermediate_size = self.config["model"]["intermediate_size"]

        # data
        self.data_path = self.config["data"]["data_path"]

        # log
        self.log_dir = self.config["log"]["log_dir"]
        self.log_name = self.config["log"]["log_name"]
        self.logging_steps = self.config["log"]["logging_steps"]

        # save
        self.save_dir = self.config["save"]["save_dir"]
        self.saved_model_dir = self.config["save"]["saved_model_dir"]

        # optimizer
        self.learning_rate = float(self.config["optimizer"]["learning_rate"])
        self.weight_decay = self.config["optimizer"]["weight_decay"]

        # scheduler
        self.warmup_ratio = self.config["scheduler"]["warmup_ratio"]

        # configs
        self.batch_size = self.config["configs"]["batch_size"]
        self.train_epochs = self.config["configs"]["train_epochs"]
        self.gradient_accumulation_steps = self.config["configs"][
            "gradient_accumulation_steps"
        ]
        self.max_grad_norm = self.config["configs"]["max_grad_norm"]
        self.valid_steps = self.config["configs"]["valid_steps"]

        # metrics
        self.accuracy = r"/home/chenghuadong.chd/code/metrics/accuracy"
        self.precision = r"/home/chenghuadong.chd/code/metrics/precision"
        self.recall = r"/home/chenghuadong.chd/code/metrics/recall"
        self.f1 = r"/home/chenghuadong.chd/code/metrics/f1"
