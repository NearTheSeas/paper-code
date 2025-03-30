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
        self.last = self.config["model"]["last_path"]
        if len(self.last) == 0:
            self.last = None
        self.model = self.config["model"]["model"]
        self.drop_prob = self.config["model"]["drop_prob"]
        self.hidden_size = self.config["model"]["hidden_size"]
        self.num_experts = self.config["model"]["num_experts"]
        self.moe_intermediate_size = self.config["model"]["moe_intermediate_size"]
        self.target_size = self.config["model"]["target_size"]
        self.top_k = self.config["model"]["top_k"]

        # data
        with open(self.config["data"]["label_dict"], "r", encoding="utf-8") as reader:
            tmp = json.load(reader)
            self.label_columns = tmp["label_columns"]
            self.label_dict = tmp["label_value"]

        self.train_path = self.config["data"]["train_path"]
        self.dev_path = self.config["data"]["dev_path"]
        self.test_path = self.config["data"]["test_path"]
        self.max_length = self.config["data"]["max_length"]
        self.source_column = self.config["data"]["source_column"]
        self.label_column = self.config["data"]["label_column"]

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
        self.accuracy = "accuracy"
        self.precision = "precision"
        self.recall = "recall"
        self.f1 = "f1"
