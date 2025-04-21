#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os

import evaluate
import torch
import torch.nn as nn
from torch import Tensor
from argument import Argument
from feature import MMDataset
from loguru import logger
from model import MultiModal
from torch.nn.utils import clip_grad_norm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from transformers import AdamW, AutoTokenizer, get_scheduler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class Train(object):
    def __init__(self, args: Argument):
        self.args = args
        self.device = self._init_environment()
        self.tokenizer = self._init_tokenizer()
        self.model = self._init_model()
        self.train_dataloader = self._init_dataloader(mode="train")
        self.dev_dataloader = self._init_dataloader(mode="valid")
        self.test_dataloader = self._init_dataloader(mode="test")
        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()
        self.writer = SummaryWriter(log_dir=self.args.log_dir)
        self.accuracy_metric = evaluate.load(self.args.accuracy)
        self.precision_metric = evaluate.load(self.args.precision)
        self.recall_metric = evaluate.load(self.args.recall)
        self.f1_metric = evaluate.load(self.args.f1)
        self.loss = nn.CrossEntropyLoss()

    def _init_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.args.model)
        return tokenizer

    def _init_environment(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device

    def _init_dataloader(self, mode):
        dataset = MMDataset(data_path=self.args.data_path,
                            tokenizer=self.tokenizer, mode=mode)
        dataloader = DataLoader(
            dataset=dataset, batch_size=self.args.batch_size, shuffle=True)
        return dataloader

    def _init_model(self):
        model = MultiModal(
            text_model_path=self.args.model,
            hidden_size=self.args.hidden_size,
            video_size=self.args.video_size,
            intermediate_size=self.args.intermediate_size,
            audio_size=self.args.audio_size,
            dropout=self.args.drop_prob,
        )
        model.to(self.device)

        if self.args.is_data_parallel:
            model = nn.DataParallel(model)
        return model

    def _init_optimizer(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            params=optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        return optimizer

    def _init_scheduler(self):
        num_update_steps_per_epoch = len(self.train_dataloader)
        max_train_steps = self.args.train_epochs * num_update_steps_per_epoch
        warm_steps = int(self.args.warmup_ratio * max_train_steps)

        scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=warm_steps,
            num_training_steps=max_train_steps,
        )
        return scheduler

    def train_and_evaluate(self):
        global_step, best_f1 = 0, 0
        loss_list = []
        for epoch in range(1, self.args.train_epochs + 1):
            for batch in self.train_dataloader:
                logits, compare_loss = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    token_type_ids=batch["token_type_ids"].to(self.device),
                    video_inputs=batch["video"].float().to(self.device),
                    audio_inputs=batch["audio"].float().to(self.device),
                    video_length=batch["video_length"].to(self.device),
                    audio_length=batch["audio_length"].to(self.device),
                )
                loss = self.loss(
                    input=logits, target=batch["label"].to(self.device)
                )
                loss = loss + compare_loss

                if self.args.is_data_parallel:
                    loss = torch.mean(loss)

                if global_step % self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                if global_step % self.args.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    loss_list.append(loss)

                if global_step % self.args.logging_steps == 0:
                    loss_avg = sum(loss_list) / len(loss_list)
                    self.writer.add_scalar(
                        "train/train_loss", loss_avg, global_step)
                    logger.info(
                        "global step: {}, epoch: {}, loss: {}",
                        global_step,
                        epoch,
                        loss_avg,
                    )

                if global_step % self.args.valid_steps == 0:
                    cur_save_dir = os.path.join(
                        self.args.save_dir, "model_%d" % global_step
                    )
                    if not os.path.exists(cur_save_dir):
                        os.makedirs(cur_save_dir)

                    if self.args.is_data_parallel:
                        torch.save(
                            self.model.module, os.path.join(
                                cur_save_dir, "model.pt")
                        )
                    else:
                        torch.save(self.model, os.path.join(
                            cur_save_dir, "model.pt"))

                    self.tokenizer.save_pretrained(cur_save_dir)

                    accuracy, precision, recall, f1 = self.evaluate_model()
                    self.writer.add_scalar(
                        "eval/accuracy", accuracy, global_step)
                    self.writer.add_scalar(
                        "eval/precision", precision, global_step)
                    self.writer.add_scalar("eval/recall", recall, global_step)
                    self.writer.add_scalar("eval/f1", f1, global_step)
                    logger.info(
                        "evaluation dev datasets accuracy: {}, precision: {}, recall: {}, F1: {}",
                        accuracy,
                        precision,
                        recall,
                        f1,
                    )

                    if f1 > best_f1:
                        best_f1 = f1
                        cur_save_dir = os.path.join(
                            self.args.save_dir, "model_best")
                        if not os.path.exists(cur_save_dir):
                            os.makedirs(cur_save_dir)

                        if self.args.is_data_parallel:
                            torch.save(
                                self.model.module,
                                os.path.join(cur_save_dir, "model.pt"),
                            )
                        else:
                            torch.save(
                                self.model, os.path.join(
                                    cur_save_dir, "model.pt")
                            )

                        self.tokenizer.save_pretrained(cur_save_dir)

                    accuracy, precision, recall, f1 = self.test_model()
                    self.writer.add_scalar(
                        "test/accuracy", accuracy, global_step)
                    self.writer.add_scalar(
                        "test/precision", precision, global_step)
                    self.writer.add_scalar("test/recall", recall, global_step)
                    self.writer.add_scalar("test/f1", f1, global_step)
                    logger.info(
                        "evaluation test datasets accuracy: {}, precision: {}, recall: {}, F1: {}",
                        accuracy,
                        precision,
                        recall,
                        f1,
                    )
                global_step += 1

    def evaluate_model(self):
        self.model.eval()
        predict_list, label_list = [], []
        with torch.no_grad():
            for step, batch in enumerate(self.dev_dataloader):
                logits, _ = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    token_type_ids=batch["token_type_ids"].to(self.device),
                    video_inputs=batch["video"].float().to(self.device),
                    audio_inputs=batch["audio"].float().to(self.device),
                    video_length=batch["video_length"].to(self.device),
                    audio_length=batch["audio_length"].to(self.device),
                )
                predictions = torch.argmax(
                    torch.softmax(logits, dim=-1), dim=-1).view(-1)
                predict_list += predictions.cpu().detach().numpy().tolist()
                label_list += batch["label"].view(-1).cpu(
                ).detach().numpy().tolist()

            accuracy = accuracy_score(y_true=label_list, y_pred=predict_list)
            precision = precision_score(
                y_true=label_list, y_pred=predict_list, average="macro"
            )
            recall = recall_score(
                y_true=label_list, y_pred=predict_list, average="macro"
            )
            f1 = f1_score(y_true=label_list,
                          y_pred=predict_list, average="macro")

            self.model.train()
            return accuracy, precision, recall, f1

    def test_model(self):
        self.model.eval()
        predict_list, label_list = [], []
        with torch.no_grad():
            for step, batch in enumerate(self.test_dataloader):
                logits, _ = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    token_type_ids=batch["token_type_ids"].to(self.device),
                    video_inputs=batch["video"].float().to(self.device),
                    audio_inputs=batch["audio"].float().to(self.device),
                    video_length=batch["video_length"].to(self.device),
                    audio_length=batch["audio_length"].to(self.device),
                )
                predictions = torch.argmax(
                    torch.softmax(logits, dim=-1), dim=-1).view(-1)
                predict_list += predictions.cpu().detach().numpy().tolist()
                label_list += batch["label"].view(-1).cpu(
                ).detach().numpy().tolist()

            accuracy = accuracy_score(y_true=label_list, y_pred=predict_list)
            precision = precision_score(
                y_true=label_list, y_pred=predict_list, average="macro"
            )
            recall = recall_score(
                y_true=label_list, y_pred=predict_list, average="macro"
            )
            f1 = f1_score(y_true=label_list,
                          y_pred=predict_list, average="macro")

            self.model.train()
            return accuracy, precision, recall, f1


if __name__ == "__main__":
    config_path = "src/resources/config/model.yaml"
    arg = Argument(config_path=config_path)
    train = Train(args=arg)

    train.train_and_evaluate()
