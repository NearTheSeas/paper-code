#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from model import MOEMultiClassification
from loguru import logger
from feature import FeatureEngineering
from argument import Argument
import torch.nn as nn
import torch
import os
import warnings
warnings.filterwarnings("ignore")


class Test(object):
    def __init__(self, args: Argument):
        self.args = args
        self.device = self._init_environment()
        self.tokenizer = self._init_tokenizer()
        self.feature = FeatureEngineering(
            tokenizer=self.tokenizer,
            max_length=self.args.max_length,
            source_column=self.args.source_column,
            label_column=self.args.label_column,
            label_columns=self.args.label_columns,
            label_dict=self.args.label_dict,
        )
        self.model = self._init_model()
        self.test_dataloader = self.feature.create_data_loader(
            file_path=self.args.test_path, batch_size=self.args.batch_size
        )

    def _init_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.args.model)
        return tokenizer

    def _init_environment(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device

    def _init_model(self):
        # 加载保存的最佳模型
        model_path = os.path.join(self.args.saved_model_dir, "model.pt")
        if os.path.exists(model_path):
            model = torch.load(model_path, map_location=self.device)
            logger.info(f"Loaded model from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model.to(self.device)
        if self.args.is_data_parallel:
            model = nn.DataParallel(model)
        return model

    def test_model(self):
        self.model.eval()
        predict_list, label_list = [], []
        with torch.no_grad():
            for step, batch in enumerate(self.test_dataloader):
                logits = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    token_type_ids=batch["token_type_ids"].to(self.device),
                )
                predictions = torch.argmax(
                    torch.softmax(logits, dim=-1), dim=-1).view(-1)
                predict_list += predictions.cpu().detach().numpy().tolist()
                label_list += batch["labels"].view(-1).cpu(
                ).detach().numpy().tolist()

            accuracy = accuracy_score(y_true=label_list, y_pred=predict_list)
            precision = precision_score(
                y_true=label_list, y_pred=predict_list, average="macro")
            recall = recall_score(
                y_true=label_list, y_pred=predict_list, average="macro")
            f1 = f1_score(y_true=label_list,
                          y_pred=predict_list, average="macro")

            logger.info(
                "Test results - accuracy: {:.4f}, precision: {:.4f}, recall: {:.4f}, F1: {:.4f}",
                accuracy, precision, recall, f1
            )

            return accuracy, precision, recall, f1


if __name__ == "__main__":
    config_path = "src/resources/configs/model.yaml"
    arg = Argument(config_path=config_path)
    test = Test(args=arg)
    test.test_model()
