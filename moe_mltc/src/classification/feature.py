#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Dict, List

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, default_data_collator


class FeatureEngineering(object):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        source_column: str,
        label_column: str,
        label_columns: List[str],
        label_dict: Dict[str, int],
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.source_column = source_column
        self.label_columns = label_columns
        self.label_column = label_column
        self.label_dict = label_dict

    def create_data_loader(self, file_path: str, batch_size: int) -> DataLoader:
        """
        创建数据收集器
        :param mode: 模型，train, dev, test
        :param file_path: 文件的地址
        :param batch_size: 批次的大小
        :return:
        """
        dataset = load_dataset("json", data_files=file_path)
        feature_inputs = dataset["train"].map(
            function=self._encode_train,
            batched=True,
            num_proc=1,
            remove_columns=[self.source_column, self.label_column],
        )
        data_loader = DataLoader(
            dataset=feature_inputs,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=default_data_collator,
        )
        return data_loader

    def _encode_train(self, example):
        model_inputs = {
            "input_ids": [],
            "token_type_ids": [],
            "attention_mask": [],
            "label_ids": [],
        }

        for query, labels in zip(
            example[self.source_column], example[self.label_column]
        ):
            f = self.tokenizer(
                text=str(query),
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
            )

            model_inputs["input_ids"].append(f["input_ids"])
            model_inputs["token_type_ids"].append(f["token_type_ids"])
            model_inputs["attention_mask"].append(f["attention_mask"])

            label_ids = []
            for label in labels:
                label_ids.append(int(self.label_dict[str(label)]))

            model_inputs["label_ids"].append(label_ids)

        return model_inputs
