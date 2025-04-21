#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer, default_data_collator


class MMDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer, mode="train"):
        super().__init__()

        self.data_path = data_path
        self.mode = mode
        self.tokenizer = tokenizer
        self.max_length = 64

        self._init_data()

    def _init_data(self):
        with open(self.data_path, "rb") as reader:
            datas = pickle.load(reader)

        label_dict = {
            "Negative": 0,
            "Positive": 1
        }

        # 音频特征，音频特征使用LibROSA(McFee等人，2015年)语音工具包，以默认参数提取22050Hz的声学特征。总共提取了33个维度的帧级声学特征，包括1维对数基频（log F0）、20维Melfrequency cepstral coefficients（MFCCs）和12维Constant-Q chromatogram（CQT）。根据（Li等，2018）这些特征与情绪和语气有关。
        self.audio = datas[self.mode]["audio"]
        self.audio[self.audio == -np.inf] = 0
        self.audio_length = datas[self.mode]["audio_lengths"]

        # 视频特征，以30Hz的频率从视频片段中提取帧。我们使用MTCNN人脸检测算法（Zhang等人，2016a）来提取对齐的人脸。然后，遵循Zadeh等人（2018b），我们使用MultiComp OpenFace2.0工具包（Baltrusaitis等人，2018）提取68个面部地标、17个面部动作单元、头部姿势、头部方向和眼睛注视的集合。最后，共提取了709个维度的帧级视觉特征。
        self.video = datas[self.mode]["vision"]
        self.video[self.video == -np.inf] = 0
        self.video_length = datas[self.mode]["vision_lengths"]

        # 情感极性标签
        self.labels = np.array(
            [label_dict[idx] for idx in datas[self.mode]["annotations"].tolist()])

        # 文本特征
        self.input_ids = []
        self.token_type_ids = []
        self.attention_mask = []

        for query in datas[self.mode]["raw_text"]:
            f = self.tokenizer(
                text=str(query),
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
            )
            self.input_ids.append(f["input_ids"])
            self.token_type_ids.append(f["token_type_ids"])
            self.attention_mask.append(f["attention_mask"])
        self.input_ids = np.array(self.input_ids)
        self.token_type_ids = np.array(self.token_type_ids)
        self.attention_mask = np.array(self.attention_mask)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sample = {
            "audio": self.audio[index],
            "audio_length": self.audio_length[index],
            "video": self.video[index],
            "video_length": self.video_length[index],
            "input_ids": self.input_ids[index],
            "token_type_ids": self.token_type_ids[index],
            "attention_mask": self.attention_mask[index],
            "label": self.labels[index]
        }
        return sample
