#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import TransformerEncoderLayer, TransformerEncoder
import torch.nn.functional as F
from transformers import AutoModel
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class MMLoss(nn.Module):
    def __init__(self):
        super(MMLoss, self).__init__()

    def forward(self, predict: torch.Tensor):
        """
        计算batch内互为负样本，相邻的01为正样本，23为正样本，45为正样本，....
        其他的为负样本
        :param predict: batch_size * 2 的句向量
        :return:
        """
        device = predict.device
        y_true = torch.arange(predict.shape[0], device=device)
        y_shifting = (y_true - y_true % 2 * 2) + 1
        y_true = torch.eq(torch.unsqueeze(y_true, dim=0),
                          torch.unsqueeze(y_shifting, dim=1))
        y_true = torch.where(y_true, torch.ones_like(
            y_true, dtype=torch.float), torch.zeros_like(y_true, dtype=torch.float))

        sim = F.cosine_similarity(x1=predict.unsqueeze(
            1), x2=predict.unsqueeze(0), dim=-1)
        sim = sim - torch.eye(predict.shape[0], device=device) * 1e12
        sim = sim / 0.05

        loss = F.cross_entropy(input=sim, target=y_true)
        return loss


class TimesModel(nn.Module):
    def __init__(self, in_size, hidden_size, dropout, bidirectional):
        super(TimesModel, self).__init__()
        self.liner = nn.Linear(in_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.rnn1 = nn.LSTM(hidden_size, hidden_size,
                            bidirectional=bidirectional)
        self.rnn2 = nn.LSTM(2 * hidden_size, hidden_size,
                            bidirectional=bidirectional)
        self.layer_norm = nn.LayerNorm((2 * hidden_size,))

    def forward(self, sequence, lengths):
        lengths = lengths.squeeze().int().detach().cpu().view(-1)
        batch_size = sequence.shape[0]
        sequence = self.dropout(self.liner(sequence))
        packed_sequence = pack_padded_sequence(
            sequence, lengths, batch_first=True, enforce_sorted=False)
        packed_h1, (final_h1, _) = self.rnn1(packed_sequence)
        padded_h1, _ = pad_packed_sequence(packed_h1)
        padded_h1 = padded_h1.permute(1, 0, 2)
        normed_h1 = self.layer_norm(padded_h1)
        packed_normed_h1 = pack_padded_sequence(
            normed_h1, lengths, batch_first=True, enforce_sorted=False)
        _, (final_h2, _) = self.rnn2(packed_normed_h1)
        utterance = torch.cat((final_h1, final_h2), dim=2).permute(
            1, 0, 2).contiguous().view(batch_size, -1)
        return utterance


class ProjectModel(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, intermediate_size):
        super(ProjectModel, self).__init__()
        self.drop = nn.Dropout(p=dropout)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.up_proj = nn.Linear(
            input_size, self.intermediate_size, bias=False)
        self.gate_proj = nn.Linear(
            self.intermediate_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        x = self.drop(x)
        x = F.silu(self.up_proj(x))
        x = F.silu(self.gate_proj(x))
        x = self.down_proj(x)
        return x


class MultiModal(nn.Module):
    def __init__(self, text_model_path, hidden_size, video_size, audio_size, dropout, intermediate_size):
        super(MultiModal, self).__init__()

        # text编码
        self.text_encoder = AutoModel.from_pretrained(text_model_path)

        # video 编码
        self.down_video_proj = nn.Linear(video_size, hidden_size, bias=False)
        self.video_times = TimesModel(
            in_size=hidden_size, hidden_size=hidden_size, dropout=dropout, bidirectional=True)
        self.video_project = ProjectModel(
            input_size=hidden_size * 4, hidden_size=hidden_size, dropout=dropout, intermediate_size=intermediate_size)

        # audio编码
        self.down_audio_proj = nn.Linear(audio_size, hidden_size, bias=False)
        self.audio_times = TimesModel(
            in_size=hidden_size, hidden_size=hidden_size, dropout=dropout, bidirectional=True)
        self.audio_project = ProjectModel(
            input_size=hidden_size * 4, hidden_size=hidden_size, dropout=dropout, intermediate_size=intermediate_size)

        self.drop = nn.Dropout(p=dropout)

        # 特征合一
        self.tva = nn.Linear(in_features=hidden_size *
                             3, out_features=hidden_size)
        self.ta = nn.Linear(in_features=hidden_size *
                            2, out_features=hidden_size)
        self.tv = nn.Linear(in_features=hidden_size *
                            2, out_features=hidden_size)

        # 模型输出
        self.output = nn.Linear(hidden_size, out_features=2)

        # loss
        self.mm_loss = MMLoss()

    def forward(self, input_ids: Tensor, token_type_ids: Tensor, attention_mask: Tensor, video_inputs: Tensor, audio_inputs: Tensor, video_length: Tensor, audio_length: Tensor):
        # 文本编码
        model_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        text_output = model_output.pooler_output

        # 音频编码
        audio_output = F.relu(self.down_audio_proj(audio_inputs))
        audio_output = self.audio_times(
            sequence=audio_output, lengths=audio_length)
        audio_output = self.audio_project(x=audio_output)

        # 视频编码
        video_output = F.relu(self.down_video_proj(video_inputs))
        video_output = self.video_times(
            sequence=video_output, lengths=video_length)
        video_output = self.video_project(x=video_output)

        tva = self.tva(torch.concat(
            [text_output, audio_output, video_output], dim=-1))
        ta = self.ta(torch.concat([text_output, audio_output], dim=-1))
        tv = self.tv(torch.concat([text_output, video_output], dim=-1))

        # ta tv 对比 loss
        in_batch_av = torch.stack([ta, tv], dim=1)
        in_batch_av = in_batch_av.reshape(-1, ta.shape[1])
        loss_av = self.mm_loss(in_batch_av)
        compare_loss = loss_av

        logits = self.output(tva)
        return logits, compare_loss
