#!/usr/bin/env python
# -*- coding: utf-8 -*-


import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from moviepy import AudioFileClip
import soundfile as sf

audio_clip = AudioFileClip(
    "/home/chenghuadong.chd/multimodal_recognition/src/resources/data/Raw/test2/00000.mp4")
audio_clip.write_audiofile(
    "/home/chenghuadong.chd/multimodal_recognition/src/resources/data/Raw/test2/00000.wav")

print(audio_clip)


processor = Wav2Vec2Processor.from_pretrained(
    "/home/chenghuadong.chd/.cache/modelscope/hub/jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn")
model = Wav2Vec2ForCTC.from_pretrained(
    "/home/chenghuadong.chd/.cache/modelscope/hub/jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn")

speech_array, sampling_rate = librosa.load(
    "/home/chenghuadong.chd/multimodal_recognition/src/resources/data/Raw/test2/00000.wav", sr=16_000)
inputs = processor(speech_array, sampling_rate=16_000,
                   return_tensors="pt", padding=True)
audio_output = model(inputs.input_values,
                     attention_mask=inputs.attention_mask, output_hidden_states=True)
print(audio_output)
