import librosa
import numpy as np
import onnx
import torch
import torch.onnx
import torch.nn as nn
import tvm
import tvm.relax as relax
from tvm.relax.frontend.onnx import from_onnx
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

from monsoon_audio_biodiversity.ml_models.model import NormalizeMelSpec
from monsoon_audio_biodiversity.audio_processor.configs.ait_bird_local import cfg as CFG


sample_rate = 32000
audio_clip = librosa.load('data/audio/random-RPi-ID/2024-10-21/10-11-54_dur=600secs.wav', sr=sample_rate)
# 5 seconds of audio
inp_clip = torch.from_numpy(audio_clip[0][:sample_rate * 5]).unsqueeze(0)
device = 'cpu'

logmelspec_extractor = nn.Sequential(
    MelSpectrogram(
        sample_rate=CFG.sample_rate,
        n_mels=CFG.n_mels,
        f_min=CFG.fmin,
        f_max=CFG.fmax,
        n_fft=CFG.n_fft,
        hop_length=CFG.hop_length,
        normalized=True,
    ),
    AmplitudeToDB(top_db=80.0),
    NormalizeMelSpec(),
)

transformed_input = logmelspec_extractor(inp_clip)

simplified_onnx_model = onnx.load("audio_classifier_model_simplified.onnx")

shape_list = {'input0': (1, 1, 128, 313)}
mod = from_onnx(simplified_onnx_model, shape_list)

target = tvm.target.Target("llvm -num-cores 8")

TOTAL_TRIALS = 8000

mod = relax.get_pipeline("static_shape_tuning", target=target, total_trials=TOTAL_TRIALS)(mod)

ex = relax.build(mod, target)
ex.export_library('llvm_8_cores_audio_classifier_model.so')

device = tvm.device('cpu')
vm = relax.VirtualMachine(ex, device)
tmv_data = tvm.nd.array(transformed_input[:, None].numpy())
time_ret = vm.time_evaluator("main", device, number=10)(tmv_data)
print(f'time_ret: {time_ret}')
