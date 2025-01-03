import time

import librosa
import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

from monsoon_audio_biodiversity.ml_models.model import InferenceAudioClassifierModel, NormalizeMelSpec
from monsoon_audio_biodiversity.audio_processor.configs.ait_bird_local import cfg as CFG


sample_rate = 32000
audio_clip = librosa.load('data/audio/random-RPi-ID/2024-10-21/10-11-54_dur=600secs.wav', sr=sample_rate)
# 5 seconds of audio
inp_clip = torch.from_numpy(audio_clip[0][:sample_rate * 5]).unsqueeze(0)

device = 'cpu'

logmelspec_extractor = torch.jit.script(nn.Sequential(
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
), example_inputs=[(torch.randn(1, 32000 * 5))])

model = InferenceAudioClassifierModel(
    backbone=CFG.backbone,
    num_class=CFG.num_classes,
    infer_period=5,
    cfg=CFG,
    training=False,
    device=device
)
model.eval()
print("Model initialized successfully.")

# Apply JIT compilation
try:
    logmelspec_extractor = torch.jit.script(logmelspec_extractor, example_inputs=[inp_clip])
    example_trace_inputs = logmelspec_extractor(inp_clip)
    model = torch.jit.trace_module(model, inputs={'forward': example_trace_inputs[:, None]})
    print("Model JIT-compiled successfully.")
except Exception as e:
    print(f"Failed to JIT-compile the model: {e}")
    raise

with torch.no_grad():
    # Time total pipeline
    start_total = time.perf_counter_ns()
    
    # Time logmelspec extraction
    start_melspec = time.perf_counter_ns()
    mel_features = logmelspec_extractor(inp_clip)
    end_melspec = time.perf_counter_ns()
    print(f"Melspec extraction time: {(end_melspec-start_melspec)/1e6} milliseconds")

    # Time model inference
    start_model = time.perf_counter_ns() 
    output = model(mel_features[:, None])
    end_model = time.perf_counter_ns()
    print(f"Model inference time: {(end_model-start_model)/1e6} milliseconds")
    
    end_total = time.perf_counter_ns()
    print(f"Total pipeline time: {(end_total-start_total)/1e6} milliseconds")
    print(output.size())
