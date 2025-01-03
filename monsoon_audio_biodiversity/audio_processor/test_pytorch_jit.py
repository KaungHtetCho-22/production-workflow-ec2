import time

import librosa
import torch

from monsoon_audio_biodiversity.ml_models.model import InferenceAudioClassifierModel
from monsoon_audio_biodiversity.audio_processor.configs.ait_bird_local import cfg as CFG


sample_rate = 32000
audio_clip = librosa.load('data/audio/random-RPi-ID/2024-10-21/10-11-54_dur=600secs.wav', sr=sample_rate)
# 5 seconds of audio
inp_clip = torch.from_numpy(audio_clip[0][:sample_rate * 5]).unsqueeze(0)

device = 'cpu'

model = InferenceAudioClassifierModel(
    backbone=CFG.backbone,
    num_class=CFG.num_classes,
    infer_period=5,
    cfg=CFG,
    training=False,
    device=device
)

model = model.to(device)
model.logmelspec_extractor = model.logmelspec_extractor.to(device)
model.eval()
print("Model initialized successfully.")

# Apply JIT compilation
try:
    model = torch.jit.script(model, example_inputs=[inp_clip])
    print("Model JIT-compiled successfully.")
except Exception as e:
    print(f"Failed to JIT-compile the model: {e}")
    raise

with torch.no_grad():
    start = time. perf_counter_ns()
    output = model(inp_clip)
    end = time. perf_counter_ns()
    print(f"Time taken: {(end-start)/1e6} miliseconds")
    print(output.size())
