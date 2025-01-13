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

from monsoon_audio_biodiversity.ml_models.model import NormalizeMelSpec, InferenceAudioClassifierModel
from monsoon_audio_biodiversity.audio_processor.configs.ait_bird_local import cfg as CFG


torch.set_grad_enabled(False)

WEIGHT_PATH = 'weights/ait_bird_local_eca_nfnet_l0/fold_0_model.pt'

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

model = InferenceAudioClassifierModel(
    backbone=CFG.backbone,
    num_class=CFG.num_classes,
    infer_period=5,
    cfg=CFG,
    training=False,
    device=device
)
state_dict = torch.load(WEIGHT_PATH, map_location='cpu', weights_only=False)['state_dict']
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
print(f'missing_keys: {missing_keys}')
# allow unexpected keys from logmelspec_extractor
print(f'unexpected_keys: {unexpected_keys}')
assert len(missing_keys) == 0, f'missing_keys: {unexpected_keys}'
assert len(unexpected_keys) == 2, f'unexpected_keys is more than 2: {unexpected_keys}'
model.eval()

transformed_input = logmelspec_extractor(inp_clip)

ex = tvm.runtime.load_module('llvm_8_cores_audio_classifier_model.so')
device = tvm.device('cpu')
vm = relax.VirtualMachine(ex, device)
tmv_data = tvm.nd.array(transformed_input[:, None].numpy())
time_ret = vm.time_evaluator("main", device, number=10)(tmv_data)
print(f'Model inference time: {time_ret}')

# test whether the tvm model give the same result as the pytorch model
pytorch_output = model(transformed_input[:, None])
tvm_output = vm['main'](tmv_data).numpy()

np.testing.assert_allclose(pytorch_output, tvm_output, rtol=1e-5)
print("PyTorch and TVM model outputs match within tolerance!")

# Print sample predictions
print("\nSample predictions:")
print(f"PyTorch output: {pytorch_output[0][:5]}")  # Show first 5 predictions
print(f"TVM output: {tvm_output[0][:5]}")
