import librosa
import numpy as np
import onnx
import onnxsim
import torch
import torch.onnx
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import onnxruntime

from monsoon_audio_biodiversity.ml_models.model import InferenceAudioClassifierModel, NormalizeMelSpec
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

torch.onnx.export(model, transformed_input[:, None], 'audio_classifier_model.onnx',
                  input_names=['input0'], output_names=['output0'],
                  training=torch.onnx.TrainingMode.EVAL)

onnx_model = onnx.load('audio_classifier_model.onnx')
simplified_onnx_model, check = onnxsim.simplify(onnx_model)
onnx.save(simplified_onnx_model, 'audio_classifier_model_simplified.onnx')

assert check, 'Failed to simplify ONNX model'

# Compare results between PyTorch and ONNX models

# Get PyTorch model prediction
torch_output = model(transformed_input[:, None])

# Get ONNX model prediction
ort_session = onnxruntime.InferenceSession('audio_classifier_model_simplified.onnx')
ort_inputs = {ort_session.get_inputs()[0].name: transformed_input[:, None].numpy()}
ort_output = ort_session.run(None, ort_inputs)[0]

# Convert outputs to numpy for comparison
torch_output_np = torch_output.numpy()
ort_output_np = ort_output

# Compare the results
np.testing.assert_allclose(torch_output_np, ort_output_np, rtol=1e-5)
print("PyTorch and ONNX model outputs match within tolerance!")

# Print sample predictions
print("\nSample predictions:")
print(f"PyTorch output: {torch_output_np[0][:5]}")  # Show first 5 predictions
print(f"ONNX output: {ort_output_np[0][:5]}")
