import argparse
import os
import librosa
import onnx
import torch
import tvm
import tvm.relax as relax
from tvm.relax.frontend.onnx import from_onnx

from monsoon_biodiversity_common.model import get_waveform_transform
from monsoon_biodiversity_common.config import cfg as CFG


parser = argparse.ArgumentParser()
parser.add_argument('--target', choices=['cpu', 'rasp3b'], default='cpu',
                    help='Target processor to compile for (default: cpu)')
parser.add_argument('--n-cpu-cores', type=int, default=8,
                    help='Target number of CPU cores to use for compilation, ignore when cpu is rasp3b (default: 8)')
parser.add_argument('--n-trials', type=int, default=8000,
                    help='Target number of trials to use for static shape tuning (default: 8000)')
parser.add_argument('--onnx-model', type=str, default='audio_classifier_model_simplified.onnx',
                    help='Path to the ONNX model to compile (default: audio_classifier_model_simplified.onnx)')
parser.add_argument('--output-dir', type=str, default='./',
                    help='Path to the directory to save the compiled model (default: ./)')
args = parser.parse_args()

sample_rate = 32000
audio_clip = librosa.load('data/audio/random-RPi-ID/2024-10-21/10-11-54_dur=600secs.wav', sr=sample_rate)
# 5 seconds of audio
inp_clip = torch.from_numpy(audio_clip[0][:sample_rate * 5]).unsqueeze(0)
torch_device = 'cpu'

logmelspec_extractor = get_waveform_transform(CFG, torch_device)

transformed_input = logmelspec_extractor(inp_clip)

simplified_onnx_model = onnx.load(args.onnx_model)

shape_list = {'input0': (1, 1, 128, 313)}
mod = from_onnx(simplified_onnx_model, shape_list)

export_kwargs = {}
if args.target == 'cpu':
    target = tvm.target.Target(f'llvm -num-cores {args.n_cpu_cores}')
elif args.target == 'rasp3b':
    # must use all 4 cores to get maximum speed
    target = 'llvm -mtriple=aarch64-linux-gnu -mattr=+neon -model=bcm2837 -num-cores 4'
    export_kwargs = {'options': ['-fuse-ld=lld', '-mfpu=neon', '-mcpu=bcm2837', '-target', 'aarch64-linux-gnu', '-v']}

mod = relax.get_pipeline("static_shape_tuning", target=target, total_trials=args.n_trials)(mod)

ex = relax.build(mod, target)

cpu_cores_txt = f'-{args.n_cpu_cores}' if args.target == 'cpu' else ''
output_file_name = f'{args.target}{cpu_cores_txt}_audio_classifier_model.so'
output_file_path = os.path.join(args.output_dir, output_file_name)

ex.export_library(output_file_path, workspace_dir='export-workdir', **export_kwargs)
print(f"TVM model exported successfully to {output_file_path}!")
