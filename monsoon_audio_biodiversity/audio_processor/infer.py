import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import torch
from tqdm.auto import tqdm
import sys
from copy import copy
import importlib
from dataset import TestDataset
from monsoon_audio_biodiversity.ml_models.model import AttModel
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import json

warnings.filterwarnings("ignore")

ROOT_AUDIO_DIR = 'continuous_monitoring_data/live_data/'  
EXPORT_DIR = './exports/'  

sys.path.append('./configs')
CFG = copy(importlib.import_module("ait_bird_local").cfg)  # Load config file

device = 'cuda' if torch.cuda.is_available() else 'cpu'
state_dict = torch.load('./weights/ait_bird_local_eca_nfnet_l0/fold_0_model.pt', map_location=device)['state_dict']

# Initialize the model
model = AttModel(
    backbone=CFG.backbone,
    num_class=CFG.num_classes,
    infer_period=5,
    cfg=CFG,
    training=False,
    device=device
)

model.load_state_dict(state_dict)
model = model.to(device)
model.logmelspec_extractor = model.logmelspec_extractor.to(device)
print(model)

def prediction_for_clip(audio_path):
    prediction_dict = {}
    classification_dict = {}

    clip, _ = librosa.load(audio_path, sr=32000)
    duration = librosa.get_duration(y=clip, sr=32000)
    seconds = list(range(5, int(duration), 5))
    
    filename = Path(audio_path).stem
    row_ids = [filename + f"_dur={int(duration)}secs_{second}" for second in seconds]  # Include duration in row_id

    test_df = pd.DataFrame({
        "row_id": row_ids,
        "seconds": seconds,
    })

    dataset = TestDataset(
        df=test_df, 
        clip=clip,
        cfg=CFG,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        **CFG.loader_params['valid']
    )

    for i, inputs in enumerate(tqdm(loader)):
        row_ids = inputs['row_id']
        inputs.pop('row_id')

        with torch.no_grad():
            output = model(inputs)['logit']

        for row_id_idx, row_id in enumerate(row_ids):
            prediction_dict[str(row_id)] = output[row_id_idx, :].sigmoid().detach().cpu().numpy()

    for row_id in list(prediction_dict.keys()):
        logits = np.array(prediction_dict[row_id])
        classification_dict[row_id] = {
            "Class": CFG.target_columns[np.argmax(logits)],
            "Score": float(np.max(logits))  # Convert to float for JSON compatibility
        }

    return classification_dict

def process_new_audio(audio_path, export_dir):
    print(f"Processing: {audio_path}")
    classification_dict = prediction_for_clip(audio_path)
    
    iot_name = Path(audio_path).parts[-3]  # Get IoT device ID from the path
    date = Path(audio_path).parts[-2]  # Get date from the path
    file_stem = Path(audio_path).stem

    # Create export path
    export_subdir = os.path.join(export_dir, iot_name, date)
    os.makedirs(export_subdir, exist_ok=True)

    # Save as JSON in the desired format
    result_data = {
        "pi_id": iot_name,
        "date": date,
        "species": classification_dict  # Species classification per segment
    }
    
    export_path = os.path.join(export_subdir, file_stem + "_results.json")
    with open(export_path, 'w') as json_file:
        json.dump(result_data, json_file, indent=4)
    
    print(f"Done! Exported predictions for {audio_path} to {export_path}")

class AudioFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return None
        else:
            if event.src_path.endswith(('.wav', '.ogg', '.mp3')):
                process_new_audio(event.src_path, EXPORT_DIR)

def monitor_directory(root_dir):
    event_handler = AudioFileHandler()
    observer = Observer()
    observer.schedule(event_handler, path=root_dir, recursive=True)
    observer.start()

    print(f"Monitoring directory: {root_dir}")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

monitor_directory(ROOT_AUDIO_DIR)
