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
from model import AttModel

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time

warnings.filterwarnings("ignore")

# Hardcoded paths
AUDIO_DIR = 'continuous_monitoring_data/live_data/RPiID-00000000b36010d2/2024-09-25'  # Directory where audio files are stored
EXPORT_DIR = './exports/'          # Directory to save results

# Hardcoded configuration settings for the model
sys.path.append('./configs')
CFG = copy(importlib.import_module("ait_bird_local").cfg)  # Load config file

# Define the device (use 'cpu' as default)
device = torch.device("cpu")
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

    # Load audio file with sample rate from config
    clip, _ = librosa.load(audio_path, sr=32000)

    # Get the duration of the clip in seconds and calculate intervals
    duration = librosa.get_duration(y=clip, sr=32000)
    seconds = list(range(5, int(duration), 5))  # Ensure it covers the whole audio length
    
    filename = Path(audio_path).stem
    
    # Generate row ids for each segment
    row_ids = [filename + f"_{second}" for second in seconds]

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

    # Process the results into dicts
    for row_id in list(prediction_dict.keys()):
        logits = np.array(prediction_dict[row_id])
        prediction_dict[row_id] = {}
        classification_dict[row_id] = {}
        for label in range(len(CFG.target_columns)):
            prediction_dict[row_id][CFG.target_columns[label]] = logits[label]
            classification_dict[row_id]['Class'] = CFG.target_columns[np.argmax(logits)]
            classification_dict[row_id]['Score'] = np.max(logits)

    return prediction_dict, classification_dict

def process_new_audio(audio_path, export_dir):
    """Process the newly detected audio file and save the results."""
    print(f"Processing: {audio_path}")

    # Perform inference on the current audio file
    prediction_dict, classification_dict = prediction_for_clip(audio_path)

    # Save logit predictions
    logit_df = pd.DataFrame.from_dict(prediction_dict, "index").rename_axis("row_id").reset_index()
    logit_export_path = os.path.join(export_dir, Path(audio_path).stem + "_logits.csv")
    logit_df.to_csv(logit_export_path, index=False)

    # Save classification results
    classification_df = pd.DataFrame.from_dict(classification_dict, "index").rename_axis("row_id").reset_index()
    classification_export_path = os.path.join(export_dir, Path(audio_path).stem + "_classification.csv")
    classification_df.to_csv(classification_export_path, index=False)

    print(f"Done! Exported predictions for {audio_path} to {export_dir}")

class AudioFileHandler(FileSystemEventHandler):
    """Custom handler to process new audio files."""
    def on_created(self, event):
        if event.is_directory:
            return None
        else:
            # Process the file if it's an audio file
            if event.src_path.endswith(('.wav', '.ogg', '.mp3')):
                process_new_audio(event.src_path, EXPORT_DIR)

def monitor_directory(audio_dir):
    """Monitor the directory for new audio files."""
    event_handler = AudioFileHandler()
    observer = Observer()
    observer.schedule(event_handler, path=audio_dir, recursive=False)
    observer.start()

    print(f"Monitoring directory: {audio_dir}")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# Start monitoring the audio directory
monitor_directory(AUDIO_DIR)