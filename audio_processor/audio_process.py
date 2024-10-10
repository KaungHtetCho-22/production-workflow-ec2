import os
import warnings
from pathlib import Path
import numpy as np
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
from datetime import datetime, timedelta

# Import database setup and models
from model import create_engine_and_session, RpiDevices, SpeciesDetection

warnings.filterwarnings("ignore")

ROOT_AUDIO_DIR = 'continuous_monitoring_data/live_data/'  # Root directory to monitor

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

# Create engine and session for database
engine, session = create_engine_and_session()

def prediction_for_clip(audio_path):
    prediction_dict = {}
    classification_dict = {}

    clip, _ = librosa.load(audio_path, sr=32000)
    duration = librosa.get_duration(y=clip, sr=32000)
    seconds = list(range(5, int(duration), 5))
    
    filename = Path(audio_path).stem
    row_ids = [filename + f"_dur={int(duration)}secs_{second}" for second in seconds]

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
            "Score": float(np.max(logits))  # Convert to float for DB
        }

    return classification_dict

def save_to_database(pi_id, date, species_data, session):
    try:
        # Check if device record exists
        existing_device = session.query(RpiDevices).filter_by(
            pi_id=pi_id,
            analysis_date=datetime.strptime(date, '%Y-%m-%d').date()
        ).first()

        if existing_device:
            print(f"Device record already exists for {pi_id} on {date}. Skipping insertion.")
            return

        # Create new device record
        new_device = RpiDevices(
            pi_id=pi_id,
            analysis_date=datetime.strptime(date, '%Y-%m-%d')
        )
        session.add(new_device)
        session.flush()

        # Save species detection records
        for time_segment, detection_data in species_data.items():
            time_str = time_segment.split('_')[0]  # Extract HH-MM-SS
            hour, minute, second = map(int, time_str.split('-'))
            start_time = datetime.strptime(date, '%Y-%m-%d') + timedelta(hours=hour, minutes=minute, seconds=second)
            end_time = start_time + timedelta(seconds=5)

            species_detection = SpeciesDetection(
                device_id=new_device.id,
                time_segment=time_segment,
                start_time=start_time,
                end_time=end_time,
                species_class=detection_data['Class'],
                confidence_score=detection_data['Score']
            )
            session.add(species_detection)

        session.commit()
        print(f"Successfully added data for {new_device.pi_id} on {new_device.analysis_date}")

    except Exception as e:
        session.rollback()
        print(f"Error saving data: {str(e)}")
    finally:
        session.close()

def process_new_audio(audio_path):
    print(f"Processing: {audio_path}")
    classification_dict = prediction_for_clip(audio_path)
    
    # Extract IoT device name and date from the path
    iot_name = Path(audio_path).parts[-3]
    date = Path(audio_path).parts[-2]

    # Save directly to the database
    save_to_database(iot_name, date, classification_dict, session)

class AudioFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return None
        else:
            if event.src_path.endswith(('.wav', '.ogg', '.mp3')):
                process_new_audio(event.src_path)

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
