import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import torch
import librosa
import numpy as np
import pandas as pd
from sqlalchemy import select
from tqdm import tqdm
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from dataset import TestDataset
from monsoon_biodiversity_common.config import cfg as CFG
from monsoon_biodiversity_common.model import AttModel
from monsoon_biodiversity_common.db_model import init_database, RpiDevices, SpeciesDetection


device = 'cuda' if torch.cuda.is_available() else 'cpu'
weight_path = os.getenv('AUDIO_CLASSIFIER_WEIGHTS')
if not weight_path:
    raise Exception("Environment variable AUDIO_CLASSIFIER_WEIGHTS not set.")
if not os.path.isfile(weight_path):
    raise Exception(f"Model weights file not found: {weight_path}")
state_dict = torch.load(weight_path, map_location='cpu', weights_only=False)['state_dict']

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
model.eval()
print("Model initialized successfully.")

APP_DATA_DIR = os.getenv('APP_DATA_DIR')
if not APP_DATA_DIR:
    raise Exception("Environment variable APP_DATA_DIR not set.")
if not os.path.isdir(APP_DATA_DIR):
    raise Exception(f"Directory {APP_DATA_DIR} not found.")
if APP_DATA_DIR.endswith('/'):
    APP_DATA_DIR = APP_DATA_DIR[:-1]

DATABASE_URL = f'sqlite:///{APP_DATA_DIR}/sql_app.db'
create_session = init_database(DATABASE_URL)


def prediction_for_clip(audio_path):
    """
    Process an audio file and generate predictions for each 5-second segment.

    This function loads an audio file, splits it into 5-second segments, and performs
    classification on each segment using a pre-trained model. The results include
    the predicted class and confidence score for each segment.

    Args:
        audio_path (str): Path to the audio file to be processed

    Returns:
        dict: A dictionary where:
            - Keys are segment identifiers in format "{filename}_dur={duration}secs_{second}"
            - Values are dictionaries containing:
                - "Class": Predicted class name (str)
                - "Score": Confidence score of the prediction (float)

    Example:
        >>> result = prediction_for_clip("path/to/audio.wav")
        >>> print(result)
        {
            'audio_dur=30secs_5': {'Class': 'bird', 'Score': 0.95},
            'audio_dur=30secs_10': {'Class': 'insect', 'Score': 0.87},
            ...

    Notes:
        - Audio is resampled to 32000Hz
        - Predictions are made every 5 seconds
        - {second} in the key start from 5 and increment by 5, so 5, 10, 15, ...
    """
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
                prediction_dict[row_id] = output[row_id_idx, :].sigmoid().cpu().numpy()

    for row_id in list(prediction_dict.keys()):
        logits = np.array(prediction_dict[row_id])
        # TODO: this doesn't filter out low-confidence predictions, add it?
        classification_dict[row_id] = {
            "Class": CFG.target_columns[np.argmax(logits)],
            "Score": float(np.max(logits))  # Convert to float for DB
        }

    return classification_dict


def save_to_database(pi_id, date, species_data):
    with create_session() as session:
        try:
            # Check if device record exists
            iot_device = session.execute(select(RpiDevices).where(RpiDevices.pi_id == pi_id)).first()

            if not iot_device:
                # Create new device record
                iot_device = RpiDevices(pi_id=pi_id)
                session.add(iot_device)
                session.flush()

            # Save species detection records
            for time_segment, detection_data in species_data.items():
                time_str = time_segment.split('_')[0]  # Extract HH-MM-SS
                hour, minute, second = map(int, time_str.split('-'))
                start_time = datetime.strptime(date, '%Y-%m-%d') + timedelta(hours=hour, minutes=minute, seconds=second)
                end_time = start_time + timedelta(seconds=5)

                species_detection = SpeciesDetection(
                    device_id=iot_device.id,
                    time_segment=time_segment,
                    start_time=start_time,
                    end_time=end_time,
                    species_class=detection_data['Class'],
                    confidence_score=detection_data['Score']
                )
                session.add(species_detection)

            session.commit()
            print(f"Successfully added data for {iot_device.pi_id} on {date}")

        except Exception as e:
            session.rollback()
            print(f"Error saving data: {str(e)}")


def process_new_audio(audio_path):
    print(f"Processing: {audio_path}")

    # Check if audio file exists and can be loaded
    if not os.path.isfile(audio_path):
        print(f"Audio file not found: {audio_path}")
        return

    classification_dict = prediction_for_clip(audio_path)
    print("Finished classify audio")

    # Extract IoT device name and date from the path
    iot_name = Path(audio_path).parts[-3]
    date = Path(audio_path).parts[-2]
    print(f"Extracted IoT name: {iot_name}, date: {date}")

    # Save directly to the database
    save_to_database(iot_name, date, classification_dict)


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


if __name__ == '__main__':
    audio_data_dir = os.getenv('AUDIO_DATA_DIR')
    if not audio_data_dir:
        raise Exception("Environment variable AUDIO_DATA_DIR not set.")

    if not os.path.isdir(audio_data_dir):
        raise Exception(f"Directory {audio_data_dir} not found.")

    monitor_directory(audio_data_dir)
