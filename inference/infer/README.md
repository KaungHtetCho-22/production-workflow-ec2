## Inference Instructions

To perform inference with your trained model, follow these steps after successfully training your classifier as described in the [train_audio](../train_audio) section.

### Running Inference

Use the `infer.py` script to predict species from audio files. This script requires several command line arguments:

- **config**: The configuration filename.
- **weight**: Path to the trained model weights.
- **audio**: Path to the audio file for inference.
- **export**: Directory path where the output CSV files will be saved.

### Example Command
```bash
python3 infer.py --config ait_bird_local --weight ./weights/ait_bird_local_eca_nfnet_l0/fold_0_model.pt --audio ./data/soundscape_29201.ogg --export ./exports/
```
This command will run inference on the audio file `./data/soundscape_29201.ogg` using the model stored at `./weights/ait_bird_local_eca_nfnet_l0/fold_0_model.pt`. The results will be exported to the `./exports/` directory.

### Download Pretrained Weights
Download the [pretrained weights](https://drive.google.com/drive/folders/16oEEUIQ_Vk643FvqOJCPdeKUgTV1dxrP?usp=sharing) and save them inside the weights folder.

### Output Details
Two CSV files are generated for each audio file:
- **Logits CSV**: Contains the logits for all species, named with the `_logits` suffix.
- **Classification CSV**: Contains the predicted class and score for each segment, named with the `_classification` suffix.

### Understanding the Output
Each row in the output CSVs corresponds to a 5-second segment of the audio data. The `Row_id` column indicates the specific time segment:
- For instance, the suffix `_5` refers to the first 5 seconds of the audio.

### Microservice 
For running the microservice, the default setup is included in the `microservice.py` file. Changes to the model, configuration file, or logic should be included in this file. When starting the container with the command
```bash
run_docker.sh start-service 
```
a container instance is spawned with the microservice enabled. Using default settings, and runnign the container on the same network as the host, the example from `test_microservice.py` can be followed to test inference. 

The microservice expects an audio file and returns the 5-second interval predictions in json format.  