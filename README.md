# Production Workflow on AWS-EC2 for Biodiversity Application

This is the repository for the production workflow on AWS-EC2 for the Biodiversity application.

### System Workflow

![System Workflow](diagram.png)

## Microservices Overview

### 1. Monitoring

- **Purpose**: Continuously monitor new incoming audio files.
- **Instructions**:
  1. Navigate to the **Monitoring** folder.
  2. Execute the following command:
     ```bash
     ./audio_process.sh
     ```

### 2. Inference

- **Purpose**: Perform inference on the monitored audio files.
- **Instructions**:
  1. Navigate to the **Inference** folder.
  2. Start the Docker container by running:
     ```bash
     ./run_docker.sh
     ```
  3. Inside the container, run the inference microservice:
     ```bash
     python microservice.py
     ```

### 3. Database

- **Purpose**: Store the results of the inference.
- **Instructions**:
  - After executing `microservice.py`, the results from the 10-minute audio files will be saved in a MySQL database named `species.db`.

### 4. Feature Extraction (optional for future implementation)

- **Purpose**: Extract log-mel spectrogram features from audio files.
- **Instructions**:
  1. Navigate to the **Feature Extraction** folder.
  2. Run the feature extraction script:
     ```bash
     python extract_features.py
     ```
  3. The extracted features will be saved in `.npy` format for further analysis.


### 6. Monitoring & Logging

- **Purpose**: Monitor the entire workflow and log relevant events.
- **Instructions**:
  1. Navigate to the **Monitoring** folder.
  2. Use the logging tool:
     ```bash
     ./monitor_logs.sh
     ```
  3. Logs will be stored in the `logs/` directory for review.

### 7. Cleanup

- **Purpose**: Remove old audio files and results to free up storage.
- **Instructions**:
  1. Navigate to the root directory.
  2. Execute the cleanup script:
     ```bash
     ./cleanup.sh
     ```
