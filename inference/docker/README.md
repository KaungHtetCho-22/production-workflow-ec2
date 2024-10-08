- Ensure you have a system with a GPU
- Run `build_docker.sh` to build the image
- Run `run_docker.sh` to start the container

While running `run_docker.sh`, if you provide a flag start-service as an argument, it will spawn the microservice for inference. By default, it is set up to run on the same network as the host, and expects the calls to be made to `http://127.0.0.1:5000/predict`. Modify according to your setup. 
