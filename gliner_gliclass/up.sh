
# Set up paths for persistent, non-temporary storage
export MODEL_REPO_PATH=$(pwd)/model_repository
export HF_CACHE_PATH=$(pwd)/hf_cache
export VLLM_CACHE_PATH=$(pwd)/vllm_cache

# Create the directories on your host machine
mkdir -p $HF_CACHE_PATH
mkdir -p $VLLM_CACHE_PATH

docker build -t triton-gliner:gliclass . \
  && docker run --runtime=nvidia --gpus all --shm-size 1G --rm -it \
  -p 8000:8003 \
  -p 8001:8004 \
  -p 8002:8005 \
  -e HF_HOME=/hf-cache \
  -v $MODEL_REPO_PATH:/models \
  -v $HF_CACHE_PATH:/hf-cache \
  -v $VLLM_CACHE_PATH:/root/.cache \
  -v /tmp:/tmp \
  -e CUDA_LAUNCH_BLOCKING=1 \
  triton-gliner:gliclass tritonserver --log-verbose=2 --model-repository=/models
