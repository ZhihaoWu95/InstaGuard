export EXPERIMENT_NAME="InstaGuard"
export MODEL_PATH="./ckpt/g_checkpoint.pth"
export INPUT_DIR="./data/n000050/set_B"
export OUTPUT_DIR="./output"
export CUDA_DEVICE="1" 

python instaguard.py \
  --input_path=$INPUT_DIR\
  --output_path=$OUTPUT_DIR\
  --cuda=$CUDA_DEVICE \
  --im_size=512 \
  --ckpt=$MODEL_PATH \
  

