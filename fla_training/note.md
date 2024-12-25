
1. Preprocess the dataset



python preprocess.py \
  --dataset chengjunyan1/smollm-12.5-corpus \
  --name smollm-125 \
  --split train \
  --context_length 2048

2. Train the model


bash scripts/fla_train.sh \
  data=chengjunyan1/smollm-12.5-corpus \
  warmup=1024 \
  model=mamba2_350M
