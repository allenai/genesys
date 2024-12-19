
1. Preprocess the dataset



python preprocess.py \
  --dataset chengjunyan1/smollm-12.5-corpus \
  --name smollm-125 \
  --split train \
  --context_length 2048


bash train.sh \
  type=gla \
  lr=3e-4 \
  steps=20480 \
  batch=8 \
  update=1 \
  warmup=1024 \
  context=2048 \
  path=exp/gla-340M-10B \
  project=fla \
  model=configs/gla_340M.json \
  data=HuggingFaceFW/fineweb-edu \
  name=sample-10BT \
  cache=data/HuggingFaceFW/fineweb-edu/sample-10BT/train

  