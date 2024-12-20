## Test

bash scripts/run_ve.sh \
  --mode test \
  --resume



## Mamba2-1.3B

bash scripts/run_ve.sh \
  --mode train \
  --hf_config Mamba2Config_1300M \
  --design_id mamba-1300M \
  --scale 1300M \
  --evoname MODEL_BASELINES \
  --gradient_accumulation_steps 16 \
  --training_token_multiplier 100 \
  --resume







