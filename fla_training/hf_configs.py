from fla.models.mamba2 import Mamba2Config

from model_discovery.configs.gam_config import GAMConfig_1300M,GAMConfig_350M


# Mamba2Configs: https://huggingface.co/state-spaces 

Mamba2Config_1300M = Mamba2Config()
Mamba2Config_350M = Mamba2Config(
    state_size = 256,
    num_heads = 16,
    head_dim = 64,
    hidden_size=1024,
    num_hidden_layers = 48,
    expand = 0,
    tie_word_embeddings = True,
)




def hf_config_from_args(hf_config):
    if hf_config == 'none':
        return None
    model_type,scale = hf_config.split('_')
    if model_type == 'Mamba2Config':
        if scale == '1300M':
            model_cfg,cfg= Mamba2Config_1300M,GAMConfig_1300M()
        elif scale == '350M':
            model_cfg,cfg= Mamba2Config_350M,GAMConfig_350M()
        else:
            raise ValueError(f"Unsupported scale for Mamba2Config: {scale}")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model_cfg.vocab_size = cfg.vocab_size

    return model_cfg,cfg
