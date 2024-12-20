from fla.models import Mamba2Config

from model_discovery.configs.gam_config import GAMConfig_1300M


Mamba2Config_1300M = Mamba2Config()




def hf_config_from_args(hf_config):
    if hf_config == 'none':
        return None
    model_type,scale = hf_config.split('_')
    if model_type == 'Mamba2Config':
        if scale == '1300M':
            return Mamba2Config_1300M,GAMConfig_1300M()
        else:
            raise ValueError(f"Unsupported scale for Mamba2Config: {scale}")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
