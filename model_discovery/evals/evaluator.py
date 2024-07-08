import torch
import sys

import transformers
from transformers import AutoTokenizer

from ..model.gam import ModisLMHeadModel
from ..model.configs.gam_config import GAMConfig, GAMConfig_10M, GAMConfig_debug

from lm_eval.api.model import LM
from lm_eval.models.huggingface import HFLM
from lm_eval.api.registry import register_model
from lm_eval.__main__ import cli_evaluate

from .. import utils as U


@register_model("modis")
class ModisEvalWrapper(HFLM):

    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(
            self,
            pretrained,
            max_length=2048,
            batch_size=None,
            device="cuda",
            dtype=torch.float16, **kwargs
        ):
        # LM.__init__(self)
        # self._model = ModisLMHeadModel.from_pretrained(pretrained, device=device, dtype=dtype)
        # e.g. pretrained="GAMConfig_10M/GPT-3"
        print("Run Evaluation")

        configname, modelname = pretrained.split("/")
        config = eval(f"{configname}")
        ckpt = U.pjoin("ckpts", configname, modelname, "pretrained")
        print(f"Trying to load from {ckpt}")
        
        model = ModisLMHeadModel.from_pretrained(
            pretrained_model_name=ckpt,
            config=config,
            dtype=torch.bfloat16,
            device="cuda" if torch.cuda.is_available() else "cpu" 
        )
        
        model.backbone.print_size()
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)   
        tokenizer.pad_token_id = tokenizer.eos_token_id

        super().__init__(model,tokenizer=tokenizer)
        # self._model = model
        self.vocab_size = self.tokenizer.vocab_size
        self._batch_size = int(batch_size) if batch_size is not None else 64
        self._max_length = max_length
        # self._device = torch.device(device)

    @property
    def batch_size(self):
        return self._batch_size

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        raise NotImplementedError()


if __name__ == "__main__":
    sys.argv = [
        "eval.py",
        "--model", "modis",
        "--model_args", "pretrained=GAMConfig_10M/GPT-3",
        "--tasks", "lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande,blimp_filtered,blimp_supplement",
        "--device", "cuda",
        "--batch_size", "256",
        # "--mixed_precision", "yes"
    ]
    cli_evaluate()
