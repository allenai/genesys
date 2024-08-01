import torch
import sys
import logging
from typing import Optional, Union  

import transformers
from transformers import AutoTokenizer

from ..model.gam import ModisLMHeadModel
from ..configs.gam_config import ( 
    GAMConfig,GAMConfig_14M,GAMConfig_31M,GAMConfig_70M,GAMConfig_125M,GAMConfig_350M,GAMConfig_760M,
    GAMConfig_1300M,GAMConfig_2700M,GAMConfig_6700M,GAMConfig_13B,GAMConfig_175B,GAMConfig_1T,GAMConfig_debug
)

from lm_eval.models.huggingface import HFLM
from lm_eval.api.registry import register_model
from lm_eval.__main__ import cli_evaluate

from .. import utils as U

util_logger = logging.getLogger('model_discovery.evals.evaluator')

from lm_eval import utils

eval_logger = utils.eval_logger



@register_model("modis")
class ModisEvalWrapper(HFLM):

    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(
            self,
            pretrained,
            # gab_name,
            gab,
            gab_config,
            ckpt_dir,
            max_length=2048,
            batch_size: Optional[Union[int, str]] = 1,
            max_batch_size: Optional[int] = 64,
            logits_cache: bool = True,
            truncation: Optional[bool] = False,
            device="cuda",
            dtype=torch.float16,
            **kwargs
        ):
        assert isinstance(batch_size, (int, str))

        util_logger.info('Running evaluation')
        
        b = pretrained.split("/")
        design_id = b[-1]
        scale = b[-2]
        evoname = b[-3]
        
        config = eval(f"GAMConfig_{scale}()")
        ckpt = U.pjoin(ckpt_dir, evoname, 've', design_id, "pretrained")
        
        util_logger.info(f'Trying to load from {ckpt}')
        
        # model = ModisLMHeadModel.from_pretrained(
        #     pretrained_model_name=ckpt,
        #     gab_name=gab_name,
        #     config=config,
        #     dtype=torch.bfloat16,
        #     device="cuda" if torch.cuda.is_available() else "cpu" 
        # )

        
        model = ModisLMHeadModel(
            config, gab, dtype=torch.bfloat16, device="cuda",
            block_config=gab_config
        ) # seems should not be bf16 for tf32 mode
        
        model.backbone.print_size()
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)   
        tokenizer.pad_token_id = tokenizer.eos_token_id
        self._config = config

        super().__init__(model,tokenizer=tokenizer)
        self.vocab_size = self.tokenizer.vocab_size
        self._max_length = max_length
        self.logits_cache = logits_cache
        self.truncation = truncation

        self.batch_schedule = 1
        self.batch_sizes = {}
        self.max_batch_size = max_batch_size

        if str(batch_size).startswith("auto"):
            batch_size = batch_size.split(":")
            self.batch_size_per_gpu = batch_size[0]
            self.batch_schedule = float(batch_size[1]) if len(batch_size) > 1 else 1
        else:
            self.batch_size_per_gpu = int(batch_size)


    @property
    def batch_size(self):
        return self.batch_size_per_gpu
    
    # @property
    # def batch_size(self):
    #     return self._batch_size

    # def _model_generate(self, context, max_length, stop, **generation_kwargs):
    #     raise NotImplementedError()

