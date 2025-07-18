import importlib.util
import sys
from .block_registry import BlockRegister
from .gam import ModisLMHeadModel

def create_code(gam_code: str,gab_code: str) -> str:
    """Creates code object by merging together both modules 

    :param gam_code: 
        The underlying model code 
    :param gab_code: 
        The new modeling autoregressive block. 
    """
    return gab_code+'\n\n\n'+gam_code.replace('from modis_gam.model.gab import GAB, gab_config','')

def reload_gam(config,gab_code: str,name: str = 'new',autocfg={},**kwargs):
    """Reloads the GAM code with new block 

    :param gab_code: 
        The new GAB block. 
    :param name: 
        The name of the new block.
    
    """
    module = {}
    exec(gab_code.replace("class GAB","class GABCustom"),module)
    assert "GABCustom" in module, "Class GAB not found in module. You should never ever change the class name of GAB and it should always inherit from GABBase."
    GAB = module["GABCustom"]
    gab_config = {} 
    assert "gab_config" in module, "Dictionary gab_config not found in module."
    gab_config = module["gab_config"]
    gab_config.update(autocfg)

    ### register the new block  
    BlockRegister.add_block(
        name,
        GAB,
        config=gab_config
    )
    ### load it 
    # model = ModisLMHeadModel.from_config(
    #     config,
    #     gab_name=name,
    #     **kwargs
    # )
    model = ModisLMHeadModel(
        config, GAB, 
        block_config=gab_config,
        **kwargs
    ) # seems should not be bf16 for tf32 mode

    return model,gab_config
