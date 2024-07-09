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

def reload_gam(config,gab_code: str,name: str = 'new'):
    """Reloads the GAM code with new block 

    :param gam_code: 
        The GAM code 
    :param gab_code: 
        The new GAB block 
    """
    module = {}
    exec(gab_code.replace("class GAB","class GABCustom"),module)
    GAB = module["GABCustom"]
    gab_config = {} 
    if "gab_config" in module:
        gab_config = module["gab_config"]()

    BlockRegister.add_block(name,GAB,config=gab_config)
    
    model = ModisLMHeadModel.from_config(
        config,
        gab_name="new",
    )

    return model 
        
    
    # code = create_code(gam_code,gab_code)
    # module_name = 'dynamic_gam_module'
    # spec = importlib.util.spec_from_loader(
    #     module_name,
    #     loader=None
    # )

    # module = importlib.util.module_from_spec(spec)
    
    # # Execute the code in the module's namespace
    # exec(code, module.__dict__)

    # # Add the module to sys.modules
    # sys.modules[module_name] = module
    
    # # Import the module dynamically
    # import dynamic_gam_module
    # return dynamic_gam_module.ModisLMHeadModel
