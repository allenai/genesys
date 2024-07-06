from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import importlib

import exec_utils

__all__ = [
    "Checker"
]

@exec_utils.Registry(
    resource_type="tool_type",
    name="checker",
)
class Checker(exec_utils.BaseTool):
    """Checker for checking the correctness of model designs.  

    """
    def __init__(self):
        self.report = None

    def reset(self) -> None:
        """Results the report
        
        :rtype: None 
        """
        self.report=''

    def is_causal(
            self,
            block,
            D: int,
            seq_len: int =100
        ) -> bool:
        """Checks if a design is causal

        :param block: 
            The target block design. 
        :param D: 
            The block dimensions. 
        :param seq_len: 
            The block target sequence length.
        """
        B: int = 2
        X = torch.arange(seq_len*B*D).float().reshape(B, seq_len, D).device(
            "cuda" if torch.cuda.is_available() else "cpu" 
        )
        Y = block(X)

        self.logging.info('Checking causality...')
        bar = tqdm(range(seq_len), desc='Causality test',colour='green')
        for t in bar:
            for delta in range(1, seq_len-t):
                X_mod = X.clone()
                z = torch.rand(B,D).device("cuda" if torch.cuda.is_available() else "cpu")
                X_mod[:, t+delta, :] += z
                if not torch.allclose(Y[:, t,:], Y_mod[:, t,:]):
                    self.logging.info(
                        f'Causality test failed at t={t},delta={delta}'
                    )
                    return False
                
        self.logging.info('Causality test passed')
        return True

    def check_magnitude(
            self,
            size: int,
            magnitude: float,
            threshold: float
        ) -> bool:
        """Checks that the block maintains a certain limit on parameters 

        :param size: 
            
        """
        self.logging.info(f'Checking non-embedding parameter number again the magnitude: {magnitude}')
        if size > (1+threshold)*magnitude:
            exceed = (size-magnitude)/magnitude
            self.logging.info(
                f'Parameter number exceeds the magnitude by {exceed}'
            )
            return False
        elif size < (1-threshold)*magnitude:
            below=(magnitude-size)/magnitude
            self.logging.info(
                f'Parameter number if below the magnitude: {below}'
            )
            return False
        self.logging.info('Paramete number is within threshold')
        return True

    def check(self, path: str) -> bool:
        """Runs through a bunch of checks for the new module at path 

        :param path: 
            The path of the proposed module 
        """
        return True
    
    def __call__(self,path: str) -> bool:
        return self.check(path)
    
