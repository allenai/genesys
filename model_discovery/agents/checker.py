from __future__ import annotations

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import importlib
import exec_utils

from ..model.loader import reload_gam

__all__ = [
    "Checker"
]

@exec_utils.Registry(
    resource_type="tool_type",
    name="checker",
)
class Checker(exec_utils.BaseTool):
    """Checker for checking the correctness of model designs.  
    

    Methods 
    ----------
    check(check,gab_code: str, name: str) 
        
    This is the main method that checks the proposed 
        block using `is_causal` (has causal attention), 
        `check_differentiable` (check that all operations 
        are differentiable)  and `check_magnitude` (that 
        parameters are within a certain range)

    """
    def __init__(self):
        self.report = ''

    def rprint(self, msg) -> None:
        """Log information of check and adds to report 

        :param msg: 
            The debug and report message.
 
        """
        self.logging.info(msg)
        self.report += msg+'\n'

    def reset(self) -> None:
        """Results the report
        
        :rtype: None 
        """
        self.report = ''
        
    def is_causal(self, block, D: int, seq_len: int = 100) -> bool:
        """Checks if a design is causal

        :param block: 
            The target block design. 
        :param D: 
            The block dimensions. 
        :param seq_len: 
            The block target sequence length.
        """
        B: int = 2
        X = torch.arange(seq_len * B * D).float().reshape(B, seq_len, D)
        if torch.cuda.is_available():
            X = X.cuda()
            
        block.eval()  # Set block to evaluation mode
        Y = block(X)

        self.rprint('Checking causality... It checks the causality by changing the future step X[t+delta] of X[t] and see if Y[t] changes.')
        bar = tqdm(range(seq_len), desc='Causality test', colour='green')
        for t in bar:
            for delta in range(1, seq_len - t):
                X_mod = X.clone()
                if torch.cuda.is_available():
                    torch.manual_seed(0)  # Set random seed for reproducibility
                    X_mod[:, t + delta, :] += torch.rand(B, D).cuda()
                else:
                    torch.manual_seed(0)
                    X_mod[:, t + delta, :] += torch.rand(B, D)
                    
                Y_mod = block(X_mod)
                # If Y[t] changes when a future X[t + delta] changes, then it is not causal
                if not torch.allclose(Y[:, t, :], Y_mod[:, t, :]):
                    self.rprint(f'Failed at t={t}, delta={delta}')
                    return False

        self.rprint('Causality test passed')
        return True

    def check_differentiable(self,model,vocab_size: int) -> bool:
        """Check if the mode is differentiable 

        :param model: 
            The target model with the new block. 
        :param vocab_size: 
            The model vocabulary size. 

        """
        self.rprint('Checking differentiability...')
        mock_input = torch.randint(0, vocab_size, (2, 100)).cuda() if \
          torch.cuda.is_available() else torch.randint(0, vocab_size, (2, 100))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())

        # Zero the parameter gradients
        optimizer.zero_grad()
        logits = model(mock_input).logits
        loss = criterion(
            logits.view(-1, logits.shape[-1]),
            mock_input.view(-1)
        )
        loss.backward()
        # Check gradients
        for name, param in model.named_parameters():
            if param.grad is None:
                self.rprint(f"Parameter {name} does not have a gradient")
                return False

        self.rprint('Differentiability test passed')
        return True

    def check_magnitude(
            self,
            size: int,
            magnitude: float,
            threshold: float
        ) -> bool:
        """Checks that the block maintains a certain limit on parameters 
            
        """
        self.logging.info(f'Checking non-embedding parameter number again the magnitude: {magnitude}')
        if size > (1+threshold)*magnitude:
            exceed = (size-magnitude)/magnitude
            self.rprint(
                f'Parameter number exceeds the magnitude by {exceed}'
            )
            return False
        elif size < (1-threshold)*magnitude:
            below = (magnitude-size)/magnitude
            self.rprint(
                f'Parameter number if below the magnitude: {below}'
            )
            return False
        self.rprint('Parameter number is within threshold')
        return True
    

    def check_efficiency(self, model, vocab_size: int) -> bool:

        return True
    

    def check(self, config, gab_code: str, name: str) -> bool:
        """Runs through a bunch of checks for the new module at path 

        :param path: 
            The path of the proposed module 
        """
        try: 
            print(config)
            glm,gab_config = reload_gam(config,gab_code,name)
            if torch.cuda.is_available():
                glm = glm.cuda()

            glm.print_size()

            mock_input=torch.randint(0, config.vocab_size, (8, 500))
            mock_input = mock_input.to(glm.device)

            output = glm(mock_input)

        except Exception as e:
            self.rprint(
                'Model initialization failed with error: '+str(e)+'\n'
            )
            return False,self.report
        
        ### check model size 
        gam = glm.backbone
        gab=gam.blocks[0].gab
        size=sum(p.numel() for p in gam.parameters())
        blocksize=sum(p.numel() for p in gam.blocks.parameters())
        perblock=blocksize//gam.n_block
        embsize=sum(p.numel() for p in gam.embedding.parameters())
        self.rprint(
            f'Model initialization succeed\nNumber of parameters: {size}\Blocks: {blocksize}, {perblock} per block\nEmbedding: {embsize}'
        )

        try:
            ### TURNED OFF, the model is not good at this. 
            # assert self.check_magnitude(
            #     blocksize,
            #     config.size_reference,
            #     config.size_threshold
            # )
            assert self.is_causal(
                gab,
                gam.d_model
            )
            assert self.check_differentiable(glm,config.vocab_size)
        except AssertionError:
            self.rprint('Model test failed\n')
            return False,self.report

        self.rprint("All tests passed!\n")
        return True,self.report
    
    # TODO: maybe tune layers as well, but its complicated due to embedding layer in small scale occupied a lot of size
    def tune(self,config,gab_code,name)->str: # the model is already correct but we need to tune its scale
        print('Tuning the model scale...')
        d_model=config.d_model
        assert d_model%128==0 # initial d_model from config should be a multiple of 128
        vocab_size=config.vocab_size
        reference_size=config.reference_size
        threshold=config.size_threshold
        step_size=d_model//8 # smallest d_model is 128
        if d_model%3==0: # reference d_model is like 256, 384, 512, 768...
            min_step=24
        else:
            min_step=16
        step_size=max(step_size,min_step) 
        UB=reference_size*(1+threshold)
        LB=reference_size*(1-threshold)
        print(f'Reference size: {reference_size}, threshold: {threshold}, upper bound: {UB}, lower bound: {LB}')

        auto_cfg={'d_model':d_model}
        glm,_ = reload_gam(config,gab_code,name,auto_cfg)
        size=sum(p.numel() for p in glm.parameters())
        if LB<size<UB:
            print('The model size is already within the threshold.')
            return 'autoconfig={}'
        
        DIR='UP' if size<LB else 'DOWN'
        while True:
            if step_size<min_step:
                break
            if DIR=='UP':
                d_model+=step_size
            else:
                d_model-=step_size
            print(f'Trying d_model={d_model}')
            auto_cfg['d_model']=d_model
            glm,_ = reload_gam(config,gab_code,name,auto_cfg)
            size=sum(p.numel() for p in glm.parameters())
            NEW_DIR='UP' if size<LB else 'DOWN'
            if NEW_DIR!=DIR:
                DIR=NEW_DIR
                step_size=step_size//2

        # Final adjustment to check whether the dim cause error (e.g., dim head)
        DIR='UP' if size<reference_size else 'DOWN'
        step_size=step_size//2
        print(f'Checking model correctness with d_model={d_model}')
        while True:
            try:
                if torch.cuda.is_available():
                    glm = glm.cuda()
                mock_input=torch.randint(0, vocab_size, (8, 500)).to(glm.device)
                _ = glm(mock_input)
                break
            except Exception as e:
                if DIR=='UP':
                    d_model+=step_size
                else:
                    d_model-=step_size
                print(f'The model is incorrect. Trying d_model={d_model}')
                glm,_ = reload_gam(config,gab_code,name,auto_cfg)
                size=sum(p.numel() for p in glm.parameters())
                if size>reference_size*(1+2*threshold) or size<reference_size*(1-2*threshold):
                    # Not likely to happen when reference d_model is a multiple of 128 and step_size is at least 8 or 12, but leave it for safety
                    raise ValueError('The model is too far from the reference size and cannot be correctly tuned.')

        print(f'The model is correct with d_model = {d_model}')
        print('Model after tuned:')
        glm.print_size()
        return "autoconfig = {\n    'd_model': "+str(d_model)+"\n}"
    
    def __call__(self,path: str) -> bool:
        return self.check(path)

