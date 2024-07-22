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

    ### HAS SOME WEIRD BUGS ### It may also due to torch
    def is_causal(self, block, D: int, seq_len: int = 100) -> bool:
        """Checks if a design is causal

        :param block: 
            The target block design. 
        :param D: 
            The block dimensions. 
        :param seq_len: 
            The block target sequence length.
        """
        B: int = 10
        X = torch.arange(seq_len * B * D).float().reshape(B, seq_len, D)
        if torch.cuda.is_available():
            X = X.cuda()
            
        block.eval()  # Set block to evaluation mode, so that dropout layers are not active
        with torch.no_grad():
            Y = block(X)

        print('Checking causality... It checks the causality by changing all future steps X[t+delta] of X[t] and see if Y[t] or any previous outputs change.')
        bar = tqdm(range(seq_len), desc='Causality test', colour='green')
        for t in bar:
            X_mod = X.clone()
            X_mod[:, t + 1:, :]*=-1 # Perturb the future steps of X[t]

            with torch.no_grad():
                Y_mod = block(X_mod)
                        
            # If any previous outputs change when future X[t + delta] changes, then it is not causal
            if not torch.equal(Y[:, :t+1, :], Y_mod[:, :t+1, :]):#, atol=1e-5):
                print(f'Failed at t={t}')
                return False

        print('Causality test passed')
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
        model.train()
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
    

    def check_efficiency(self, model, vocab_size: int) -> bool:

        return True
    

    def check(self, config, gab_code: str, name: str) -> bool:
        """Runs through a bunch of checks for the new module at path 

        :param path: 
            The path of the proposed module 
        """
        # try: 
        print(config)
        glm,gab_config = reload_gam(config,gab_code,name)
        if torch.cuda.is_available():
            glm = glm.cuda()

        glm.print_size()

        mock_input=torch.randint(0, config.vocab_size, (8, 500))
        mock_input = mock_input.to(glm.device)

        output = glm(mock_input)

        # except Exception as e:
        #     self.rprint(
        #         'Model initialization failed with error: '+str(e)+'\n'
        #     )
        #     return False,self.report
        
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
    
    def tune(self,config,gab_code,name,tune_dim=True)->str: # the model is already correct but we need to tune its scale
        print('Tuning the model scale...')
        d_model=config.d_model
        n_block=config.n_block
        # assert d_model%128==0 # initial d_model from config should be a multiple of 128
        vocab_size=config.vocab_size
        reference_size=config.reference_size
        threshold=config.size_threshold
        UB=reference_size*(1+threshold)
        LB=reference_size*(1-threshold)
        print(f'Reference size: {reference_size}, threshold: {threshold}, upper bound: {UB}, lower bound: {LB}')

        glm,_ = reload_gam(config,gab_code,name)
        size=sum(p.numel() for p in glm.parameters())
        if LB<size<UB:
            print('The model size is already within the threshold.')
            return 'autoconfig={}'
        
        # Tune n_blocks first, then d_model, idea is to maximally keep the size of embedding layer first
        DIR=1 if size<LB else -1
        while True:
            n_block+=1
            print(f'Trying n_block={n_block}')
            auto_cfg={'n_block':n_block}
            glm,_ = reload_gam(config,gab_code,name,auto_cfg)
            size=sum(p.numel() for p in glm.parameters())
            if LB<size<UB:
                print('Model after tuned:')
                glm.print_size()
                return "autoconfig = {\n    'n_block': "+str(n_block)+"\n}"
            if (DIR==1 and size>UB) or (DIR==-1 and size<LB):
                print('The model size requirement cannot be met by tuning n_block.')
                break
    
        if not tune_dim:
            raise ValueError('The model size requirement cannot be met by tuning n_block.')
                
        print('Tuning d_model...')
        step_size=d_model//8 # smallest d_model is 128
        if d_model%3==0: # like 384, 768...
            min_step=24
        else:
            min_step=16
        step_size=max(step_size,min_step) 

        DIR=1 if size<LB else -1
        while True: # tune d_model as little as possible
            if (step_size<min_step) or (LB<size<UB):
                break
            d_model+=step_size*DIR
            print(f'Trying d_model={d_model}, n_block={n_block}')
            auto_cfg={'d_model':d_model,'n_block':n_block}
            glm,_ = reload_gam(config,gab_code,name,auto_cfg)
            size=sum(p.numel() for p in glm.parameters())
            NEW_DIR=1 if size<reference_size else -1
            if NEW_DIR!=DIR:
                DIR=NEW_DIR
                step_size=step_size//2
        # if not LB<size<UB: # usually unless the agent create a over huge block
        #     raise ValueError('The model size requirement cannot be met by tuning d_model.')
        
        # Final adjustment of dim to check whether the dim cause error (e.g., dim head)
        DIR=1 if size<reference_size else -1
        print(f'Checking model correctness with d_model={d_model}')
        while True:
            try:
                if torch.cuda.is_available():
                    glm = glm.cuda()
                mock_input=torch.randint(0, vocab_size, (8, 500)).to(glm.device)
                _ = glm(mock_input)
                break
            except Exception as e:
                d_model+=step_size*DIR
                print(f'The model is incorrect. Trying d_model={d_model}')
                glm,_ = reload_gam(config,gab_code,name,auto_cfg)
                size=sum(p.numel() for p in glm.parameters())
                if size>reference_size*(1+2*threshold) or size<reference_size*(1-2*threshold):
                    # Not likely to happen when reference d_model is a multiple of 128 and step_size is at least 8 or 12, but leave it for safety
                    raise ValueError('The model is too far from the reference size and cannot be correctly tuned.')

        print(f'The model is correct with d_model = {d_model}')
        print('Model after tuned:')
        glm.print_size()
        return "autoconfig = {\n    'd_model': "+str(d_model)+"\n    'n_block': "+str(n_block)+"\n}"
    
    def __call__(self,path: str) -> bool:
        return self.check(path)

