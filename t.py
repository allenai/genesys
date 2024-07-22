import torch
import torch.nn as nn

from mamba_ssm.modules.mha import MHA


class GAB(nn.Module):
    """Generalized Autoregressive Block
        Input:        X: (batch, seqlen, embed_dim)
        Output:       Y: (batch, seqlen, embed_dim)
        Constraints:  Causal, differentiable, parameter number, complexity, parallelizable
    """
    def __init__(self, embed_dim: int, device=None, dtype=None, n_heads=8, dropout=0.1): 
        factory_kwargs = {"device": device, "dtype": dtype} 
        super().__init__()
        
        # self.attention = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, **factory_kwargs)
        self.bidir_attn = MHA(embed_dim, n_heads, **factory_kwargs)
        self.causal_attn = MHA(embed_dim, n_heads, causal=True, **factory_kwargs)

    def forward(self, X, **kwargs): 
        # mask = nn.Transformer.generate_square_subsequent_mask(len(X)).to(X.device)
        # causal_attn_output, _ = self.attention(X, X, X, attn_mask=mask)
        # bidir_attn_output, _ = self.attention(X, X, X)
        
        causal_attn_output = self.causal_attn(X)
        bidir_attn_output = self.bidir_attn(X)
        return causal_attn_output, bidir_attn_output
    
mha=GAB(32)
mha.eval()

input = torch.arange(10 * 32 * 32).float().reshape(10, 32, 32)
causal_out,bidir_out= mha(input)

input_=input.clone()
input_[:, 16:,:] *= -1
causal_out_pert,bidir_out_pert = mha(input_)

print('Causal MHA:',torch.equal(causal_out[:, :16,:], causal_out_pert[:, :16,:]))  
print('Bi-Dir MHA:',torch.equal(bidir_out[:,:16], bidir_out_pert[:,:16]))

