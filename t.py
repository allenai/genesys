# import torch
# import torch.nn as nn

# from mamba_ssm.modules.mha import MHA


# class GAB(nn.Module):
#     """Generalized Autoregressive Block
#         Input:        X: (batch, seqlen, embed_dim)
#         Output:       Y: (batch, seqlen, embed_dim)
#         Constraints:  Causal, differentiable, parameter number, complexity, parallelizable
#     """
#     def __init__(self, embed_dim: int, device=None, dtype=None, n_heads=8, dropout=0.1): 
#         factory_kwargs = {"device": device, "dtype": dtype} 
#         super().__init__()
        
#         self.attention = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, **factory_kwargs)
#         self.bidir_attn = MHA(embed_dim, n_heads, **factory_kwargs)
#         self.causal_attn = MHA(embed_dim, n_heads, causal=True, **factory_kwargs)

#         self.lstm=nn.LSTM(embed_dim, embed_dim, batch_first=True)
#         self.bilstm=nn.LSTM(embed_dim, embed_dim, batch_first=True, bidirectional=True)

#         self.causal_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=2, groups=4, **factory_kwargs)
#         self.bidir_conv = nn.Conv1d(embed_dim, embed_dim, 3, padding=1)

#     def forward(self, X, **kwargs): 
#         mask = nn.Transformer.generate_square_subsequent_mask(len(X)).to(X.device)
#         causal_output, _ = self.attention(X, X, X, attn_mask=mask)
#         bidir_output, _ = self.attention(X, X, X)
        
#         # causal_output = self.causal_attn(X)
#         # bidir_output = self.bidir_attn(X)

#         # causal_output,_ = self.lstm(X)
#         # bidir_output,_ = self.bilstm(X)

#         # causal_output = self.causal_conv(X.permute(0,2,1)).permute(0,2,1)[:,:-2]
#         # bidir_output = self.bidir_conv(X.permute(0,2,1)).permute(0,2,1)
#         return causal_output, bidir_output

# mha=GAB(32)
# mha.eval()

# input = torch.arange(10 * 32 * 32).float().reshape(10, 32, 32)
# causal_out,bidir_out= mha(input)

# input_=input.clone()
# input_[:, 16:,:] *= -1
# causal_out_pert,bidir_out_pert = mha(input_)

# print('Causal MHA:',torch.allclose(causal_out[:, :16,:], causal_out_pert[:, :16,:]))  
# print('Bi-Dir MHA:',torch.allclose(bidir_out[:,:16], bidir_out_pert[:,:16]))


import numpy as np
from scipy.optimize import curve_fit

# Define the function forms for fitting
def runtime_function(n, a, b,c):
    return a * np.power(n, b)+c

def memory_function(n, c, d,e):
    return c * np.power(n, d)+e

def analyze_complexity(sequence_lengths, runtimes, memory_usages):
    # Convert lists to numpy arrays for fitting
    sequence_lengths = np.array(sequence_lengths)
    runtimes = np.array(runtimes)
    memory_usages = np.array(memory_usages)

    # Fit the runtime data to the runtime_function
    popt_runtime, _ = curve_fit(runtime_function, sequence_lengths, runtimes)
    a_runtime, b_runtime,_ = popt_runtime

    # Fit the memory usage data to the memory_function
    popt_memory, _ = curve_fit(memory_function, sequence_lengths, memory_usages)
    c_memory, d_memory,_ = popt_memory

    print(f"Runtime complexity: O(n^{b_runtime:.2f})")
    print(f"Memory usage complexity: O(n^{d_memory:.2f})")

    return popt_runtime, popt_memory

# Example usage with collected data
sequence_lengths = [500 * K for K in range(1, 11)]
runtimes = [0.05, 0.12, 0.23, 0.41, 0.64, 0.94, 1.30, 1.74, 2.23, 2.80]  # Replace with actual measured runtimes
memory_usages = [200, 400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600]  # Replace with actual measured memory usage

analyze_complexity(sequence_lengths, runtimes, memory_usages)
