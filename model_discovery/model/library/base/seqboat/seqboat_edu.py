# Copyright (c) Liliang Ren.
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter

from fairseq import utils
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.modules.fairseq_dropout import FairseqDropout, FairseqFeatureDropout
from fairseq.modules.sequence_norm import SequenceNorm
from fairseq.modules.exponential_moving_average import MultiHeadEMA
from fairseq.modules.seqboat_utils import * 

from fairseq.modules.s4d_layer import S4D

@with_incremental_state
class SeqBoatUnit(nn.Module):

    def __init__(
        self,
        embed_dim,
        zdim,
        hdim,
        ndim,
        dropout=0.0,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        activation='silu',
        attention_activation='relu2',
        bidirectional=False,
        chunk_size=-1,
        truncation=None,
        norm_type='layernorm',
        prenorm=True,
        norm_affine=True,
        feature_dropout=False,
        rel_pos_bias='simple',
        max_positions=1024,
        export=False,
        density_threshold=0.9,
        max_steps = 180000,
        init_temp_scale=0.3,
        local_pos = False,
        always_act = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hdim = hdim
        self.zdim = zdim
        self.ndim = ndim
        self.activation = utils.get_activation_fn(activation=activation)
        self.attention_activation = attention_activation
        self.truncation = truncation

        dropout_module = FairseqFeatureDropout if feature_dropout else FairseqDropout
        self.dropout = dropout_module(dropout, module_name=self.__class__.__name__)
        self.hidden_dropout = dropout_module(hidden_dropout, module_name=self.__class__.__name__)

        local = True 
        self.local = local and chunk_size>0
        self.latent_conf = not always_act
        self.local_pos = self.local and local_pos and rel_pos_bias == 'simple'
        if self.local:
            density_threshold = 1
            self.window_size = chunk_size
            chunk_size = -1
        
        self.s4d= False
        
        self.use_rot_bias=False

        self.scaling = self.zdim ** -0.5 
        self.chunk_size = chunk_size
        self.prenorm = prenorm
        self.norm = SequenceNorm(norm_type, embed_dim, affine=norm_affine, export=export)
            
        if self.s4d:
            self.move = S4D(embed_dim, ndim=ndim, bidirectional=bidirectional, truncation=truncation)
        else:
            self.move = MultiHeadEMA(embed_dim, ndim=ndim, bidirectional=bidirectional, truncation=truncation, truc_norm=True)


        self.gamma = Parameter(torch.Tensor(2, zdim))
        self.beta = Parameter(torch.Tensor(2, zdim))

        # Attention dropout is standard dropout
        self.attention_dropout = FairseqDropout(attention_dropout, module_name=self.__class__.__name__)
        self.rel_pos_type = rel_pos_bias
        self.max_positions = max_positions
        
        if rel_pos_bias == 'simple':
            max_positions = max_positions if chunk_size < 0 else chunk_size
            if self.local_pos:
                max_positions = self.window_size*3 # Most 3x with non-causal
            self.rel_pos_bias = FastRelativePositionalBias(max_positions)

        elif rel_pos_bias == 'rotary':
            self.max_pos = max_positions
            self.freqs_cis = precompute_freqs_cis(
                self.zdim, max_positions
            )

            if self.use_rot_bias:
                self.rot_a = nn.Parameter(torch.Tensor(1, self.zdim))
                self.rot_b = nn.Parameter(torch.Tensor(1, self.zdim))
            
        elif rel_pos_bias == 't5':
            self.rel_pos_bias =  T5RelativePositionBias(scale = 1, num_buckets = 32 ,max_distance = 128)
        else:
            raise ValueError('unknown relative position bias: {}'.format(rel_pos_bias))


        self.use_sigmoid = False
        if self.use_sigmoid:
            num_options = 1
        else:
            num_options = 2
        self.density_threshold = density_threshold
        
        if self.latent_conf:
            self.q_router = LatentConfigurator(
                dim = embed_dim,
                num_vars = num_options,
                max_steps= max_steps,
                init_temp_scale= init_temp_scale
            ) 
 
        
       
        self.mx_proj = nn.Linear(embed_dim, zdim +hdim+hdim)
        self.opt_dim = num_options 
        if self.latent_conf:
            self.hx_proj = nn.Linear(embed_dim, self.opt_dim+embed_dim)
        else:
            self.hx_proj = nn.Linear(embed_dim, embed_dim)
        
        self.h_proj = nn.Linear(hdim, embed_dim)

        self.reset_parameters()

        self.onnx_trace = False
        self.tpu = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True

    def reset_parameters(self):
        with torch.no_grad():
            std = 0.02

            Normal(self.mx_proj.weight, mean=0.0, std=std)
            nn.init.constant_(self.mx_proj.bias, 0.0)

            Normal(self.h_proj.weight, mean=0.0, std=std)
            nn.init.constant_(self.h_proj.bias, 0.0)

            Normal(self.gamma, mean=0.0, std=std)

            nn.init.constant_(self.beta, 0.0)
            
            if self.rel_pos_type == 'rotary' and self.use_rot_bias:
                Normal(self.rot_a, mean=0.0, std=std)
                Normal(self.rot_b, mean=0.0, std=std)

            Normal(self.hx_proj.weight, mean=0.0, std=std)
            nn.init.constant_(self.hx_proj.bias, 0.0)



    def local_attention(self, q, k, v, attn_bias, mask, causal):
        
        # attn_bias: b,k,c,c
        shape, autopad, pad_value, window_size = q.shape, True, -1, self.window_size
        exact_windowsize = True
        if causal:
            look_backward = 1
            look_forward = 0
        else:
            look_backward = 1
            look_forward = 1
            
        # https://github.com/arogozhnikov/einops/blob/master/docs/4-pack-and-unpack.ipynb
        (q, packed_shape), (k, _), (v, _) = map(lambda t: pack([t], '* n d'), (q, k, v))

        # auto padding
        if autopad:
            orig_seq_len = q.shape[1]
            (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, self.window_size, dim = -2), (q, k, v))
            
        b, n, dim_head, device, dtype = *q.shape, q.device, q.dtype

        assert (n % window_size) == 0, f'sequence length {n} must be divisible by window size {window_size} for local attention'

        windows = n // window_size


        seq = torch.arange(n, device = device)
        b_t = rearrange(seq, '(w n) -> 1 w n', w = windows, n = window_size)

        # bucketing
        bq, bk, bv = map(lambda t: rearrange(t, 'b (w n) d -> b w n d', w = windows), (q, k, v))

        look_around_kwargs = dict(
            backward =  look_backward,
            forward =  look_forward,
            pad_value = pad_value
        )

        bk = look_around(bk, **look_around_kwargs)
        bv = look_around(bv, **look_around_kwargs)


        # calculate positions for masking
        bq_t = b_t
        bq_k = look_around(b_t, **look_around_kwargs)

        bq_t = rearrange(bq_t, '... i -> ... i 1')
        bq_k = rearrange(bq_k, '... j -> ... 1 j')

        pad_mask = bq_k == pad_value

        sim = einsum('b h i e, b h j e -> b h i j', bq, bk)
        
        if self.local_pos:
            sim = sim + self.rel_pos_bias(sim.shape[-1])[:sim.shape[-2],:]
        elif exists(attn_bias):
            if autopad:
                _,attn_bias = pad_to_multiple(attn_bias, self.window_size, dim = -2)
                _,attn_bias = pad_to_multiple(attn_bias, self.window_size, dim = -1)
            attn_bias = look_around_2d(attn_bias,window_size,look_backward,look_forward)
            ab,ak,aw,ac,av = attn_bias.shape
            sim = sim + attn_bias.view(-1,aw,ac,av)


        if self.attention_activation =="softmax":
            mask_value =  TOKEN_SELF_ATTN_VALUE
        else:
            mask_value = 0

        if causal:
            causal_mask = bq_t < bq_k

            if exact_windowsize:
                max_causal_window_size = (self.window_size * look_backward)
                causal_mask = causal_mask | (bq_t > (bq_k + max_causal_window_size))

            sim = sim.masked_fill(causal_mask, mask_value)
            del causal_mask

        # masking out for exact window size for non-causal
        # as well as masking out for padding value
        if not causal and exact_windowsize:
            max_backward_window_size = (self.window_size * look_backward)
            max_forward_window_size = (self.window_size * look_forward)
            window_mask = ((bq_k - max_forward_window_size) > bq_t) | (bq_t > (bq_k + max_backward_window_size)) | pad_mask
            sim = sim.masked_fill(window_mask, mask_value)
        else:
            sim = sim.masked_fill(pad_mask, mask_value)

        # take care of key padding mask passed in
        if exists(mask):
            batch = mask.shape[0]
            assert (b % batch) == 0

            if autopad:
                _, mask = pad_to_multiple(mask, window_size, dim = -1, value = False)

            mask = rearrange(mask, '... (w n) -> (...) w n', w = windows, n = window_size)
            mask = look_around(mask, **{**look_around_kwargs, 'pad_value': False})
            mask = rearrange(mask, '... j -> ... 1 j')
            sim = sim.masked_fill(~mask, mask_value)
            del mask
        
        # attention
        if self.attention_activation == 'softmax':
            attn = utils.softmax(sim, dim=-1).type_as(sim)
        elif self.attention_activation == 'relu2':
            attn = utils.relu2(sim).type_as(sim)
        elif self.attention_activation == 'laplace':
            attn = utils.laplace(sim).type_as(sim)      
        else:
            raise ValueError('Unknown attention activation function: {}'.format(self.attention_activation))

        attn = self.attention_dropout(attn)

        # aggregation
        out = einsum('b h i j, b h j e -> b h i e', attn, bv)
        out = rearrange(out, 'b w n d -> b (w n) d')

        if autopad:
            out = out[:, :orig_seq_len, :]

        out, *_ = unpack(out, packed_shape, '* n d')
        
        return out



    def element_attention(self, q, k, v, padding_mask, attn_mask, tick,compress, seq_len):
        # padding_mask: b,k,c
        
        slen = k.size(2)
        causal = attn_mask is not None

        if self.local_pos:
            bias = None
        elif self.rel_pos_type == 'rotary':
            if self.use_rot_bias:
                bias_a = apply_rotary_emb(self.rot_a[None,None,:,:].expand(k.size(0),k.size(1),slen,-1),tick)
                bias_b = apply_rotary_emb(self.rot_b[None,None,:,:].expand(k.size(0),k.size(1),slen,-1),tick)
                bias = torch.matmul(bias_a,bias_b.transpose(-1,-2))
            else:
                q = apply_rotary_emb(q, tick)
                k = apply_rotary_emb(k, tick)
                bias = None
            
        elif self.rel_pos_type == 't5':
            bias = self.rel_pos_bias(tick,tick,causal)
        else:
            # C x C
            bias = self.rel_pos_bias(seq_len)

            if slen != q.size(2):
                assert q.size(2) == 1
                # 1 x C
                bias = bias[-1:]

            
            if compress:
                bias = F.pad(bias,(1,0,1,0))
                bias = bias[tick]
                bias = torch.gather(bias.float(), -1, tick[:,:,None,:].expand(-1,-1,bias.shape[-2],-1)).type_as(bias)
            else:
                bias = bias[None,None,:,:]
        
        # bias : # b k c c
        
        if self.attention_activation == 'softmax':
            q = q * self.scaling
            if padding_mask is not None:
                padding_mask_all = padding_mask.all(dim=-1, keepdim=True)
                padding_mask = torch.logical_and(padding_mask, ~padding_mask_all)
        else:
            if padding_mask is not None:
                # B x K x C
                inverse_mask = ~padding_mask
                if not self.local:
                    # B x K x 1
                    lengths = inverse_mask.sum(dim=-1, keepdim=True)
                    # B x K x 1 x 1
                    lengths = lengths.clamp(min=1).unsqueeze(-1)

            else:
                lengths = slen
                inverse_mask = None
    
            if self.local:
                if causal:
                    lengths = self.window_size
                else:
                    lengths = self.window_size*2
            elif attn_mask is not None:
                # C x 1
                lengths = attn_mask.sum(dim=-1, keepdim=True)   
            # q: b k c h
            # tick: b k c 

            q =q / lengths
            
        if self.local:
            h = self.local_attention(q, k, v, bias, ~padding_mask, causal) #1 for keep
        else:
            # B x K x C x C1 (C1 can be non-equal to C)
            qk = torch.matmul(q, k.transpose(2, 3)) 
            if bias is not None:
                qk += bias

            if self.attention_activation == 'softmax':
                if attn_mask is not None:
                    qk = qk + attn_mask

                if padding_mask is not None:
                    qk = qk.masked_fill(padding_mask.unsqueeze(2).to(torch.bool), float('-inf'))
                attn_weights = utils.softmax(qk, dim=-1).type_as(qk)
            
            else:
                if self.attention_activation == 'relu2':
                    attn_weights = utils.relu2(qk).type_as(qk)
                elif self.attention_activation == 'laplace':
                    attn_weights = utils.laplace(qk).type_as(qk)      
                else:
                    raise ValueError('Unknown attention activation function: {}'.format(self.attention_activation))

                if inverse_mask is not None:
                    attn_weights = attn_weights * inverse_mask.unsqueeze(2)

                if attn_mask is not None:
                    attn_weights = attn_weights * attn_mask
                
            kernel = self.attention_dropout(attn_weights)
            # B xK x C x E  -> B x L x E -> L x B x E
            h = torch.matmul(kernel, v)

        return h


    def forward(
        self,
        x,
        padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_attn_fn: bool = False,
        act_bias: float = 0.0,
        verbose: bool=False,
        layer_idx: int=0,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_attn_fn (bool, optional): return the raw attention
                weights and values before the attention softmax.
        """

        seq_len, bsz, embed_dim = x.size()

        assert embed_dim == self.embed_dim

        saved_state = None

        residual = x
        if self.prenorm:
           x = self.norm(x)


        # L x B x D
        mx = self.move(x, padding_mask, incremental_state)


        mx = self.activation(mx)
        mx = self.dropout(mx)
        

        mx = mx.transpose(0,1)
       
        if saved_state is not None:
            # assert self.chunk_size < 0 or q.size(1) <= self.chunk_size
            # saved states are stored with shape (bsz, seq_len, dim)
            if "prev_mx" in saved_state:
                prev_mx= saved_state["prev_mx"]
                assert prev_mx is not None
                assert mx is not None
                mx = torch.cat([prev_mx, mx], dim=1)

            prev_padding_mask: Optional[Tensor] = None
            if "prev_padding_mask" in saved_state:
                prev_padding_mask = saved_state["prev_padding_mask"]
            padding_mask = SeqBoatUnit._append_prev_padding_mask(
                padding_mask=padding_mask,
                prev_padding_mask=prev_padding_mask,
                batch_size=bsz,
                seq_len=mx.size(1),
            )
            
            if self.chunk_size < 0:
                if self.local:
                    max_len = self.truncation
                    saved_state["prev_mx"] = mx[:,-max_len:]
                    saved_state["prev_key_padding_mask"] = padding_mask
                else:
                    saved_state["prev_mx"] = mx
                    saved_state["prev_key_padding_mask"] = padding_mask
            else:
                curr_len = mx.size(1) % self.chunk_size
                if curr_len == 0:
                    if "prev_mx" in saved_state:
                        del saved_state["prev_mx"]
                        del saved_state["prev_key_padding_mask"]
                else:
                    saved_state["prev_mx"] = mx
                    saved_state["prev_key_padding_mask"] = padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            self._set_input_buffer(incremental_state, saved_state)

        seq_len = mx.size(1)
        ctx_len = seq_len
        
        tick=None
        if self.rel_pos_type == 'rotary':
            
            if seq_len>self.max_pos:
                self.max_pos = seq_len  
                self.freqs_cis=precompute_freqs_cis(
                    self.zdim, seq_len
                ).to(x.device)
            else:
                self.freqs_cis = self.freqs_cis.to(x.device)
            tick = self.freqs_cis[None,:seq_len,:].expand(bsz,-1,-1)
            
            
        if self.chunk_size < 0:
            # B x L x S -> B x 1 x L x S
            mx = mx.unsqueeze(1)
            if self.rel_pos_type == 'rotary':
                tick = tick.unsqueeze(1)
            if padding_mask is not None:
                # B x L -> B x 1 x L
                padding_mask = padding_mask.unsqueeze(1)
        else:
            if seq_len < self.chunk_size:
                mx = mx.unsqueeze(1)
                if self.rel_pos_type == 'rotary':
                    tick = tick.unsqueeze(1)
            else:
                # B x L x S -> B x K x C x S
                nc = seq_len // self.chunk_size
                mx = mx.reshape(bsz, nc, self.chunk_size, -1)
                if self.rel_pos_type == 'rotary':
                    tick = tick.reshape(bsz, nc, self.chunk_size, -1)
            if ctx_len < self.chunk_size:
                if padding_mask is not None:
                    padding_mask = padding_mask.unsqueeze(1)
            else:
                # B x L x S -> B x K x C x S
                nc = ctx_len // self.chunk_size
                if padding_mask is not None:
                    # B x L -> B x K x C
                    padding_mask = padding_mask.view(bsz, nc, self.chunk_size)
        
        
        seq_len = mx.shape[-2]
        cks= mx.shape[1]
        
        hu = self.hx_proj(mx)
       
        if self.latent_conf: 
            u, hx= torch.split(hu, [self.opt_dim, self.embed_dim], dim=-1)
        else:
            hx = hu


        if not self.rel_pos_type == 'rotary' and not self.local_pos:
            tick = torch.arange(1,1+seq_len,device =mx.device)[None,None,:,None].expand(bsz,cks,-1,-1)
        if verbose:
            span_tick = torch.arange(1,1+seq_len,device =mx.device)[None,None,:,None].expand(bsz,cks,-1,-1)
            
        if self.latent_conf:
            result = self.q_router(u, act_bias=act_bias)
            
            px = result["probs"]
            loss  = [0]
            if verbose:
                print("temp ",result["temp"].item())
            if self.use_sigmoid:
                q_p = px
                mask_q = q_p>0.5
            else:
                q_p, q_ind = px.max(dim=-1, keepdim = True) # B x K x C x 1
                mask_q = (q_ind==1) # B x K x C x 1
                    
            if padding_mask is not None:
                pad_mask = ~padding_mask.unsqueeze(-1).bool()
                mask_q *= pad_mask

  
            q_probs = (q_p * mask_q)

            max_sl = mask_q.sum(-2).max()
            density = max_sl/mask_q.shape[-2]

            if verbose:
                torch.save(q_probs, "act_"+str(layer_idx)+".pt")
                print("ACT_Time\t", mask_q.float().mean(-2).mean().item(),"\t",mask_q.float().mean(-2).std().item())
            compress = density < self.density_threshold
            index_q = None
            if compress:
                
                index_q = (mask_q*torch.cumsum(mask_q.long(),dim=-2))

                mx = compress_seq(mx,index_q.expand(-1,-1,-1,mx.shape[-1]), max_sl ,dim = -2) # mq, B, c
                
                if not self.local_pos:
                    tick = compress_seq(tick,index_q.expand(-1,-1,-1,tick.shape[-1]), max_sl ,dim = -2) # mq, B, c
                if verbose:
                    span_tick = compress_seq(span_tick,index_q.expand(-1,-1,-1,span_tick.shape[-1]), max_sl ,dim = -2) # mq, B, c
         
                pad_mask = compress_seq(mask_q,index_q.expand(-1,-1, -1,1), max_sl ,dim = -2) # mq, B, 1
                padding_mask = ~pad_mask.squeeze(-1)
                
                if attn_mask is not None:
                    assert len(attn_mask.shape) == 2
                    if max_sl==0:
                        attn_mask = attn_mask[:1, :1]
                    else:
                        attn_mask = attn_mask[:max_sl, :max_sl]
            else:
                
                padding_mask = ~mask_q.squeeze(-1)
        else:
            compress= False
            loss = [0]
        
        if verbose:
            wsize = self.window_size if self.local else seq_len
            avg_span = calc_avg_attn_span(span_tick.squeeze(-1),win = wsize ,causal = attn_mask is not None)
            print("AVG_SPAN: ", (avg_span.sum()/((avg_span!=0).sum()+1e-6)).item())
        if not self.rel_pos_type == 'rotary' and not self.local_pos:
            tick = tick.squeeze(-1)
        # L x B x E
        base = self.mx_proj(mx)
            
        z, r, v = torch.split(self.activation(base), [self.zdim, self.hdim, self.hdim], dim=-1)
        # L x B x S -> L x B x 1 x S -> L x B x 2 x S
        z = z.unsqueeze(-2) * self.gamma + self.beta

        # L x B x 2 x S -> L x B x S
        q, k = torch.unbind(z, dim=-2)
    

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if padding_mask is not None and padding_mask.dim() == 0:
            padding_mask = None

        v = self.hidden_dropout(v, batch_first=True)
        
        h = self.element_attention(q, k, v, padding_mask, attn_mask, tick, compress, seq_len)

        h = self.h_proj(h*r)
        if attn_mask is None:
            h = self.dropout(h)

        if self.latent_conf:
            if compress:
                h = extract(h, index_q)
            h = h*q_probs
        
        h = h.view(bsz, -1, self.embed_dim).transpose(0, 1)
        hx = hx.view(bsz, -1, self.embed_dim).transpose(0, 1)

        if saved_state is not None:
            slen = residual.shape[0]
            hx = hx[-slen:,:,:]
            h = h[-slen:,:,:]
            
        if attn_mask is None:
            out = hx + h + residual
        else:
            out = self.dropout(hx + h) + residual
            
        out = self.activation(out)

        if not self.prenorm:
            out = self.norm(out)     

        return out, None, loss

    @staticmethod
    def _append_prev_padding_mask(
        padding_mask: Optional[Tensor],
        prev_padding_mask: Optional[Tensor],
        batch_size: int,
        seq_len: int,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_padding_mask is not None and padding_mask is not None:
            new_padding_mask = torch.cat([prev_padding_mask, padding_mask], dim=1)
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_padding_mask is not None:
            filler = torch.zeros((batch_size, seq_len - prev_padding_mask.size(1)), device=prev_padding_mask.device)
            new_padding_mask = torch.cat([prev_padding_mask, filler.bool()], dim=1)
        elif padding_mask is not None:
            filler = torch.zeros((batch_size, seq_len - padding_mask.size(1)), device=padding_mask.device)
            new_padding_mask = torch.cat([filler.bool(), padding_mask], dim=1)
        else:
            new_padding_mask = prev_padding_mask
        return new_padding_mask


if __name__=="__main__":
    torch.set_printoptions(precision=20)
    torch.manual_seed(1)
    layer1 = SeqBoatUnit(10,20,30,40, window_size=0, max_positions=4096,)
    torch.manual_seed(1)
    layer2 = SeqBoatUnit(10,20,30,40, window_size=100, max_positions=4096,)
    x = torch.rand(4096,22,10)
    print(layer1(x))

    x = torch.rand(4096,22,10)
    x.requires_grad_()
    # x1=torch.clone(x)
    # x1.requires_grad_()
    torch.manual_seed(1)
    out1  = layer1(x)[0]
    torch.manual_seed(1)
    out2 = layer2(x)[0]
    def calc_max_diff(out1,out2):
        nz= (out1!=out2).nonzero()
        print((out1[nz]-out2[nz]).abs().max())
    calc_max_diff(out1,out2)
    #assert (out1==out2).all()
    loss = out1.sum()
    loss.backward()
    xmx = torch.clone(layer1.mx_proj.weight.grad)
    xg1 = torch.clone(x.grad)
    loss = out2.sum()
    loss.backward()
    #assert (xg1==x.grad/2).all()
    calc_max_diff(xg1,x.grad/2)

    print(xmx[-1,1])
    print(layer2.mx_proj.weight.grad[-1,1])
    print(xmx[-1,1] == layer2.mx_proj.weight.grad[-1,1])
    comp= xmx==layer2.mx_proj.weight.grad
    print(comp)
    assert comp.all()
    