import flax.linen as nn
import jax.numpy as jnp
import jax
from jax.numpy import einsum
from einops import rearrange
from typing import Callable
import sympy



default_init = nn.initializers.normal(1.0)


def get_bigram_ids(ids, vocab_size, segment_pos = None):
    """Generate bi-gram ids from uni-gram ids.
    
    Args:
      ids: uni-gram cluster ids, dtype int32, shape [B, L, H].
      vocab_size: Vocabulary size of `ids`, must be > 0.
      segment_pos: If not None (meaning `ids` is packed, i.e. each example
      containing multiple segments), a tensor of shape [B, L], containing
      the position of each id in `ids` in a segment.
    
    Returns:
      ngram_ids: dtype int64, shape [B, L, H].
    
    """
    assert vocab_size > 0, "vocab_size has to be greater than 0"
    batch_size = ids.shape[0]
    num_heads = ids.shape[-1]
    # Cast to int64 to avoid overflow (only if it is enabled)
    ids = jnp.array(ids, dtype=jnp.int64)  # [batch, seq, heads]
    pad = jnp.zeros([batch_size, 1, num_heads], dtype=ids.dtype)  # [batch, 1, heads]
    
    #   bigram_id = original_id + shifted_id * vocab_size.
    ids_0 = jnp.concatenate([ids, pad], 1)  # [batch, seq+1, heads]
    ids_1 = jnp.concatenate([pad, ids], 1)  # [batch, seq+1, heads]
    if segment_pos is not None:
        mask = jnp.array(jnp.equal(segment_pos, 0), dtype=ids.dtype) #[batch, seq]
        mask = 1 - mask
        mask = mask[:,:,jnp.newaxis] # [batch, seq, 1]
        mask = jnp.concatenate([mask, jnp.zeros([batch_size, 1, 1], dtype=ids.dtype)],1)
        ids_1 *= mask 


    ngram_ids = ids_0 + ids_1 * vocab_size  # Bigram ids.
    ngram_ids = ngram_ids[:, 0:-1]
    return ngram_ids




class ProductQuantization(nn.Module):
    """Implements the Product Quantization layer

  use the following capital letters to denote shape parameters:
    B = batch size
    L = length of the input sequence 
    H = number of attention heads
    D = dimensions of each attention head
    K = number of clusters
  """

    num_clusters: int
    num_heads: int
    dim_per_head: int
    decay : float = 0.999
    epsilon: float = 1e-6
    mean_init: Callable = default_init

    @nn.compact
    def __call__(self, x, train: bool = True):
        
        initializing = self.is_mutable_collection('params')
        # means is the k-mean of the mini-batch
        if initializing:
          key = self.make_rng('batch_stats')
          means = self.variable(
              'batch_stats', 'means', self.mean_init, 
              key,(self.num_heads, self.num_clusters, self.dim_per_head))
        else:
          means = self.variable('batch_stats','means', None)
        
        # compute distances of the input to all centroids 
        # [B,L,H,K]
        dists = -2 * jnp.einsum('B L H D, H K D -> B L H K', x, means.value)
        # [B,L,H,1]
        x_norm_sq = jnp.sum(jnp.square(x), axis=-1, keepdims=True)
        # [H,K]
        means_norm_sq = jnp.sum(jnp.square(means.value), axis=-1, keepdims=False)
        # [1, 1, H, K]
        means_norm_sq = means_norm_sq[jnp.newaxis, jnp.newaxis, :, :]
        # [B,L,H,K]
        dists += x_norm_sq + means_norm_sq

        #  find the nearest centroids id for each input vector
        # [B,L,H]
        cluster_ids = jnp.argmin(dists, axis=-1)
      
        
        if train and not initializing:
          # [B,L,H,K]
          nearest_one_hot = jax.nn.one_hot(cluster_ids, self.num_clusters)
          per_cluster_count = jnp.sum(nearest_one_hot, axis=[0, 1])
          sum_x = jnp.einsum('B L H K, B L H D -> H K D', nearest_one_hot,x)
          # means_x is the average over all the input vectors closest to this centroid.
          means_x = sum_x / (self.epsilon + jnp.expand_dims(per_cluster_count, axis=-1))
          # exponential moving average
          new_means = (1 - self.decay) * means_x + self.decay * means.value
          # update the means 
          means.value = new_means

        return(cluster_ids)

class Ngrammer(nn.Module):
    """Augments the input embeddings with VQ n-gram layer embeddings.

    Attributes:
      unigram_vocab_size: Size of the unigram vocabulary, i.e. number of unique centeroids 
      dim_per_head: The dimension per each head of the input
      num_heads: Number of attention heads
      ngram_emb_dim: Size of the ngram dimension per head
      ngram_vocab_size : Size of the ngram vocabulary
      concat_ngrams:If True, then concat ngrams and unigram, otherwise add
      embed_init: initializer function for ngram embedding layer 
    """
    unigram_vocab_size: int
    dim_per_head: int
    num_heads:int  = 1
    ngram_emb_dim: int = 8
    ngram_vocab_size :int = 768 * 256
    concat_ngrams: bool = True
    embed_init: Callable = default_init


    @nn.compact
    def __call__(self, ids, x, mask = None, segment_pos = None):
        """
        Args:
          ids: Input unigram id tensor of shape  [B, L, H].
          x: Input unigram embedding tensor of shape [B, L, H, D] to which to add the ngram embedding.
      
        Returns:
          out: output with ngram embedding added of shape [B, L, H * D]

        """
        if self.concat_ngrams:
          # The ngram_emb_dim must be smaller than dim_per_head.
           assert self.ngram_emb_dim <= self.dim_per_head
        else:
         # If not concatenating ngram embeddings, check the dims are compatible.
           assert self.ngram_emb_dim == self.dim_per_head

        ngram_cluster_ids = get_bigram_ids(ids, self.unigram_vocab_size, segment_pos)
        
        #  multi-way hash ids 
        primes = list(sympy.primerange(self.ngram_vocab_size + 1,2 * self.ngram_vocab_size))[0:self.num_heads]
        primes = jnp.array(primes)[jnp.newaxis,jnp.newaxis,:]
        
        head_range = jnp.arange(self.num_heads)[jnp.newaxis,jnp.newaxis,:]

        def _multi_way_hash_ids(x, a, b, prime, buckets):
            return ((x * a + b) % prime) % buckets

        ngram_ids = _multi_way_hash_ids(ngram_cluster_ids, head_range+1, head_range+1, primes, self.ngram_vocab_size)
        
        # shift vocab range for each head appropriately by the head number
        ngram_ids = ngram_ids + (self.ngram_vocab_size * head_range)

        ngram_embed = nn.Embed(self.ngram_vocab_size * self.num_heads, self.ngram_emb_dim, embedding_init = self.embed_init)
        
        y = ngram_embed(ngram_ids)
        
        normed_x = nn.LayerNorm(epsilon = 1e-5, reduction_axes=-1, feature_axes=(-2,-1))(x)
        normed_y = nn.LayerNorm(epsilon = 1e-5, reduction_axes=-1, feature_axes=(-2,-1))(y)
        
        input_sliced_dim = normed_x.shape[-1] - normed_y.shape[-1]
        out = jnp.concatenate((
                normed_x[..., :input_sliced_dim],
                normed_y), axis = -1)
        
        out = rearrange(out, 'b n ... -> b n (...)')

        # mask if needed

        if mask is not None:
            out = out * mask[:,:,jnp.newaxis]
        

        return(out)

    

class PQNgrammer(nn.Module):
    """Implements a PQ based ngrammer layer which looks up latent ngram id.

    We use the following capital letters to denote shape parameters:
    B = batch size
    L = length of the input sequence (referred to as S or T elsewhere)
    H = number of attention heads
    D = dimensions of each attention head
    K = number of clusters
    """
    num_clusters: int 
    num_heads: int 
    dim_per_head: int 
    ngram_vocab_size :int = 768 * 256
    ngram_emb_dim: int = 8
    decay : float = 0.999
    epsilon : float = 1e-6
    concat_ngrams: bool = True
    mean_init: Callable = default_init
    embed_init: Callable = default_init
    
    @nn.compact
    def __call__(self, x, train: bool = True,  mask = None, segment_pos = None):

        assert x.shape[-1] == (self.num_heads * self.dim_per_head), f'input embedding feature dimension must be {self.num_heads * self.dim_per_head}'
        
        x = rearrange(x, 'B L (H D) -> B L H D', H = self.num_heads)
        
        cluster_ids = ProductQuantization(
            num_clusters = self.num_clusters,
            num_heads = self.num_heads,
            dim_per_head = self.dim_per_head,
            decay = self.decay,
            epsilon = self.epsilon,
            mean_init = self.mean_init)(x, train)
 
        output_embs = Ngrammer(
            unigram_vocab_size = self.num_clusters,
            dim_per_head = self.dim_per_head,
            num_heads = self.num_heads,
            ngram_emb_dim = self.ngram_emb_dim,
            ngram_vocab_size  = self.ngram_vocab_size,
            concat_ngrams = self.concat_ngrams,
            embed_init = self.embed_init)(cluster_ids,x, mask, segment_pos)


        return(output_embs)
        
        