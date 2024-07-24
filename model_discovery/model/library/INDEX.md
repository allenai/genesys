# Baselines and Core Reference Designs (155+)

Major and latest branches of (autoregressive) model architecture variants after Transformers/GPTs. Collected from surveys, popular repos, and recent papers. Excluded MoE, Hierarchical and Heterogeneous Architectures, and non-causal models. Only consider identical single causal blocks. Can use S2 with reference type to build the phylogenetic tree later. Do not considering param-sharing like Albert.

## Collected Varaints (50)

### Example Variants
1. GPT, Language Models are Few-Shot Learners
2. ✅ TTT, Learning to (Learn at Test Time): RNNs with Expressive Hidden States, https://github.com/test-time-training/ttt-lm-pytorch
3. xLSTM, xLSTM: Extended Long Short-Term Memory, https://github.com/NX-AI/xlstm
4. Griffin, Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models, https://github.com/google-deepmind/recurrentgemma
5. Hyena, Hyena Hierarchy: Towards Larger Convolutional Language Models (ICML'23 Oral), https://github.com/HazyResearch/safari
6. M2, Monarch Mixer: A Simple Sub-Quadratic GEMM-Based Architecture, https://github.com/HazyResearch/m2
7. SpikeGPT, SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks (TMLR,24), https://github.com/ridgerchu/SpikeGPT 

### SSMs (https://github.com/state-spaces):
8. ✅ Mamba2, Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality
9. S4, Efficiently Modeling Long Sequences with Structured State Spaces
10. HiPPO, HiPPO: Recurrent Memory with Optimal Polynomial Projections
11. LSSL, Combining Recurrent, Convolutional, and Continuous-time Models with Linear State-Space Layers
12. HTTYH, How to Train Your HiPPO: State Space Models with Generalized Orthogonal Basis Projections
13. S4D, On the Parameterization and Initialization of Diagonal State Space Models
14. 👉 Mamba, Mamba: Linear-Time Sequence Modeling with Selective State Spaces

### Flash Linear Attention (https://github.com/sustcsonglin/flash-linear-attention):
15. 2024-06	Samba, Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling	
16. 2023-07	RetNet, Retentive network: a successor to transformer for large language models	
17. 2023-12	GLA, Gated Linear Attention Transformers with Hardware-Efficient Training	
18. 2023-12	Based, An Educational and Effective Sequence Mixer	
19. 2024-01	Rebased, Linear Transformers with Learnable Kernel Functions are Better In-Context Models	
20. 2021-02	Delta Net, Linear Transformers Are Secretly Fast Weight Programmers	
21. 2023-09	Hedgehog, The Hedgehog & the Porcupine: Expressive Linear Attentions with Softmax Mimicry	
22. 2023-10	PolySketchFormer, Fast Transformers via Sketching Polynomial Kernels	
23. 2023-07	TransnormerLLM, A Faster and Better Large Language Model with Improved TransNormer 
24. 2023-05	RWKV-v4, Reinventing RNNs for the Transformer Era	
25. 2023-10	GateLoop, Fully Data-Controlled Linear Recurrence for Sequence Modeling		
26. 2021-10	ABC, Attention with Bounded-memory Control		
27. 2023-09	VQ-transformer, Linear-Time Transformers via Vector Quantization		
28. 2023-09	HGRN, Hierarchically Gated Recurrent Neural Network for Sequence Modeling	
29. 2024-04	HGRN2, HGRN2: Gated Linear RNNs with State Expansion	
30. 2024-04	RWKV6, Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence	

### From latest papers: 
31. Mega: Moving Average Equipped Gated Attention, https://github.com/lucidrains/Mega-pytorch?tab=readme-ov-file, ICLR 2023
32. DCMHA (ICML’24 Oral), Improving Transformers with Dynamically Composable Multi-Head Attention
33. AugLA (ICML’24), When Linear Attention Meets Autoregressive Decoding: Towards More Effective and Efficient Linearized Large Language Models, https://github.com/GATECH-EIC/Linearized-LLM/blob/main/flash_pytorch.py 
34. Ring (ICLR’24), Ring Attention with Blockwise Transformers for Near-Infinite Context
35. BPT (NeurIPS’23), Blockwise Parallel Transformer for Large Context Models
36. Efficient Attention via Control Variates, EVA (ICLR’23 Oral)
37. DiJiang (ICML’24 Oral), DiJiang: Efficient Large Language Models through Compact Kernelization
38. TNN (ICLR’23 Spotlight), Toeplitz Neural Network for Sequence Modeling 
39. CoPE (arXiv 2405), Contextual Position Encoding: Learning to Count What's Important
40. Infiniti (arXiv 2404), Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention
41. PEER (arXiv 2407), Mixture of A Million Experts
42. CoLT5: Faster Long-Range Transformers with Conditional Computation, https://github.com/lucidrains/CoLT5-attention?tab=readme-ov-file, EMNLP 2023
43. When Linear Attention Meets Autoregressive Decoding: Towards More Effective and Efficient Linearized Large Language Models, ICML 2024
44. Functional Interpolation for Relative Positions Improves Long Context Transformers, ICLR 2024
45. Self-attention Networks Localize When QK-eigenspectrum Concentrates (ICML 2024)

### More (e.g., from ET Survey https://arxiv.org/pdf/2009.06732)
46. Synthesizer: Rethinking Self-Attention in Transformer Models, https://github.com/10-zin/Synthesizer?tab=readme-ov-file, ICML 2021
47. cosFormer: Rethinking Softmax in Attention, https://github.com/OpenNLPLab/cosFormer, ICLR 2022
48. LARA: Linear complexity randomized self-attention mechanism (ICML 2022)
49. Luna: Linear unified nested attention, https://github.com/sooftware/luna-transformer, NeurIPS 2021
50. Coneheads: Hierarchy Aware Attention, NeurIPS 2023


## From *ELLM Survey* (https://github.com/AIoT-MLSys-Lab/Efficient-LLMs-Survey) (75)

### Efficient Architecture

#### Efficient Attention

##### Sharing-based Attention
1. LoMA: Lossless Compressed Memory Attention, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2401.09486)]
2. MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2402.14905)]
3. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints, <ins>EMNLP, 2023</ins> [[Paper](https://arxiv.org/abs/2305.13245)]
4. Fast Transformer Decoding: One Write-Head is All You Need, <ins>arXiv, 2019</ins> [[Paper](https://arxiv.org/abs/1911.02150)]

##### Feature Information Reduction
5. Nyströmformer: A nyström-based algorithm for approximating self-attention, <ins>AAAI, 2021</ins> [[Paper](https://arxiv.org/abs/2102.03902)] [[Code](https://github.com/mlpen/Nystromformer)] [[Code](https://github.com/lucidrains/nystrom-attention)]
6. Funnel-Transformer: Filtering out Sequential Redundancy for Efficient Language Processing, <ins>NeurIPS, 2020</ins> [[Paper](https://arxiv.org/abs/2006.03236)] [[Code](https://github.com/laiguokun/Funnel-Transformer)]
7. Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks, <ins>ICML, 2019</ins> [[Paper](https://arxiv.org/abs/1810.00825)]

##### Kernelization or Low-Rank
8. Loki: Low-Rank Keys for Efficient Sparse Attention, <ins>ICML Workshop, 2023</ins> [[Paper](https://arxiv.org/abs/2406.02542)]
9. Sumformer: Universal Approximation for Efficient Transformers, <ins>ICML Workshop, 2023</ins> [[Paper](https://arxiv.org/abs/2307.02301)]
10. FLuRKA: Fast fused Low-Rank & Kernel Attention, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2306.15799)]
11. Scatterbrain: Unifying Sparse and Low-rank Attention,  <ins>NeurlPS, 2021</ins> [[Paper](https://openreview.net/forum?id=SehIKudiIo1)] [[Code](https://github.com/HazyResearch/fly)]
12. Rethinking Attention with Performers,  <ins>ICLR, 2021</ins> [[Paper](https://openreview.net/forum?id=Ua6zuk0WRH)] [[Code](https://github.com/lucidrains/performer-pytorch)]
13. Random Feature Attention, <ins>ICLR, 2021</ins> [[Paper](https://arxiv.org/abs/2103.02143)]
14. Linformer: Self-Attention with Linear Complexity, <ins>arXiv, 2020</ins> [[Paper](https://arxiv.org/abs/2006.04768)] [[Code](https://github.com/lucidrains/linformer)]
15. Lightweight and Efficient End-to-End Speech Recognition Using Low-Rank Transformer, <ins>ICASSP, 2020</ins> [[Paper](https://arxiv.org/abs/1910.13923)]
16. Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention, <ins>ICML, 2020</ins> [[Paper](https://arxiv.org/abs/2006.16236)] [[Code](https://github.com/idiap/fast-transformers)]

##### Fixed Pattern Strategies
17. Simple linear attention language models balance the recall-throughput tradeoff, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2402.18668)] 
18. Lightning Attention-2: A Free Lunch for Handling Unlimited Sequence Lengths in Large Language Models, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2401.04658)] [[Code](https://github.com/OpenNLPLab/lightning-attention)]
19. Faster Causal Attention Over Large Sequences Through Sparse Flash Attention, <ins>ICML Workshop, 2023</ins> [[Paper](https://arxiv.org/abs/2306.01160)]
20. Poolingformer: Long Document Modeling with Pooling Attention, <ins>ICML, 2021</ins> [[Paper](https://arxiv.org/abs/2105.04371)]
21. Big Bird: Transformers for Longer Sequences, <ins>NeurIPS, 2020</ins> [[Paper](https://arxiv.org/abs/2007.14062)] [[Code](https://github.com/google-research/bigbird)]
22. Longformer: The Long-Document Transformer, <ins>arXiv, 2020</ins> [[Paper](https://arxiv.org/abs/2004.05150)] [[Code](https://github.com/allenai/longformer)]
23. Blockwise Self-Attention for Long Document Understanding, <ins>EMNLP, 2020</ins> [[Paper](https://arxiv.org/abs/1911.02972v)] [[Code](https://github.com/xptree/BlockBERT)]
24. Generating Long Sequences with Sparse Transformers, <ins>arXiv, 2019</ins> [[Paper](https://arxiv.org/abs/1904.10509)] 

##### Learnable Pattern Strategies
25. MoA: Mixture of Sparse Attention for Automatic Large Language Model Compression, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/html/2406.14909v1)]
26. HyperAttention: Long-context Attention in Near-Linear Time, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2310.05869)] [[Code](https://github.com/insuhan/hyper-attn)]
27. ClusterFormer: Neural Clustering Attention for Efficient and Effective Transformer, <ins>ACL, 2022</ins> [[Paper](https://aclanthology.org/2022.acl-long.170/)]
28. Reformer: The Efficient Transformer,  <ins>ICLR, 2022</ins> [[Paper](https://openreview.net/forum?id=rkgNKkHtvB)] [[Code](https://github.com/lucidrains/reformer-pytorch)]
29. Sparse Sinkhorn Attention, <ins>ICML, 2020</ins> [[Paper](https://arxiv.org/abs/2002.11296)] [[Code](https://github.com/lucidrains/sinkhorn-transformer?tab=readme-ov-file)]
30. Fast Transformers with Clustered Attention, <ins>NeurIPS, 2020</ins> [[Paper](https://arxiv.org/pdf/2007.04825.pdf)] [[Code](https://github.com/idiap/fast-transformers)]
31. Efficient Content-Based Sparse Attention with Routing Transformers, <ins>TACL, 2020</ins> [[Paper](https://arxiv.org/abs/2003.05997)] [[Code](https://github.com/google-research/google-research/tree/master/routing_transformer)] [[Code](https://github.com/lucidrains/routing-transformer?tab=readme-ov-file)]


#### Long Context LLMs

##### Extrapolation and Interpolation
32. Two Stones Hit One Bird: Bilevel Positional Encoding for Better Length Extrapolation, <ins>ICML, 2024</ins> [[Paper](https://arxiv.org/abs/2401.16421)]
33. ∞-Bench: Extending Long Context Evaluation Beyond 100K Tokens, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2402.13718)]
34. Resonance RoPE: Improving Context Length Generalization of Large Language Models, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2403.00071)] [[Code](https://github.com/sheryc/resonance_rope)]
35. LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2402.13753)]
36. E^2-LLM:Efficient and Extreme Length Extension of Large Language Models, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2401.06951)]
37. Scaling Laws of RoPE-based Extrapolation, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2310.05209)]
38. A Length-Extrapolatable Transformer, <ins>ACL, 2023</ins> [[Paper](https://aclanthology.org/2023.acl-long.816/)] [[Code](https://aka.ms/LeX-Transformer)]
39. Extending Context Window of Large Language Models via Positional Interpolation, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2306.15595)]
40. NTK Interpolation, <ins>Blog, 2023</ins> [[Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/)]
41. YaRN: Efficient Context Window Extension of Large Language Models, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2309.00071)] [[Code](https://github.com/jquesnelle/yarn)]
42. CLEX: Continuous Length Extrapolation for Large Language Models, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2310.16450)][[Code](https://github.com/DAMO-NLP-SG/CLEX)]
43. PoSE: Efficient Context Window Extension of LLMs via Positional Skip-wise Training, <ins>arXiv, 2023</ins> [[Paper](https://paperswithcode.com/paper/pose-efficient-context-window-extension-of)][[Code](https://github.com/dwzhu-pku/pose)]
44. Functional Interpolation for Relative Positions Improves Long Context Transformers, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/pdf/2310.04418.pdf)]
45. Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation, <ins>ICLR, 2022</ins> [[Paper](https://arxiv.org/pdf/2108.12409.pdf)] [[Code](https://github.com/ofirpress/attention_with_linear_biases)]
46. Exploring Length Generalization in Large Language Models, <ins>NeurIPS, 2022</ins> [[Paper](https://arxiv.org/abs/2207.04901)]

##### Recurrent Structure
47. Recurrent Memory Transformer, <ins>NeurIPS, 2022</ins> [[Paper](https://arxiv.org/abs/2207.06881)] [[Code](https://github.com/booydar/LM-RMT)] [[Code](https://github.com/lucidrains/recurrent-memory-transformer-pytorch)]
48. Block-Recurrent Transformers, <ins>NeurIPS, 2022</ins> [[Paper](https://arxiv.org/abs/2203.07852)] [[Code](https://github.com/google-research/meliad)] [[Code](https://github.com/lucidrains/block-recurrent-transformer-pytorch?tab=readme-ov-file)]
49. ∞-former: Infinite Memory Transformer, <ins>ACL, 2022</ins> [[Paper](https://arxiv.org/abs/2109.00301)] [[Code](https://github.com/deep-spin/infinite-former)]
50. Memformer: A Memory-Augmented Transformer for Sequence Modeling, <ins>AACL-Findings, 2020</ins> [[Paper]](https://arxiv.org/abs/2010.06891) [[Code](https://github.com/deep-spin/infinite-former)]
51. Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context, <ins>ACL, 2019</ins> [[Paper](https://arxiv.org/abs/1901.02860)] [[Code](https://github.com/kimiyoung/transformer-xl)]

##### Segmentation and Sliding Window
52. XL3M: A Training-free Framework for LLM Length Extension Based on Segment-wise Inference, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2405.17755)]
53. TransformerFAM: Feedback attention is working memory, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2404.09173)]
54. Naive Bayes-based Context Extension for Large Language Models, <ins>NAACL, 2024</ins> [[Paper](https://arxiv.org/abs/2403.17552)]
55. Training LLMs over Neurally Compressed Text, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2404.03626)]
56. LM-Infinite: Zero-Shot Extreme Length Generalization for Large Language Models, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2308.16137v6)]
57. Training-Free Long-Context Scaling of Large Language Models, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2402.17463)] [[Code](https://github.com/HKUNLP/ChunkLlama)]
58. Long-Context Language Modeling with Parallel Context Encoding, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2402.16617)] [[Code](https://github.com/princeton-nlp/CEPE)]
59. Soaring from 4K to 400K: Extending LLM’s Context with Activation Beacon, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2401.03462)] [[Code](https://github.com/FlagOpen/FlagEmbedding)]
60. LLM Maybe LongLM: Self-Extend LLM Context Window Without Tuning, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2305.15265)] [[Code](https://github.com/datamllab/LongLM)]
61. Extending Context Window of Large Language Models via Semantic Compression, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2312.09571)]
62. Efficient Streaming Language Models with Attention Sinks, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2309.17453)] [[Code](https://github.com/mit-han-lab/streaming-llm)]
63. Parallel Context Windows for Large Language Models, <ins>ACL, 2023</ins> [[Paper](https://arxiv.org/abs/2212.10947)] [[Code](https://github.com/ai21labs/parallel-context-windows)]
64. LongNet: Scaling Transformers to 1,000,000,000 Tokens, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2307.02486)] [[Code](https://github.com/microsoft/unilm/tree/master)]
65. Efficient Long-Text Understanding with Short-Text Models, <ins>TACL, 2023</ins> [[Paper](https://arxiv.org/abs/2208.00748)] [[Code](https://github.com/Mivg/SLED)]


#### Transformer Alternative Architecture

##### State Space Models
66. DenseMamba: State Space Models with Dense Hidden Connection for Efficient Large Language Models, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2403.00818)] [[Code](https://github.com/WailordHe/DenseSSM)]
67. MambaByte: Token-free Selective State Space Model, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2401.13660)] 
68. Sparse Modular Activation for Efficient Sequence Modeling, <ins>NeurIPS, 2023</ins> [[Paper](https://arxiv.org/abs/2306.11197)] [[Code](https://github.com/renll/SeqBoat)]
69. Long Range Language Modeling via Gated State Spaces, <ins>ICLR, 2023</ins> [[Paper](https://arxiv.org/abs/2206.13947)] [[Code](https://github.com/lucidrains/gated-state-spaces-pytorch)]
70. Block-State Transformers, <ins>NeurIPS, 2023</ins> [[Paper](https://arxiv.org/abs/2306.09539)]
71. Diagonal State Spaces are as Effective as Structured State Spaces, <ins>NeurIPS, 2022</ins> [[Paper](https://arxiv.org/abs/2203.14343)] [[Code](https://github.com/ag1988/dss)]
72. Hungry Hungry Hippos: Towards Language Modeling with State Space Models, <ins>ICLR 2023</ins> [[Paper](https://arxiv.org/abs/2212.14052)] [[Code](https://github.com/HazyResearch/H3)]

##### Other Sequential Models
73. Scalable MatMul-free Language Modeling, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2406.02528)]
74. MEGALODON: Efficient LLM Pretraining and Inference with Unlimited Context Length, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2404.08801)]
75. PanGu-π: Enhancing Language Model Architectures via Nonlinearity Compensation, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2312.17276)]


## Selected from Lucidrains (https://github.com/LAION-AI/lucidrains-projects) (30)

### Paper implementations
1. Compressive Transformers for Long-Range Sequence Modelling, https://github.com/lucidrains/compressive-transformer-pytorch?tab=readme-ov-file, ICLR 2020
2. Large Memory Layers with Product Keys, https://github.com/lucidrains/product-key-memory?tab=readme-ov-file, NeurIPS 2019
3. Flash Attention, https://github.com/lucidrains/flash-attention 
4. MetaFormer Is Actually What You Need for Vision (AR Version), https://github.com/lucidrains/metaformer-gpt?tab=readme-ov-file, CVPR 2022 Oral
5. VN-Transformer: Rotation-Equivariant Attention for Vector Neurons, https://github.com/lucidrains/VN-transformer?tab=readme-ov-file, TMLR 2023
6. Compositional Attention: Disentangling Search and Retrieval, https://github.com/lucidrains/compositional-attention-pytorch, ICLR 2022
7. Transformer Quality in Linear Time, https://github.com/lucidrains/FLASH-pytorch, ICML 2022
8. Self-attention Does Not Need O(n²) Memory, https://github.com/lucidrains/memory-efficient-attention-pytorch?tab=readme-ov-file
9. Sparse Attention with Linear Units (ReLA), https://github.com/lucidrains/rela-transformer?tab=readme-ov-file, EMNLP 2021
10. N-grammer: Augmenting Transformers with latent n-grams, https://github.com/lucidrains/n-grammer-pytorch?tab=readme-ov-file, Google
11. Remixer, https://github.com/lucidrains/remixer-pytorch?tab=readme-ov-file
12. Mogrifier LSTM, https://github.com/lucidrains/mogrifier?tab=readme-ov-file, ICLR 2020 Oral
13. H-Transformer-1D: Fast One-Dimensional Hierarchical Attention for Sequences, https://github.com/lucidrains/h-transformer-1d?tab=readme-ov-file, ACL 2021 Oral
14. Long-Short Transformer: Efficient Transformers for Language and Vision, https://github.com/lucidrains/long-short-transformer, NeurIPS 2021
15. RoFormer: Enhanced Transformer with Rotary Position Embedding, https://github.com/lucidrains/rotary-embedding-torch?tab=readme-ov-file
16. Pay Attention to MLPs, https://github.com/lucidrains/g-mlp-gpt?tab=readme-ov-file, Google
17. Addressing Some Limitations of Transformers with Feedback Memory, https://github.com/lucidrains/feedback-transformer-pytorch?tab=readme-ov-file, arXiv 2002, FAIR
18. Kronecker Attention Networks, https://github.com/lucidrains/kronecker-attention-pytorch?tab=readme-ov-file, KDD 2020
19. Generating Wikipedia by Summarizing Long Sequences, https://github.com/lucidrains/memory-compressed-attention?tab=readme-ov-file, ICLR 2018
20. Normalized Attention Without Probability Cage, https://github.com/lucidrains/all-normalization-transformer?tab=readme-ov-file, arXiv 2005

### Experimental projects
21. Token Shift GPT, https://github.com/lucidrains/token-shift-gpt 
22. Linear Attention Transformer, https://github.com/lucidrains/linear-attention-transformer
23. x-Transformers, https://github.com/lucidrains/x-transformers
24. Local attention, https://github.com/lucidrains/local-attention 
25. Self Reasoning Tokens (wip), https://github.com/lucidrains/self-reasoning-tokens-pytorch
26. Flash Cosine Similarity Attention (wip), https://github.com/lucidrains/flash-cosine-sim-attention, Lucidrains
27. Memory Transformer-XL (wip), https://github.com/lucidrains/memory-transformer-xl, Lucidrains
28. Simple Hierarchical Transformer (wip), https://github.com/lucidrains/simple-hierarchical-transformer, Lucidrains
29. Panoptic Transformer (wip), https://github.com/lucidrains/panoptic-transformer, Lucidrains
30. Building Blocks for a Complex-Valued Transformer Architecture, https://github.com/lucidrains/complex-valued-transformer?tab=readme-ov-file, ICASSP 2023