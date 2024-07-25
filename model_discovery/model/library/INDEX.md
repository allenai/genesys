# Language Modeling Library (256)
Library of methods and theory that may inspire the language modeling designs. Around 256 major papers & projects, and other interesting stuffs.

## Baselines and Core Reference Designs (155)
Major and latest branches of (autoregressive) model architecture variants after Transformers/GPTs. Collected from surveys, popular repos, and recent papers. Excluded MoE, Hierarchical and Heterogeneous Architectures, and non-causal models. Only consider identical single causal blocks. Can use S2 with reference type to build the phylogenetic tree later. Do not considering param-sharing like Albert. Sheet: https://docs.google.com/spreadsheets/d/1GxMjIY-RZWChS6g03NP9q4kv9tRjvQ9N8ZBkHIPvR1Y/edit?usp=sharing

### Collected Varaints (50)

#### Example Variants
1. GPT, Language Models are Few-Shot Learners
2. ✅ TTT, Learning to (Learn at Test Time): RNNs with Expressive Hidden States, https://github.com/test-time-training/ttt-lm-pytorch
3. 👉 xLSTM, xLSTM: Extended Long Short-Term Memory, https://github.com/NX-AI/xlstm
4. Griffin, Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models, https://github.com/google-deepmind/recurrentgemma
5. Hyena, Hyena Hierarchy: Towards Larger Convolutional Language Models (ICML'23 Oral), https://github.com/HazyResearch/safari
6. M2, Monarch Mixer: A Simple Sub-Quadratic GEMM-Based Architecture, https://github.com/HazyResearch/m2
7. SpikeGPT, SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks (TMLR,24), https://github.com/ridgerchu/SpikeGPT 

#### SSMs (https://github.com/state-spaces):
8. ✅ Mamba2, Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality
9. S4, Efficiently Modeling Long Sequences with Structured State Spaces
10. HiPPO, HiPPO: Recurrent Memory with Optimal Polynomial Projections
11. LSSL, Combining Recurrent, Convolutional, and Continuous-time Models with Linear State-Space Layers
12. HTTYH, How to Train Your HiPPO: State Space Models with Generalized Orthogonal Basis Projections
13. S4D, On the Parameterization and Initialization of Diagonal State Space Models
14. Mamba, Mamba: Linear-Time Sequence Modeling with Selective State Spaces

#### Flash Linear Attention (https://github.com/sustcsonglin/flash-linear-attention):
15. 2024-06	Samba, Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling	
16. ❎ 2023-07	RetNet, Retentive network: a successor to transformer for large language models	
17. 2023-12	GLA, Gated Linear Attention Transformers with Hardware-Efficient Training	
18. ✅ 2024-04	RWKV6, Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence	
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

#### From latest papers: 
30. Just read twice: closing the recall gap for recurrent language models (arXiv 2407)
31. Mega: Moving Average Equipped Gated Attention, https://github.com/lucidrains/Mega-pytorch?tab=readme-ov-file, ICLR 2023
32. DCMHA (ICML’24 Oral), Improving Transformers with Dynamically Composable Multi-Head Attention
33. AugLA (ICML’24), When Linear Attention Meets Autoregressive Decoding: Towards More Effective and Efficient Linearized Large Language Models (ICML 2024), https://github.com/GATECH-EIC/Linearized-LLM/blob/main/flash_pytorch.py 
34. Ring (ICLR’24), Ring Attention with Blockwise Transformers for Near-Infinite Context
35. BPT (NeurIPS’23), Blockwise Parallel Transformer for Large Context Models
36. Efficient Attention via Control Variates, EVA (ICLR’23 Oral)
37. DiJiang (ICML’24 Oral), DiJiang: Efficient Large Language Models through Compact Kernelization
38. TNN (ICLR’23 Spotlight), Toeplitz Neural Network for Sequence Modeling 
39. CoPE (arXiv 2405), Contextual Position Encoding: Learning to Count What's Important
40. Infiniti (arXiv 2404), Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention
41. PEER (arXiv 2407), Mixture of A Million Experts
42. CoLT5: Faster Long-Range Transformers with Conditional Computation, https://github.com/lucidrains/CoLT5-attention?tab=readme-ov-file, EMNLP 2023
43. Self-attention Networks Localize When QK-eigenspectrum Concentrates (ICML 2024)
44. Coneheads: Hierarchy Aware Attention, NeurIPS 2023
45. Linear Attention Sequence Parallelism

#### More (e.g., from ET Survey https://arxiv.org/pdf/2009.06732)
46. Synthesizer: Rethinking Self-Attention in Transformer Models, https://github.com/10-zin/Synthesizer?tab=readme-ov-file, ICML 2021
47. cosFormer: Rethinking Softmax in Attention, https://github.com/OpenNLPLab/cosFormer, ICLR 2022
48. LARA: Linear complexity randomized self-attention mechanism (ICML 2022)
49. Luna: Linear unified nested attention, https://github.com/sooftware/luna-transformer, NeurIPS 2021
50. VN-Transformer: Rotation-Equivariant Attention for Vector Neurons, https://github.com/lucidrains/VN-transformer?tab=readme-ov-file, TMLR 2023


### From *ELLM Survey* (https://github.com/AIoT-MLSys-Lab/Efficient-LLMs-Survey) (75)

#### Efficient Architecture

##### Efficient Attention

###### Sharing-based Attention
1. LoMA: Lossless Compressed Memory Attention, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2401.09486)]
2. MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2402.14905)]
3. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints, <ins>EMNLP, 2023</ins> [[Paper](https://arxiv.org/abs/2305.13245)]
4. Fast Transformer Decoding: One Write-Head is All You Need, <ins>arXiv, 2019</ins> [[Paper](https://arxiv.org/abs/1911.02150)]

###### Feature Information Reduction
5. Nyströmformer: A nyström-based algorithm for approximating self-attention, <ins>AAAI, 2021</ins> [[Paper](https://arxiv.org/abs/2102.03902)] [[Code](https://github.com/mlpen/Nystromformer)] [[Code](https://github.com/lucidrains/nystrom-attention)]
6. Funnel-Transformer: Filtering out Sequential Redundancy for Efficient Language Processing, <ins>NeurIPS, 2020</ins> [[Paper](https://arxiv.org/abs/2006.03236)] [[Code](https://github.com/laiguokun/Funnel-Transformer)]
7. Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks, <ins>ICML, 2019</ins> [[Paper](https://arxiv.org/abs/1810.00825)]

###### Kernelization or Low-Rank
8. Loki: Low-Rank Keys for Efficient Sparse Attention, <ins>ICML Workshop, 2023</ins> [[Paper](https://arxiv.org/abs/2406.02542)]
9. Sumformer: Universal Approximation for Efficient Transformers, <ins>ICML Workshop, 2023</ins> [[Paper](https://arxiv.org/abs/2307.02301)]
10. FLuRKA: Fast fused Low-Rank & Kernel Attention, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2306.15799)]
11. Scatterbrain: Unifying Sparse and Low-rank Attention,  <ins>NeurlPS, 2021</ins> [[Paper](https://openreview.net/forum?id=SehIKudiIo1)] [[Code](https://github.com/HazyResearch/fly)]
12. Rethinking Attention with Performers,  <ins>ICLR, 2021</ins> [[Paper](https://openreview.net/forum?id=Ua6zuk0WRH)] [[Code](https://github.com/lucidrains/performer-pytorch)]
13. Random Feature Attention, <ins>ICLR, 2021</ins> [[Paper](https://arxiv.org/abs/2103.02143)]
14. Linformer: Self-Attention with Linear Complexity, <ins>arXiv, 2020</ins> [[Paper](https://arxiv.org/abs/2006.04768)] [[Code](https://github.com/lucidrains/linformer)]
15. Lightweight and Efficient End-to-End Speech Recognition Using Low-Rank Transformer, <ins>ICASSP, 2020</ins> [[Paper](https://arxiv.org/abs/1910.13923)]
16. Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention, <ins>ICML, 2020</ins> [[Paper](https://arxiv.org/abs/2006.16236)] [[Code](https://github.com/idiap/fast-transformers)]

###### Fixed Pattern Strategies
17. Simple linear attention language models balance the recall-throughput tradeoff, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2402.18668)] 
18. Lightning Attention-2: A Free Lunch for Handling Unlimited Sequence Lengths in Large Language Models, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2401.04658)] [[Code](https://github.com/OpenNLPLab/lightning-attention)]
19. Faster Causal Attention Over Large Sequences Through Sparse Flash Attention, <ins>ICML Workshop, 2023</ins> [[Paper](https://arxiv.org/abs/2306.01160)]
20. Poolingformer: Long Document Modeling with Pooling Attention, <ins>ICML, 2021</ins> [[Paper](https://arxiv.org/abs/2105.04371)]
21. Big Bird: Transformers for Longer Sequences, <ins>NeurIPS, 2020</ins> [[Paper](https://arxiv.org/abs/2007.14062)] [[Code](https://github.com/google-research/bigbird)]
22. Longformer: The Long-Document Transformer, <ins>arXiv, 2020</ins> [[Paper](https://arxiv.org/abs/2004.05150)] [[Code](https://github.com/allenai/longformer)]
23. Blockwise Self-Attention for Long Document Understanding, <ins>EMNLP, 2020</ins> [[Paper](https://arxiv.org/abs/1911.02972v)] [[Code](https://github.com/xptree/BlockBERT)]
24. Generating Long Sequences with Sparse Transformers, <ins>arXiv, 2019</ins> [[Paper](https://arxiv.org/abs/1904.10509)] 

###### Learnable Pattern Strategies
25. MoA: Mixture of Sparse Attention for Automatic Large Language Model Compression, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/html/2406.14909v1)]
26. HyperAttention: Long-context Attention in Near-Linear Time, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2310.05869)] [[Code](https://github.com/insuhan/hyper-attn)]
27. ClusterFormer: Neural Clustering Attention for Efficient and Effective Transformer, <ins>ACL, 2022</ins> [[Paper](https://aclanthology.org/2022.acl-long.170/)]
28. Reformer: The Efficient Transformer,  <ins>ICLR, 2022</ins> [[Paper](https://openreview.net/forum?id=rkgNKkHtvB)] [[Code](https://github.com/lucidrains/reformer-pytorch)]
29. Sparse Sinkhorn Attention, <ins>ICML, 2020</ins> [[Paper](https://arxiv.org/abs/2002.11296)] [[Code](https://github.com/lucidrains/sinkhorn-transformer?tab=readme-ov-file)]
30. Fast Transformers with Clustered Attention, <ins>NeurIPS, 2020</ins> [[Paper](https://arxiv.org/pdf/2007.04825.pdf)] [[Code](https://github.com/idiap/fast-transformers)]
31. Efficient Content-Based Sparse Attention with Routing Transformers, <ins>TACL, 2020</ins> [[Paper](https://arxiv.org/abs/2003.05997)] [[Code](https://github.com/google-research/google-research/tree/master/routing_transformer)] [[Code](https://github.com/lucidrains/routing-transformer?tab=readme-ov-file)]


##### Long Context LLMs

###### Extrapolation and Interpolation
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

###### Recurrent Structure
47. Recurrent Memory Transformer, <ins>NeurIPS, 2022</ins> [[Paper](https://arxiv.org/abs/2207.06881)] [[Code](https://github.com/booydar/LM-RMT)] [[Code](https://github.com/lucidrains/recurrent-memory-transformer-pytorch)]
48. Block-Recurrent Transformers, <ins>NeurIPS, 2022</ins> [[Paper](https://arxiv.org/abs/2203.07852)] [[Code](https://github.com/google-research/meliad)] [[Code](https://github.com/lucidrains/block-recurrent-transformer-pytorch?tab=readme-ov-file)]
49. ∞-former: Infinite Memory Transformer, <ins>ACL, 2022</ins> [[Paper](https://arxiv.org/abs/2109.00301)] [[Code](https://github.com/deep-spin/infinite-former)]
50. Memformer: A Memory-Augmented Transformer for Sequence Modeling, <ins>AACL-Findings, 2020</ins> [[Paper]](https://arxiv.org/abs/2010.06891) [[Code](https://github.com/deep-spin/infinite-former)]
51. Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context, <ins>ACL, 2019</ins> [[Paper](https://arxiv.org/abs/1901.02860)] [[Code](https://github.com/kimiyoung/transformer-xl)]

###### Segmentation and Sliding Window
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


##### Transformer Alternative Architecture

###### State Space Models
66. DenseMamba: State Space Models with Dense Hidden Connection for Efficient Large Language Models, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2403.00818)] [[Code](https://github.com/WailordHe/DenseSSM)]
67. MambaByte: Token-free Selective State Space Model, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2401.13660)] 
68. Sparse Modular Activation for Efficient Sequence Modeling, <ins>NeurIPS, 2023</ins> [[Paper](https://arxiv.org/abs/2306.11197)] [[Code](https://github.com/renll/SeqBoat)]
69. Long Range Language Modeling via Gated State Spaces, <ins>ICLR, 2023</ins> [[Paper](https://arxiv.org/abs/2206.13947)] [[Code](https://github.com/lucidrains/gated-state-spaces-pytorch)]
70. Block-State Transformers, <ins>NeurIPS, 2023</ins> [[Paper](https://arxiv.org/abs/2306.09539)]
71. Diagonal State Spaces are as Effective as Structured State Spaces, <ins>NeurIPS, 2022</ins> [[Paper](https://arxiv.org/abs/2203.14343)] [[Code](https://github.com/ag1988/dss)]
72. Hungry Hungry Hippos: Towards Language Modeling with State Space Models, <ins>ICLR 2023</ins> [[Paper](https://arxiv.org/abs/2212.14052)] [[Code](https://github.com/HazyResearch/H3)]

###### Other Sequential Models
73. Scalable MatMul-free Language Modeling, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2406.02528)]
74. MEGALODON: Efficient LLM Pretraining and Inference with Unlimited Context Length, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2404.08801)]
75. PanGu-π: Enhancing Language Model Architectures via Nonlinearity Compensation, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2312.17276)]


### Other, Selects from Lucidrains (https://github.com/LAION-AI/lucidrains-projects) (30)

#### Paper implementations
1. Compressive Transformers for Long-Range Sequence Modelling, https://github.com/lucidrains/compressive-transformer-pytorch?tab=readme-ov-file, ICLR 2020
2. Large Memory Layers with Product Keys, https://github.com/lucidrains/product-key-memory?tab=readme-ov-file, NeurIPS 2019
3. Flash Attention, https://github.com/lucidrains/flash-attention 
4. MetaFormer Is Actually What You Need for Vision (AR Version), https://github.com/lucidrains/metaformer-gpt?tab=readme-ov-file, CVPR 2022 Oral
5. Generating Wikipedia by Summarizing Long Sequences, https://github.com/lucidrains/memory-compressed-attention?tab=readme-ov-file, ICLR 2018
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
19. Normalized Attention Without Probability Cage, https://github.com/lucidrains/all-normalization-transformer?tab=readme-ov-file, arXiv 2005
20. Building Blocks for a Complex-Valued Transformer Architecture, https://github.com/lucidrains/complex-valued-transformer?tab=readme-ov-file, ICASSP 2023

#### More papers
21. Hierarchical Transformers Are More Efficient Language Models (NAACL 2022 Findings)

22. An Attention Free Transformer (arXiv 2021, Apple)

23. Efficient Beam Tree Recursion (NeurIPS 2023)

24. Fast-R2D2: A Pretrained Recursive Neural Network based on Pruned CKY for Grammar Induction and Text Representation (EMNLP 2022)

25. ChordMixer: A Scalable Neural Attention Model for Sequences with Different Lengths (ICLR 2023)

26. Temporal Latent Bottleneck: Synthesis of Fast and Slow Processing Mechanisms in Sequence Learning (NeurIPS 2022)

27. BP-Transformer: Modelling Long-Range Context via Binary Partitioning (arXiv 2019)

28. Time-aware Large Kernel Convolutions (ICML 2020)

29. Simplified State Space Layers for Sequence Modeling (ICLR 2023 Oral)

30. Staircase Attention for Recurrent Processing of Sequences (NeurIPS 2022)


## Extention (e.g. from https://github.com/Xnhyacinth/Awesome-LLM-Long-Context-Modeling) (101)


### 2. Efficient Attention

#### 2.1 Sparse Attention

4. [**ETC: Encoding Long and Structured Inputs in Transformers.**](https://aclanthology.org/2020.emnlp-main.19/) *Joshua Ainslie, Santiago Ontanon, Chris Alberti, Vaclav Cvicek, Zachary Fisher, Philip Pham, Anirudh Ravula, Sumit Sanghai, Qifan Wang, Li Yang.* EMNLP 2020.

8. [**Sparse and continuous attention mechanisms.**](https://arxiv.org/abs/2006.07214) *André F. T. Martins, António Farinhas, Marcos Treviso, Vlad Niculae, Pedro M. Q. Aguiar, Mário A. T. Figueiredo.* NIPS 2020. 

10. [**LongT5: Efficient text-to-text transformer for long sequences.**](https://aclanthology.org/2022.findings-naacl.55/) *Mandy Guo, Joshua Ainslie, David Uthus, Santiago Ontanon, Jianmo Ni, Yun-Hsuan Sung, Yinfei Yang.* NAACL 2022.  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [![GitHub Repo stars](https://img.shields.io/github/stars/google-research/longt5)](https://github.com/google-research/longt5)

13. [**Unlimiformer: Long-Range Transformers with Unlimited Length Input.**](https://arxiv.org/abs/2305.01625) *Amanda Bertsch, Uri Alon, Graham Neubig, Matthew R. Gormley.* Arxiv 2023.  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [![GitHub Repo stars](https://img.shields.io/github/stars/abertsch72/unlimiformer)](https://github.com/abertsch72/unlimiformer)

14. [**Landmark Attention: Random-Access Infinite Context Length for Transformers.**](https://arxiv.org/abs/2305.16300) *Amirkeivan Mohtashami, Martin Jaggi* Arxiv 2023.  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [![GitHub Repo stars](https://img.shields.io/github/stars/epfml/landmark-attention)](https://github.com/epfml/landmark-attention)

16. [**Adapting Language Models to Compress Contexts.**](https://arxiv.org/abs/2305.14788) *Alexis Chevalier, Alexander Wettig, Anirudh Ajith, Danqi Chen.* Arxiv 2023.  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [![GitHub Repo stars](https://img.shields.io/github/stars/princeton-nlp/AutoCompressors)](https://github.com/princeton-nlp/AutoCompressors)

18. [**MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers.**](https://arxiv.org/abs/2305.07185) *Lili Yu, Dániel Simig, Colin Flaherty, Armen Aghajanyan, Luke Zettlemoyer, Mike Lewis.* Arxiv 2023.  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [![GitHub Repo stars](https://img.shields.io/github/stars/lucidrains/MEGABYTE-pytorch)](https://github.com/lucidrains/MEGABYTE-pytorch)

19. [**Dynamic Context Pruning for Efficient and Interpretable Autoregressive Transformers.**](https://arxiv.org/abs/2305.15805) *Sotiris Anagnostidis, Dario Pavllo, Luca Biggio, Lorenzo Noci, Aurelien Lucchi, Thomas Hofmann.* Arxiv 2023. 

20. [**Long-range Language Modeling with Self-retrieval.**](https://arxiv.org/abs/2306.13421) *Ohad Rubin, Jonathan Berant.* Arxiv 2023. 

21. [**Max-Margin Token Selection in Attention Mechanism.**](https://arxiv.org/abs/2306.13596) *Davoud Ataee Tarzanagh, Yingcong Li, Xuechen Zhang, Samet Oymak.* Arxiv 2023. 

22. [**Chunk, Align, Select: A Simple Long-sequence Processing Method for Transformers.**](https://arxiv.org/abs/2308.13191) *Jiawen Xie, Pengyu Cheng, Xiao Liang, Yong Dai, Nan Du.* Arxiv 2023. 

23. [**Sparse Token Transformer with Attention Back Tracking.**](https://openreview.net/forum?id=VV0hSE8AxCw) *Heejun Lee, Minki Kang, Youngwan Lee, Sung Ju Hwang.* ICLR 2023. 

24. [**Empower Your Model with Longer and Better Context Comprehension.**](https://arxiv.org/pdf/2307.13365v2.pdf) *YiFei Gao, Lei Wang, Jun Fang, Longhua Hu, Jun Cheng.* Arxiv 2023.  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [![GitHub Repo stars](https://img.shields.io/github/stars/yileijin/attention-transition)](https://github.com/yileijin/attention-transition)

31. [**LongHeads: Multi-Head Attention is Secretly a Long Context Processor.**](https://arxiv.org/abs/2402.10685) *Yi Lu, Xin Zhou, Wei He, Jun Zhao, Tao Ji, Tao Gui, Qi Zhang, Xuanjing Huang.* Arxiv 2024.

32. [**Zebra: Extending Context Window with Layerwise Grouped Local-Global Attention.**](https://arxiv.org/abs/2312.08618) *Kaiqiang Song, Xiaoyang Wang, Sangwoo Cho, Xiaoman Pan, Dong Yu.* Arxiv 2023.

34. [**Sequence can Secretly Tell You What to Discard.**](https://arxiv.org/abs/2404.15949) *Jincheng Dai, Zhuowei Huang, Haiyun Jiang, Chen Chen, Deng Cai, Wei Bi, Shuming Shi.* Arxiv 2024.

35. [**SinkLoRA: Enhanced Efficiency and Chat Capabilities for Long-Context Large Language Models.**](https://arxiv.org/abs/2406.05678) *Hengyu Zhang.* Arxiv 2024.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [![GitHub Repo stars](https://img.shields.io/github/stars/Dexter-GT-86/SinkLoRA)](https://github.com/Dexter-GT-86/SinkLoRA)

36. [**HiP Attention: Sparse Sub-Quadratic Attention with Hierarchical Attention Pruning.**](https://arxiv.org/abs/2406.09827) *Heejun Lee, Geon Park, Youngwan Lee, Jina Kim, Wonyoung Jeong, Myeongjae Jeon, Sung Ju Hwang.* Arxiv 2024.

37. [**Taking a Deep Breath: Enhancing Language Modeling of Large Language Models with Sentinel Tokens.**](https://arxiv.org/abs/2406.10985) *Weiyao Luo, Suncong Zheng, Heming Xia, Weikang Wang, Yan Lei, Tianyu Liu, Shuang Chen, Zhifang Sui.* Arxiv 2024.

39. [**Sparser is Faster and Less is More: Efficient Sparse Attention for Long-Range Transformers.**](https://arxiv.org/abs/2406.16747) *Chao Lou, Zixia Jia, Zilong Zheng, Kewei Tu.* Arxiv 2024.

40. [**Near-Lossless Acceleration of Long Context LLM Inference with Adaptive Structured Sparse Attention.**](https://arxiv.org/abs/2406.15486) *Qianchao Zhu, Jiangfei Duan, Chang Chen, Siran Liu, Xiuhong Li, Guanyu Feng, Xin Lv, Huanqi Cao, Xiao Chuanfu, Xingcheng Zhang, Dahua Lin, Chao Yang.* Arxiv 2024.

41. [**Neurocache: Efficient Vector Retrieval for Long-range Language Modeling.**](https://arxiv.org/abs/2407.02486) *Ali Safaya, Deniz Yuret.* Arxiv 2024.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [![GitHub Repo stars](https://img.shields.io/github/stars/alisafaya/neurocache)](https://github.com/alisafaya/neurocache)

42. [**Weighted Grouped Query Attention in Transformers.**](https://arxiv.org/abs/2407.10855) *Sai Sena Chinnakonduru, Astarag Mohapatra.* Arxiv 2024.

#### 2.2 Linear Attention

2. [**Learning Fast Algorithms for Linear Transforms Using Butterfly Factorizations.**](https://arxiv.org/abs/1903.05895) *Tri Dao, Albert Gu, Matthew Eichhorn, Atri Rudra, Christopher Ré.* Arxiv 2019.  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [![GitHub Repo stars](https://img.shields.io/github/stars/HazyResearch/butterfly)](https://github.com/HazyResearch/butterfly)

8. [**Fnet: Mixing tokens with fourier transforms.**](https://arxiv.org/abs/2105.03824) *James Lee-Thorp, Joshua Ainslie, Ilya Eckstein, Santiago Ontanon.* Arxiv 2021.  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [![GitHub Repo stars](https://img.shields.io/github/stars/jaketae/fnet)](https://github.com/jaketae/fnet)

10. [**Latent Attention for Linear Time Transformers.**](https://arxiv.org/abs/2402.17512) *Rares Dolga, Marius Cobzarenco, David Barber.* Arxiv 2024.  

13. [**Softmax Attention with Constant Cost per Token.**](https://arxiv.org/abs/2404.05843) *Franz A. Heinsen.* Arxiv 2024.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [![GitHub Repo stars](https://img.shields.io/github/stars/glassroom/heinsen_attention](https://github.com/glassroom/heinsen_attention)

15. [**Various Lengths, Constant Speed: Efficient Language Modeling with Lightning Attention.**](https://arxiv.org/abs/2405.17381) *Zhen Qin, Weigao Sun, Dong Li, Xuyang Shen, Weixuan Sun, Yiran Zhong.* Arxiv 2024.

16. [**Unlocking the Secrets of Linear Complexity Sequence Model from A Unified Perspective.**](https://arxiv.org/abs/2405.17383) *Zhen Qin, Xuyang Shen, Weigao Sun, Dong Li, Stan Birchfield, Richard Hartley, Yiran Zhong.* Arxiv 2024.

17. [**Attention as an RNN.**](https://arxiv.org/abs/2405.13956) *Leo Feng, Frederick Tung, Hossein Hajimirsadeghi, Mohamed Osama Ahmed, Yoshua Bengio, Greg Mori.* Arxiv 2024.

18. [**You Only Scan Once: Efficient Multi-dimension Sequential Modeling with LightNet.**](https://arxiv.org/abs/2405.21022) *Zhen Qin, Yuxin Mao, Xuyang Shen, Dong Li, Jing Zhang, Yuchao Dai, Yiran Zhong.* Arxiv 2024.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [![GitHub Repo stars](https://img.shields.io/github/stars/OpenNLPLab/LightNet](https://github.com/OpenNLPLab/LightNet)

#### 2.3 Hierarchical Attention

1. [**Neural Legal Judgment Prediction in English.**](https://aclanthology.org/P19-1424.pdf) *Ilias Chalkidis, Ion Androutsopoulos, Nikolaos Aletras.* ACL 2019. 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [![GitHub Repo stars](https://img.shields.io/github/stars/PolarisRisingWar/pytorch_ljp)](https://github.com/PolarisRisingWar/pytorch_ljp)

2. [**Hierarchical Neural Network Approaches for Long Document Classification.**](https://arxiv.org/abs/2201.06774) *Snehal Khandve, Vedangi Wagh, Apurva Wani, Isha Joshi, Raviraj Joshi.* ICML 2022. 

3. [**Hi-transformer: Hierarchical interactive transformer for efficient and effective long document modeling.**](https://arxiv.org/abs/2106.01040) *Chuhan Wu, Fangzhao Wu, Tao Qi, Yongfeng Huang.* ACL-IJCNLP 2021 

4. [**Erniesparse: Learning hierarchical efficient transformer through regularized self-attention.**](https://arxiv.org/abs/2203.12276) *Yang Liu, Jiaxiang Liu, Li Chen, Yuxiang Lu, Shikun Feng, Zhida Feng, Yu Sun, Hao Tian, Hua Wu, Haifeng Wang.* Arxiv 2022.


### 3. Recurrent Transformers

5. [**Memorizing Transformers.**](https://arxiv.org/abs/2203.08913) *Yuhuai Wu, Markus N. Rabe, DeLesley Hutchins, Christian Szegedy.* Arxiv 2022.  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [![GitHub Repo stars](https://img.shields.io/github/stars/lucidrains/memorizing-transformers-pytorch)](https://github.com/lucidrains/memorizing-transformers-pytorch)

6. [**Recurrent Attention Networks for Long-text Modeling.**](https://aclanthology.org/2023.findings-acl.188/) *Xianming Li, Zongxi Li, Xiaotian Luo, Haoran Xie, Xing Lee, Yingbin Zhao, Fu Lee Wang, Qing Li.* ACL 2023.  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [![GitHub Repo stars](https://img.shields.io/github/stars/4ai/ran)](https://github.com/4ai/ran)

8. [**Segmented Recurrent Transformer: An Efficient Sequence-to-Sequence Model.**](https://arxiv.org/abs/2305.16340) *Yinghan Long, Sayeed Shafayet Chowdhury, Kaushik Roy.* Arxiv 2023. 

11. [**TRAMS: Training-free Memory Selection for Long-range Language Modeling.**](https://arxiv.org/abs/2310.15494) *Haofei Yu, Cunxiang Wang, Yue Zhang, Wei Bi.* Arxiv 2023. 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [![GitHub Repo stars](https://img.shields.io/github/stars/lwaekfjlk/TRAMS)](https://github.com/lwaekfjlk/TRAMS)

13. [**Extensible Embedding: A Flexible Multipler For LLM's Context Length.**](https://arxiv.org/abs/2402.11577) *Ninglu Shao, Shitao Xiao, Zheng Liu, Peitian Zhang.* Arxiv 2024. 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [![GitHub Repo stars](https://img.shields.io/github/stars/FlagOpen/FlagEmbedding)](https://github.com/FlagOpen/FlagEmbedding)

17. [**Linearizing Large Language Models.**](https://arxiv.org/abs/2405.06640) *Jean Mercat, Igor Vasiljevic, Sedrick Keh, Kushal Arora, Achal Dave, Adrien Gaidon, Thomas Kollar.* Arxiv 2024. 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [![GitHub Repo stars](https://img.shields.io/github/stars/TRI-ML/linear_open_lm)](https://github.com/TRI-ML/linear_open_lm)

20. [**Associative Recurrent Memory Transformer.**](https://arxiv.org/abs/2407.04841) *Ivan Rodkin, Yuri Kuratov, Aydar Bulatov, Mikhail Burtsev.* ICML 2024 Workshop.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [![GitHub Repo stars](https://img.shields.io/github/stars/RodkinIvan/associative-recurrent-memory-transformer)](https://github.com/RodkinIvan/associative-recurrent-memory-transformer)


### 4. State Space Models

4. [**LOCOST: State-Space Models for Long Document Abstractive Summarization.**](https://arxiv.org/abs/2401.17919) *Florian Le Bronnec, Song Duong, Mathieu Ravaut, Alexandre Allauzen, Nancy F. Chen, Vincent Guigue, Alberto Lumbreras, Laure Soulier, Patrick Gallinari.* Arxiv 2024.

5. [**State Space Models as Foundation Models: A Control Theoretic Overview.**](https://arxiv.org/abs/2403.16899) *Carmen Amo Alonso, Jerome Sieber, Melanie N. Zeilinger.* Arxiv 2024.

7. [**Robustifying State-space Models for Long Sequences via Approximate Diagonalization.**](https://openreview.net/forum?id=DjeQ39QoLQ) *Annan Yu, Arnur Nigmetov, Dmitriy Morozov, Michael W. Mahoney, N. Benjamin Erichson.* ICLR 2024 Spotlight.

8. [**Zamba: A Compact 7B SSM Hybrid Model.**](https://arxiv.org/abs/2405.16712) *Paolo Glorioso, Quentin Anthony, Yury Tokpanov, James Whittington, Jonathan Pilault, Adam Ibrahim, Beren Millidge.* Arxiv 2024.

11. [**An Empirical Study of Mamba-based Language Models.**](https://arxiv.org/abs/2406.07887) *Roger Waleffe, Wonmin Byeon, Duncan Riach, Brandon Norick, Vijay Korthikanti, Tri Dao, Albert Gu, Ali Hatamizadeh, Sudhakar Singh, Deepak Narayanan, Garvit Kulshreshtha, Vartika Singh, Jared Casper, Jan Kautz, Mohammad Shoeybi, Bryan Catanzaro.* Arxiv 2024.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [![GitHub Repo stars](https://img.shields.io/github/stars/NVIDIA/Megatron-LM)](https://github.com/NVIDIA/Megatron-LM/tree/ssm/examples/mamba)

12. [**B'MOJO: Hybrid State Space Realizations of Foundation Models with Eidetic and Fading Memory.**](https://arxiv.org/abs/2407.06324) *Luca Zancato, Arjun Seshadri, Yonatan Dukler, Aditya Golatkar, Yantao Shen, Benjamin Bowman, Matthew Trager, Alessandro Achille, Stefano Soatto.* Arxiv 2024.

13. [**MambaForGCN: Enhancing Long-Range Dependency with State Space Model and Kolmogorov-Arnold Networks for Aspect-Based Sentiment Analysis.**](https://arxiv.org/abs/2407.10347) *Adamu Lawan, Juhua Pu, Haruna Yunusa, Aliyu Umar, Muhammad Lawan.* Arxiv 2024.

14. [**Discrete Diffusion Language Model for Long Text Summarization.**](https://arxiv.org/abs/2407.10998) *Do Huu Dat, Do Duc Anh, Anh Tuan Luu, Wray Buntine.* Arxiv 2024.

### 5. Length Extrapolation

3. [**KERPLE: Kernelized Relative Positional Embedding for Length Extrapolation.**](https://arxiv.org/abs/2205.09921) *Ta-Chung Chi, Ting-Han Fan, Peter J. Ramadge, Alexander I. Rudnicky.* Arxiv 2022. 

4. [**Dissecting Transformer Length Extrapolation via the Lens of Receptive Field Analysis.**](https://aclanthology.org/2023.acl-long.756/) *Ta-Chung Chi, Ting-Han Fan, Alexander I. Rudnicky, Peter J. Ramadge.* ACL 2023. 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [![GitHub Repo stars](https://img.shields.io/github/stars/McGill-NLP/length-generalization)](https://github.com/McGill-NLP/length-generalization)


10. [**Exploring Transformer Extrapolation.**](https://arxiv.org/abs/2307.10156) *Zhen Qin, Yiran Zhong, Hui Deng.* Arxiv 2023.  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [![GitHub Repo stars](https://img.shields.io/github/stars/OpenNLPLab/Rpe)](https://github.com/OpenNLPLab/Rpe)

16. [**Attention Alignment and Flexible Positional Embeddings Improve Transformer Length Extrapolation.**](https://arxiv.org/pdf/2311.00684v1.pdf) *Ta-Chung Chi,Ting-Han Fan,Alexander I. Rudnicky.* Arxiv 2023.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [![GitHub Repo stars](https://img.shields.io/github/stars/chijames/Attention-Alignment-Transformer-Length-Extrapolation)](https://github.com/chijames/Attention-Alignment-Transformer-Length-Extrapolation)

26. [**Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens.**](https://arxiv.org/abs/2401.17377) *Jiacheng Liu, Sewon Min, Luke Zettlemoyer, Yejin Choi, Hannaneh Hajishirzi.* Arxiv 2024.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [![GitHub Repo stars](https://img.shields.io/github/stars/liujch1998/infini-gram)](https://github.com/liujch1998/infini-gram)

41. [**Length Generalization of Causal Transformers without Position Encoding.**](https://arxiv.org/abs/2404.12224) *Jie Wang, Tao Ji, Yuanbin Wu, Hang Yan, Tao Gui, Qi Zhang, Xuanjing Huang, Xiaoling Wang.* Arxiv 2024.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [![GitHub Repo stars](https://img.shields.io/github/stars/AntNLP/nope_head_scale)](https://github.com/AntNLP/nope_head_scale)

45. [**CAPE: Context-Adaptive Positional Encoding for Length Extrapolation.**](https://arxiv.org/abs/2405.14722) *Chuanyang Zheng, Yihang Gao, Han Shi, Minbin Huang, Jingyao Li, Jing Xiong, Xiaozhe Ren, Michael Ng, Xin Jiang, Zhenguo Li, Yu Li.* Arxiv 2024.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [![GitHub Repo stars](https://img.shields.io/github/stars/chuanyang-Zheng/CAPE)](https://github.com/chuanyang-Zheng/CAPE)

48. [**Position Coupling: Leveraging Task Structure for Improved Length Generalization of Transformers.**](https://arxiv.org/abs/2405.20671) *Hanseul Cho, Jaeyoung Cha, Pranjal Awasthi, Srinadh Bhojanapalli, Anupam Gupta, Chulhee Yun.* Arxiv 2024.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [![GitHub Repo stars](https://img.shields.io/github/stars/HanseulJo/position-coupling)](https://github.com/HanseulJo/position-coupling)

50. [**Explicitly Encoding Structural Symmetry is Key to Length Generalization in Arithmetic Tasks.**](https://arxiv.org/abs/2406.01895) *Mahdi Sabbaghi, George Pappas, Hamed Hassani, Surbhi Goel.* Arxiv 2024.

52. [**3D-RPE: Enhancing Long-Context Modeling Through 3D Rotary Position Encoding.**](https://arxiv.org/abs/2406.09897) *Xindian Ma, Wenyuan Liu, Peng Zhang, Nan Xu.* Arxiv 2024.

54. [**Human-like Episodic Memory for Infinite Context LLMs.**](https://arxiv.org/abs/2407.09450) *Zafeirios Fountas, Martin A Benfeghoul, Adnan Oomerjee, Fenia Christopoulou, Gerasimos Lampouras, Haitham Bou-Ammar, Jun Wang.* Arxiv 2024.

55. [**Scaling Granite Code Models to 128K Context.**](https://arxiv.org/abs/2407.13739) *Matt Stallone, Vaibhav Saxena, Leonid Karlinsky, Bridget McGinn, Tim Bula, Mayank Mishra, Adriana Meza Soria, Gaoyuan Zhang, Aditya Prasad, Yikang Shen, Saptha Surendran, Shanmukha Guttula, Hima Patel, Parameswaran Selvam, Xuan-Hong Dang, Yan Koyfman, Atin Sood, Rogerio Feris, Nirmit Desai, David D. Cox, Ruchir Puri, Rameswar Panda.* Arxiv 2024.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [![GitHub Repo stars](https://img.shields.io/github/stars/ibm-granite/granite-code-models)](https://github.com/ibm-granite/granite-code-models)


### More

1. **(Arxiv 24.06.04) GrootVL: Tree Topology is All You Need in State Space Model** [Paper](https://arxiv.org/abs/2406.02395) [Code](https://github.com/EasonXiao-888/GrootVL) ![Stars](https://img.shields.io/github/stars/EasonXiao-888/GrootVL)

2. (Arxiv 24.05.26) A Unified Implicit Attention Formulation for Gated-Linear Recurrent Sequence Models [Paper](https://arxiv.org/abs/2405.16504) [Code](https://github.com/Itamarzimm/UnifiedImplicitAttnRepr) ![Stars](https://img.shields.io/github/stars/Itamarzimm/UnifiedImplicitAttnRepr)

3. (Arxiv 24.05.27) The Expressive Capacity of State Space Models: A Formal Language Perspective [Paper](https://arxiv.org/abs/2405.17394) [Code](https://github.com/LeapLabTHU/MLLA) ![Stars](https://img.shields.io/github/stars/LeapLabTHU/MLLA)

4. KAN or MLP: A Fairer Comparison, https://arxiv.org/pdf/2407.16674, https://github.com/yu-rp/KANbeFair 

5. Graph Language Models (ACL 2024), https://github.com/Heidelberg-NLP/GraphLanguageModels

6. Towards mental time travel: a hierarchical memory for reinforcement learning agents (NeurIPS 2021), https://github.com/lucidrains/HTM-pytorch?tab=readme-ov-file

7. Multi-Stream Transformers, https://github.com/lucidrains/multistream-transformers?tab=readme-ov-file 

8. Perceiver IO: A General Architecture for Structured Inputs & Outputs (ICLR 2022 Spotlight), https://github.com/lucidrains/perceiver-ar-pytorch

9. Axial Attention in Multidimensional Transformers, https://github.com/lucidrains/axial-attention?tab=readme-ov-file

10. (Arxiv 24.03.03) The Hidden Attention of Mamba Models [Paper](https://arxiv.org/abs/2403.01590) [Code ](https://github.com/AmeenAli/HiddenMambaAttn)![Stars](https://img.shields.io/github/stars/AmeenAli/HiddenMambaAttn)

11. (Arxiv 24.03.28) Jamba: A Hybrid Transformer-Mamba Language Model [Paper](https://arxiv.org/abs/2403.19887) [Code](https://huggingface.co/ai21labs/Jamba-v0.1) 

12. Sequence Modeling with Multiresolution Convolutional Memory (ICML 2023)

13. Resurrecting Recurrent Neural Networks for Long Sequences (Deepmind 2023)

14. HyperMixer: An MLP-based Low Cost Alternative to Transformers (ACL 2023)

15. Pay Less Attention with Lightweight and Dynamic Convolutions (ICLR 2019 Oral)

16. The Devil in Linear Transformer (EMNLP 2022)

17. Flowformer: Linearizing Transformers with Conservation Flows (ICML 2022)

18. Universal Transformers (ICLR 2019)

19. The Neural Data Router: Adaptive Control Flow in Transformers Improves Systematic Generalization (ICLR 2022)

20. Ordered Memory (NeurIPS 2019)

21. Modeling Hierarchical Structures with Continuous Recursive Neural Networks (ICML 2021)

22. SOFT: Softmax-free Transformer with Linear Complexity (NeurIPS 2021 Spotlight)

### Non standard models
1. Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution (ICML 2024 Best Paper Award), https://arxiv.org/abs/2310.16834
2. Semi-autoregressive Simplex-based Diffusion Language Model (ACL 2023), https://github.com/xhan77/ssd-lm 
3. DiffusionBERT: Improving Generative Masked Language Models with Diffusion Models (ACL 2023)
4. Structured Denoising Diffusion Models in Discrete State-Spaces (NeurIPS 2021)
5. Diffusion-LM Improves Controllable Text Generation (NeurIPS 2022)
6. Likelihood-Based Diffusion Language Models (NeurIPS 2023)
7. TESS: Text-to-Text Self-Conditioned Simplex Diffusion (arXiv 2023)
8. DiffuSeq: Sequence to Sequence Text Generation with Diffusion Models (ICLR 2023)
9. Continuous diffusion for categorical data (Deepmind 2022)
10. Classifier-Free Diffusion Guidance (arXiv 2022)
11. Self-conditioned Embedding Diffusion for Text Generation (arXiv 2023)
12. Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning (ICLR 2023)
13. A Reparameterized Discrete Diffusion Model for Text Generation (arXiv 2023)
14. Fast Sampling via De-randomization for Discrete Diffusion Models (arXiv 2023)
15. DINOISER: Diffused Conditional Sequence Learning by Manipulating Noises (TACL 2023)
16. KAN-GPT, https://github.com/AdityaNG/kan-gpt 
17. Self Reasoning Tokens (wip), https://github.com/lucidrains/self-reasoning-tokens-pytorch


## Community projects

1. Token Shift GPT, https://github.com/lucidrains/token-shift-gpt 
2. Flash Cosine Similarity Attention (wip), https://github.com/lucidrains/flash-cosine-sim-attention, Lucidrains
3. Memory Transformer-XL (wip), https://github.com/lucidrains/memory-transformer-xl, Lucidrains
4. Simple Hierarchical Transformer (wip), https://github.com/lucidrains/simple-hierarchical-transformer, Lucidrains
5. Panoptic Transformer (wip), https://github.com/lucidrains/panoptic-transformer, Lucidrains
6. Linear Attention Transformer, https://github.com/lucidrains/linear-attention-transformer
7. x-Transformers, https://github.com/lucidrains/x-transformers
8. Local attention, https://github.com/lucidrains/local-attention 