# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-10-19 11:14:19 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Efficient CPU-GPU collaborative MoE inference via Adaptive Expert Assignment](https://arxiv.org/abs/2610.23006)

**Authors**: Wenhui Lv, Yunming Liao, Hongli Xu, Yang Xu, Junhao Cheng  
**Category**: cs.AI  
**Published**: 2026-10-19  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2610.23006v1  

#### Abstract
Mixture-of-Experts (MoE) models reduce inference computation through sparse expert activation, but their large expert memory footprint often exceeds GPU capacity, making expert offloading necessary for memory-constrained inference. However, existing systems typically handle GPU-missed experts by eit...

---

### 2. [ScoutRoute: Efficient Mixture-of-Experts Routing Using Internal Language Model Signals](https://arxiv.org/abs/2610.23443)

**Authors**: Rajdeep Pandhare, Delzaan Bharucha  
**Category**: cs.AI  
**Published**: 2026-10-19  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2610.23443v1  

#### Abstract
Efficient expert routing is a fundamental challenge in Mixture-of-Experts (MoE) systems, particularly for heterogeneous collections of independently trained Large Language Mod els (LLMs). Existing routing approaches typically rely on jointly trained gating networks, prompt-based predictors, or compu...

---

### 3. [FuTransformer: FP8 prefill and INT4 decode for Llama-style inference on Hopper](https://arxiv.org/abs/2610.23796)

**Authors**: Wei Wang  
**Category**: cs.AI  
**Published**: 2026-10-19  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2610.23796v1  

#### Abstract
Transformer prefill reuses weights across tokens and is dominated by matrix multiplication and activation data movement; single-token decode has limited reuse and is dominated by weight and KV-cache reads. We present FUTRANSFORMER, a CUDA implementation of a Llama-style Transformer block for NVIDIA ...

---

### 4. [Where Research Ideation Begins: Forward-Citation Supervision for Actionable Limitation Generation](https://arxiv.org/abs/2610.23038)

**Authors**: Liyang Zhao, Yong Cao, Haibin Ling  
**Category**: cs.AI  
**Published**: 2026-10-19  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2610.23038v1  

#### Abstract
Automated research ideation has advanced rapidly, with multi-agent workflows that generate whole research ideas end to end. These frameworks leave largely unexamined the step that seeds a good idea: identifying an actionable limitation from literature. In this paper, we focus on this first step in t...

---

### 5. [HydraLLM: Adaptive Width Slicing for Elastic Text and Vision-Language Models](https://arxiv.org/abs/2610.23455)

**Authors**: Oscar Kemme, Janek Haberer, Ali Hojjat, Olaf Landsiedel  
**Category**: cs.CL  
**Published**: 2026-10-19  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2610.23455v1  

#### Abstract
LLMs and LVLMs are increasingly deployed across heterogeneous hardware, ranging from data-center GPUs to resource-constrained edge devices. Their fixed-size architectures limit the ability to reuse a single model across deployment settings with different compute, latency, and model-size targets. We ...

---

### 6. [DLite: History-free Speculative Decoding for Agentic AI](https://arxiv.org/abs/2610.23787)

**Authors**: Jiaming Xu, Hanzhen Wang, Tianlang Zhao, Jiewen Xiao, Chengze Yuan, Jiayi Pan, Yongkang Zhou, Jiancai Ye, Yu Wang, Guohao Dai  
**Category**: cs.CL  
**Published**: 2026-10-19  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2610.23787v1  

#### Abstract
Speculative decoding has become a mainstream technique for accelerating large language model (LLM) inference by using a lightweight draft model to generate candidate tokens that are subsequently verified by the target model. However, agentic workloads introduce long and continuously evolving context...

---

### 7. [DIFFLET: A Fast Inference Engine for Diffusion Transformer Models on AWS Trainium](https://arxiv.org/abs/2610.23047)

**Authors**: Bin Ma, Min Yu, Vijay Kodamalla, Jialin Liu, Dong Li  
**Category**: cs.DC  
**Published**: 2026-10-19  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2610.23047v1  

#### Abstract
On AWS Trainium2, every component of a diffusion model runs as a program compiled ahead of time for fixed shapes, while pipelines differ across models, prompt lengths across requests, and caching decisions across denoising steps. We present Difflet, a serving engine in which five model pipelines sha...

---

### 8. [Generative adversarial networks with Monte Carlo dropout uncertainty quantification for structural health monitoring: Fibre Bragg grating data augmentation comparison in Hawk T1A aircraft](https://arxiv.org/abs/2610.22436)

**Authors**: Alvaro Peiro, Marios Impraimakis  
**Category**: cs.LG  
**Published**: 2026-10-19  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2610.22436v1  

#### Abstract
The signal generation capabilities of a novel data augmentation framework are examined in aerospace structures. First, the framework learns class-conditional temporal and spectral characteristics of the measured structural responses. It uses conditional generative adversarial network training and Mo...

---

### 9. [On the SPOT: Enhancing Speculative Decoding for Diffusion Language Models via Commit-Order Prediction](https://arxiv.org/abs/2610.22543)

**Authors**: Eliezer Shlomi, Eliron Rahimi, Avi Mendelson, Chaim Baskin  
**Category**: cs.LG  
**Published**: 2026-10-19  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2610.22543v1  

#### Abstract
Diffusion language models (DLMs) promise parallel text generation, but in practice face an unresolved speed-quality trade-off. Exact decoding fills in only one masked position per model pass, while faster multi-token decoding schemes are lossy. Speculative decoding is the standard lossless accelerat...

---

### 10. [Noise-shaped Sampling and Annealing for Scalable On-the-edge Training of Binary and Ternary AI Models](https://arxiv.org/abs/2610.22621)

**Authors**: Zihao Chen, Zhili Xiao, Rui-Jie Zhu, Yipeng Wang, Jason Eshraghian, Shantanu Chakrabartty  
**Category**: cs.LG  
**Published**: 2026-10-19  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2610.22621v1  

#### Abstract
The training loss of binary and ternary neural networks, expanded as a multilinear polynomial over the discrete weights, is structurally equivalent to the energy of a hierarchical high-order Ising model. We exploit this equivalence to apply Ising solvers to discrete-weight network training, while ac...

---

### 11. [Beyond Reward Scoring: Online Multi-Agent Preference Learning for Multimodal Response Generation](https://arxiv.org/abs/2610.23256)

**Authors**: Ziyun Zou, Jiwoo Hong, Yifan Lu, Shao Tang, Weiyi Wu, Aman Gupta, Jin Tian, Imran Razzak  
**Category**: cs.AI  
**Published**: 2026-10-19  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2610.23256v1  

#### Abstract
Multimodal preference alignment often follows a score-based RLHF pipeline: a reward model is first trained to learn human preferences, then used to score on-policy responses and optimize the response model's generation. This turns preference supervision into a scalar signal for generation: a model m...

---

### 12. [CoTA++: Understanding the Repeat Curse in dMLLMs from an Information Flow Perspective](https://arxiv.org/abs/2610.23290)

**Authors**: Qiyan Zhao, Xiaofeng Zhang, Xu-Yao Zhang, Shijie Han, Zhepeng Li, Shunzhi Zhu, Da-Han Wang  
**Category**: cs.AI  
**Published**: 2026-10-19  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2610.23290v1  

#### Abstract
Diffusion-based multimodal large language models (dMLLMs) decode by iterative denoising with bidirectional attention, which precludes the standard autoregressive key--value cache. Practical deployments therefore rely on approximate caching that reuses intermediate states across denoising steps. Such...

---

### 13. [RL-Guided Temporal Localization for Dual-Channel Retrieval in Long-Horizon Agent Memory](https://arxiv.org/abs/2610.23598)

**Authors**: Dabin Sheng, Zhe Wu, Jinming Zhao, Junliang Xing, Yuanchun Shi  
**Category**: cs.AI  
**Published**: 2026-10-19  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2610.23598v1  

#### Abstract
Time-aware retrieval is crucial for long-horizon LLM agents to effectively leverage ever-growing long-term memory. Existing methods focus on lexical or semantic similarity, leading to target memories being crowded out by temporally mismatched yet semantically similar candidates. Even time-aware appr...

---

### 14. [BaGuaLu-2: Taming Dynamic Memory Fluctuation for Efficient MoE Model Training over 10,000 GPUs](https://arxiv.org/abs/2610.23064)

**Authors**: Mingshu Zhai, Minkun Guo, Hongrui Lian, Shuangyu Li, Qi Zhang, Chen Hu, Xiawei Du, Yuyang Jin, Zan Zong, Jidong Zhai  
**Category**: cs.DC  
**Published**: 2026-10-19  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2610.23064v1  

#### Abstract
Fine-grained Mixture-of-Experts (MoE) is the de facto architecture for scaling large language models. However, dynamic expert load imbalance causes severe memory fluctuations that degrade training efficiency. Existing systems rely on static memory provisioning: they either over-provision, leaving GP...

---

### 15. [Lean4AgenticRL: Advancing Agentic RL with Verifiable Multi-level Process Rewards](https://arxiv.org/abs/2610.22448)

**Authors**: Ruida Wang, Gerald Friedland, Tong Zhang, Chris Kong  
**Category**: cs.LG  
**Published**: 2026-10-19  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2610.22448v1  

#### Abstract
Enhancing the agentic capabilities of Large Language Models (LLMs) to tackle complex software engineering (SWE) and general terminal-usage problems is a pressing challenge. Current state-of-the-art LLMs often go through large-scale agentic Reinforcement Learning (RL). However, they mostly rely on hi...

---

### 16. [FlexUMM: Flexible Representation Learning for Unified Multimodal Models](https://arxiv.org/abs/2610.22682)

**Authors**: Yinyi Luo, Haoyue Bai, Yi Ge, Nilton Cesar Rojas Vales, Marios Savvides, Jindong Wang  
**Category**: cs.AI  
**Published**: 2026-10-19  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2610.22682v1  

#### Abstract
Unified multimodal models (UMMs) can integrate understanding and generation within a single backbone. However, it is commonly known that these two capabilities can cause conflicting representations, leading to a central question: how representation and computation should be shared across functionali...

---

### 17. [Accelerated Dataflow: scaling inference in streaming data analytics with efficiency](https://arxiv.org/abs/2610.22731)

**Authors**: RuiLong Jiang, Dustin Rhodes, Danny McCormick, Yuta Labur, Reza Rokni  
**Category**: cs.DC  
**Published**: 2026-10-19  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2610.22731v1  

#### Abstract
Streaming data analytics helps businesses identify opportunities and spot risks in a timely fashion. As large language model (LLM) usage grows, people are exploring incorporating agents or LLMs in streaming data pipelines to help find patterns in datasets without supervision or interpret natural lan...

---

### 18. [Pearl-Chat: Engineering a native Luganda language model from first principles in JAX and Flax NNX](https://arxiv.org/abs/2610.22529)

**Authors**: Wesley Kambale, Roland Sankara  
**Category**: cs.CL  
**Published**: 2026-10-19  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2610.22529v1  

#### Abstract
Autoregressive foundation language models predominantly target high-resource Indo-European languages, imposing computational and morphological penalties on under-represented African languages. Generic multilingual tokenizers routinely exceed fertility rates of 3.2 tokens per word on agglutinative Ba...

---

### 19. [Do Speech BCIs Need Larger Models? Rethinking Neural Decoding beyond Scaling](https://arxiv.org/abs/2610.23175)

**Authors**: Zehui Feng, Weichuan Wang, Xiaohan Chen, Cuntai Guan, Ting Han  
**Category**: cs.CL  
**Published**: 2026-10-19  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2610.23175v1  

#### Abstract
Recent speech brain-computer interfaces (BCIs) increasingly rely on large models and complex multi-stage pipelines for neural decoding. However, in speech BCI, where training data are scarce, noisy, and expensive to collect, the benefits of model scaling remain unclear. This work revisits a fundamen...

---

### 20. [Structure-Aware Spatial Kernels for Gaussian Process Distributed Port-Hamiltonian Systems](https://arxiv.org/abs/2610.22575)

**Authors**: Tianyi Yang, Thomas Beckers  
**Category**: cs.LG  
**Published**: 2026-10-19  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2610.22575v1  

#### Abstract
Gaussian process distributed port-Hamiltonian systems (GP-dPHS) enable data-driven modeling of distributed dynamical systems while preserving their physical structure. However, standard stationary kernels can be inefficient for spatially heterogeneous systems because they neither exploit known input...

---

### 21. [DeDiCa: A Scalable Deconfounding Discrete Causal Generative Model](https://arxiv.org/abs/2610.23272)

**Authors**: Kunchangtai Liang, Adri\'an Javaloy, Antonio Vergari, Isabel Valera  
**Category**: cs.LG  
**Published**: 2026-10-19  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2610.23272v1  

#### Abstract
We introduce DeDiCa, a practical framework for efficient causal inference with discrete variables and hidden confounding that revisits classical probabilistic modeling through a tensorized parameterization of causal Bayesian networks, enabling tractable inference. Given a causal graph, DeDiCa is tra...

---

### 22. [JAXtari: High-Throughput and Easy-to-Modify Arcade Learning Environment](https://arxiv.org/abs/2610.23650)

**Authors**: Quentin Delfosse, Raban Emunds, Paul Seitz, Sebastian Wette, Jannis Bl\"uml, Daniel Kirn, Dominik Mandok, Kristian Kersting  
**Category**: cs.LG  
**Published**: 2026-10-19  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2610.23650v1  

#### Abstract
The Arcade Learning Environment (ALE) has been a cornerstone of Deep Reinforcement Learning research, providing a standardized benchmark for evaluating general agents. However, its lack of native support for object-extraction and environment modification restricts research into evaluating whether ag...

---

### 23. [DiffusionRVGE: Reinforcement Learning via Regression-Based Value Gradient Estimation](https://arxiv.org/abs/2610.22973)

**Authors**: Jiayuan Wang, Tongyao Pang  
**Category**: cs.AI  
**Published**: 2026-10-19  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.22973v1  

#### Abstract
Reinforcement learning improves diffusion models using rewards on generated images, but repeated denoising and gradient estimation make training expensive. We introduce \method, an efficient method that constructs useful update directions by regressing centered rewards on centered final denoised ima...

---

### 24. [Generative Policy Improvement from Observational Data](https://arxiv.org/abs/2610.23436)

**Authors**: Jonas Schweisthal, Dennis Frauen, Stefan Feuerriegel, Emma Brunskill  
**Category**: cs.AI  
**Published**: 2026-10-19  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.23436v1  

#### Abstract
We study how to improve decision policies from observational data to generate better actions when actions are complex, structured objects, such as text messages or images. This task is subject to two major challenges: (i) action-level overlap is often too weak for reliable off-policy evaluation and ...

---

### 25. [MultiAgent-OPD: End-to-End Training of Hierarchical Multi-Agent Systems with On-Policy Distillation](https://arxiv.org/abs/2610.23234)

**Authors**: Yuhan Sun, Jiaxin Zhao, Yongdong Luo, Yibo Hu, Meiguang Jin, Junfeng Ma, Weixun Wang, Weihang Pan, Zhongan Bi  
**Category**: cs.CL  
**Published**: 2026-10-19  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.23234v1  

#### Abstract
In a hierarchical multi-agent system, a Leader assigns tasks and subagents carry them out. Jointly training the Leader and the subagents is difficult because a final answer gives only indirect feedback to the subagents. An LLM judge can score the performance of the subagents, but these scores depend...

---

### 26. [MailRail: Mitigating Incast Congestion with User-Space, Node-Based Flow Control](https://arxiv.org/abs/2610.22890)

**Authors**: Noel Gomillion, Lance Fletcher, Trevor Steil, Eun Jung Kim, Roger Pearce  
**Category**: cs.DC  
**Published**: 2026-10-19  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.22890v1  

#### Abstract
Irregular workloads, such as those stemming from Graph Neural Networks (GNNs) and Mixture-of-Experts (MoE) models, frequently induce severe incast congestion in large-scale parallel environments. Existing fabric-level congestion control mechanisms rely on reactive, link-level rate limiting that degr...

---

### 27. [Learning smooth dynamics via Kinematic Interpolants](https://arxiv.org/abs/2610.22519)

**Authors**: Yiheng (Roger),  Jiang, Youssef Marzouk, Avishek Joey Bose, Aram-Alexandre Pooladian  
**Category**: cs.LG  
**Published**: 2026-10-19  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.22519v1  

#### Abstract
Inferring dynamics from real-world experimental data is inherently ambiguous, as many plausible dynamical systems can explain the same positional observations at measurement times. For higher fidelity modeling of applications ranging from single-cell RNA to vortex dynamics and even flocking of birds...

---

### 28. [Normalizing Flow Maps](https://arxiv.org/abs/2610.22711)

**Authors**: Francesco M. Ruscio, Timothy James Hitge, Fred Zhangzhi Peng, Danyal Rehman, Zander W. Blasingame, Nicholas M. Boffi, Michael M. Bronstein, Alexander Tong, Avishek Joey Bose  
**Category**: cs.LG  
**Published**: 2026-10-19  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.22711v1  

#### Abstract
Scalable approaches to learning neural dynamical systems have placed diffusion, flow matching, and few-step generators in flow maps at the forefront of generative modeling. Such state-of-the-art models have limited access to explicit model densities despite evolving from classical generative modelin...

---

### 29. [SETAREH Transform: A Sorting-Assisted Haar-Wavelet Sparsification Method for Approximate Inner-Product Computation](https://arxiv.org/abs/2610.23338)

**Authors**: Seyed Kamaledin Setarehdan  
**Category**: cs.LG  
**Published**: 2026-10-19  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.23338v1  

#### Abstract
Modern deep learning systems increasingly rely on high-dimensional vector operations, in which large input and parameter vectors are repeatedly stored, transferred, and processed. Although substantial research has focused on reducing neural-network model size through pruning, quantization, knowledge...

---

### 30. [Beyond Large Language Models: A Mental World Model Framework for Mental Modeling, Simulation, Evolution, and Interpretability](https://arxiv.org/abs/2610.23722)

**Authors**: Qingshi Wang  
**Category**: cs.AI  
**Published**: 2026-10-19  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.23722v1  

#### Abstract
Understanding and simulating how subjective cognition and mental states emerge, evolve, and shape behavior is essential for advancing artificial intelligence toward deeper social cognition and general intelligence. Existing large language models (LLMs) and Theory of Mind (ToM) methods primarily infe...

---

## 🔧 Configuration

This bot is configured to look for papers containing the following keywords:
- framework, System, Generation, Linear, LLM, RL, RLHF, Reinforcement learning, Reinforcement Learning, Inference, Training, Attention, Pipeline, MOE, Sparse, Quantization, Speculative, Efficient, Efficiency, Framework, Parallel, Parallelism, Distributed, Kernel, Decode, Decoding, Prefill, Throughput, Fast, Network, Hardware, Cluster, FP8, FP4, Optimization, Scalable, Communication

## 📅 Schedule

The bot runs daily at 12:00 UTC via GitHub Actions to fetch the latest papers.

## 🚀 How to Use

1. **Fork this repository** to your GitHub account
2. **Customize the configuration** by editing `config.json`:
   - Add/remove arXiv categories (e.g., `cs.AI`, `cs.LG`, `cs.CL`)
   - Modify keywords to match your research interests
   - Adjust `max_papers` and `days_back` settings
3. **Enable GitHub Actions** in your repository settings
4. **The bot will automatically run daily** and update the README.md

## 📝 Customization

### arXiv Categories
Common categories include:
- `cs.AI` - Artificial Intelligence
- `cs.LG` - Machine Learning
- `cs.CL` - Computation and Language
- `cs.CV` - Computer Vision
- `cs.NE` - Neural and Evolutionary Computing
- `stat.ML` - Machine Learning (Statistics)

### Keywords
Add keywords that match your research interests. The bot will search for these terms in paper titles and abstracts.

### Exclude Keywords
Add terms to exclude certain types of papers (e.g., "survey", "review", "tutorial").

## 🔍 Manual Trigger

You can manually trigger the bot by:
1. Going to the "Actions" tab in your repository
2. Selecting "arXiv Bot Daily Update"
3. Clicking "Run workflow"

---
*Generated automatically by arXiv Bot* 
