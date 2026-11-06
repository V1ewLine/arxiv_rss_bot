# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-11-06 13:12:18 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [FlashOffloading: Low-Latency KV Cache Offloading for Long-Context LLM Serving](https://arxiv.org/abs/2611.07542)

**Authors**: Xuan Wang, Zhuanhao Wu, Tingzhu Fang, Xushi Li, Sicheng Zhang, Zhenyu Yang, Feng Liu, Zhaohui Fan, Qiang Xie, Huayi Zou, Yuncong Liu, Zeyu Song, Ke Zhu, Yanjia Zeng, Jingchun Gao, Yan Shi, Kui Luo, Gezi Li, Liangjun Feng, Cheng Li  
**Category**: cs.AI  
**Published**: 2026-11-06  
**Score**: 15.5  
**Type**: new  
**ArXiv ID**: 2611.07542v1  

#### Abstract
In large language model (LLM) serving systems, the key-value (KV) cache grows linearly with sequence length and concurrency, so long-context inference easily hits the high-bandwidth memory (HBM) capacity wall and results in low concurrency. Offloading the KV cache to host dynamic random-access memor...

---

### 2. [Resource-Efficient Reinforcement Learning Post-Training through Elastic Expert Parallelism](https://arxiv.org/abs/2611.06984)

**Authors**: Jia Wei, Ziyi Qiu, Yi Xu, Xuehai Qian  
**Category**: cs.AI  
**Published**: 2026-11-06  
**Score**: 15.0  
**Type**: new  
**ArXiv ID**: 2611.06984v1  

#### Abstract
Reinforcement learning post-training improves the capabilities of large language models, but generating training responses is a major computational bottleneck. In mixture-of-experts models, workers that finish their assigned responses may remain occupied because their experts are still needed by wor...

---

### 3. [FastAFD: When Does Attention-FFN Disaggregation Pay on Rack-Scale NVLink?](https://arxiv.org/abs/2611.07149)

**Authors**: Yichao Fu, Yuxuan Zhang, Minshen Zhang, Ruitian Wang, Junda Chen, Hao Zhang  
**Category**: cs.DC  
**Published**: 2026-11-06  
**Score**: 14.0  
**Type**: new  
**ArXiv ID**: 2611.07149v1  

#### Abstract
Attention-FFN disaggregation (AFD) separates attention from the experts of a mixture-of-experts (MoE) model and has been reported to raise decode throughput over colocated serving. Whether this advantage holds on rack-scale NVLink systems such as GB200 NVL72 remains unclear: prior results were obtai...

---

### 4. [Stochastic Dynamics of Generative AI: From Autoregressive Models to Interacting LLM Networks and Neural SDEs](https://arxiv.org/abs/2611.07065)

**Authors**: Vikram Krishnamurthy, Yiming Zhang, Luke Snow  
**Category**: cs.LG  
**Published**: 2026-11-06  
**Score**: 12.0  
**Type**: new  
**ArXiv ID**: 2611.07065v1  

#### Abstract
This paper develops a stochastic dynamical-systems perspective on three classes of generative AI models. First, we formulate large language models (LLMs) as high-dimensional nonlinear autoregressive models with attention-based dependencies. This formulation connects pretraining by conditional maximu...

---

### 5. [On the Opportunities and Challenges of Prefill-to-Autoregressive Cross-Model Transfer](https://arxiv.org/abs/2611.06976)

**Authors**: Zhenyi Shen, Junru Lu, Wei Liu, Jiangnan Ye, Jiazheng Li, Di Yin, Xing Sun, Tameem Adel, Yali Du, Lin Gui, Yulan He  
**Category**: cs.CL  
**Published**: 2026-11-06  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2611.06976v1  

#### Abstract
Prefill-to-Autoregressive Cross-model Transfer (PACT), a form of latent communication between language models, combines a larger model's prompt representations with a smaller model's efficient generation, offering a promising way to balance capability and inference cost. However, existing systems pr...

---

### 6. [DMuon: Scaling Matrix Optimizers in Data Parallelism with Topology-Aware Matrix Ownership and Communication](https://arxiv.org/abs/2611.07084)

**Authors**: Starrick Liu, Ruizhi Chengze, Lufang Chen, Michael Zhao, Ran Yu, Xiaofan Li, dance kuzi, Yue Zhang, Zijuan Lin, SongAo Han, YongHua Chen, Ruyi Gan, Hao Wang  
**Category**: cs.DC  
**Published**: 2026-11-06  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2611.07084v1  

#### Abstract
Matrix optimizers such as Muon update whole weight matrices, which conflicts with the element- or row-wise sharding of data-parallel runtimes. Once parameters are sharded as well, matrix optimizers incur extra communication or repeated computation. We present DMuon, a matrix-granular optimizer runti...

---

### 7. [SEER: A Scalable Event-Driven Runtime for Parallel Execution of Multi-Agent Reinforcement Learning Environments](https://arxiv.org/abs/2611.07996)

**Authors**: Jiming Su, Feng Zhu, Yuhao Xiao, Miao Zhang  
**Category**: cs.DC  
**Published**: 2026-11-06  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2611.07996v1  

#### Abstract
Multi-agent reinforcement learning (MARL) is increasingly used in systems where many entities interact in a shared environment. In these workloads, policy inference can be batched on GPUs, but the CPU-side environment path still updates entity states, resolves local interactions, and builds observat...

---

### 8. [Muir: Linear Attention with Sensitivity-Allocated Gating and Matched-Filter Readout](https://arxiv.org/abs/2611.08376)

**Authors**: Tushaar Gangavarapu, Gil Tabak, Clemens JS Schaefer  
**Category**: cs.LG  
**Published**: 2026-11-06  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2611.08376v1  

#### Abstract
Linear attention compresses context into a fixed-size key-value state for linear-time training and constant-memory inference. Gated delta-rule models decay this state and erase its prediction for an incoming key before writing a new value; Gated DeltaNet-2 (GDN-2) generalizes this with channel-wise ...

---

### 9. [ACSeL: Adaptive Configuration Search for Hardware-Constrained LLM Inference](https://arxiv.org/abs/2611.07570)

**Authors**: Bokun Wang, Hung-Yueh Chiang, Diana Marculescu  
**Category**: cs.LG  
**Published**: 2026-11-06  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2611.07570v1  

#### Abstract
Serving large language models (LLMs) under tight memory and latency service-level objectives (SLOs), including on edge devices, requires jointly selecting the model size, batch size, quantization algorithm, and mixed-precision quantization recipes. Exhaustive search over this combinatorially large s...

---

### 10. [The Blindfolded Controller: Decision Model-Guided Training for Reinforcement Learning](https://arxiv.org/abs/2611.07934)

**Authors**: Muhammad Ibrahim Khan, Aniket Dixit, Faizan Ahmed, James Brusey  
**Category**: cs.LG  
**Published**: 2026-11-06  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2611.07934v1  

#### Abstract
Reinforcement learning (RL) provides a framework for continuous improvement through interaction with an environment. However, RL agents struggle against tasks that offer exploration challenges and sparse rewards, because learning efficient policies from scratch is slow and sample-inefficient. In con...

---

### 11. [SPARE: Fast Linear Attention Decoding with Compact Recurrent States](https://arxiv.org/abs/2611.07552)

**Authors**: Ke Wan, Zaiyi Zheng, Chen Chen  
**Category**: cs.CL  
**Published**: 2026-11-06  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2611.07552v1  

#### Abstract
Linear attention reduces the quadratic sequence-length complexity of softmax attention by maintaining fixed-size recurrent states. However, these states scale quadratically with the representation dimension and require repeated data transfers during decoding, creating a substantial GPU memory bottle...

---

### 12. [Do Recommenders Need to Think in Words? Latent Reasoning for Efficient LLM-Based Recommendation](https://arxiv.org/abs/2611.07866)

**Authors**: Hongyang Su, Beibei Kong, Lei Cheng, Chengxiang Zhuo, Zang Li  
**Category**: cs.CL  
**Published**: 2026-11-06  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2611.07866v1  

#### Abstract
LLM-based recommender systems increasingly generate chain-of-thought (CoT) text before each item prediction to activate the LLM's reasoning capacity, costing tens to hundreds of extra tokens per recommendation at inference. We ask whether inference-time reasoning artifacts (textual or latent) can be...

---

### 13. [Not All Loops Cost the Same: Phase-Dependent Depth for Serving Looped Transformers](https://arxiv.org/abs/2611.07211)

**Authors**: Zhixu Du, Yuhan Cheng, Hai "Helen" Li, Yiran Chen  
**Category**: cs.DC  
**Published**: 2026-11-06  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2611.07211v1  

#### Abstract
Looped transformers scale inference-time reasoning by repeatedly reusing a shared core, increasing computation without increasing parameter count. Serving prefilling and decoding with a uniform loop budget couples their costs. We present LoopDuet, a phase-dependent depth and training-free serving ap...

---

### 14. [ZYFA: Speed and Accuracy of FP8 FlashAttention on AMD CDNA4](https://arxiv.org/abs/2611.07496)

**Authors**: Quentin Anthony, Ganesh Bikshandi, Praneeth Medepalli, Stephen Ebert  
**Category**: cs.DC  
**Published**: 2026-11-06  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2611.07496v1  

#### Abstract
Attention is a core building block of frontier models and often dominates compute-bound prefill at long context. FP8 matrix instructions offer twice the peak throughput of BF16, but they accelerate only matrix multiplication, and the softmax between the two matrix multiplications of attention runs o...

---

### 15. [Enhancing Repository-Level Code Completion with Reinforcement Learning in Real-World Scenarios](https://arxiv.org/abs/2611.07510)

**Authors**: Zhenlong Dai, Zhuoluo Zhao, Xiu Tang, Sai Wu, Chang Yao, Zhipeng Gao, Jingyuan Chen  
**Category**: cs.CL  
**Published**: 2026-11-06  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2611.07510v1  

#### Abstract
The rapid development of large language models (LLMs) for code, which closely aligns with the practical needs of real-world software development, has sparked growing attention in repository-level code completion. However, existing Code LLMs present challenges in focusing on suitable contexts from th...

---

### 16. [FewStepRL: Fine-tuning Few-step Diffusion Models with Reinforcement Learning](https://arxiv.org/abs/2611.08240)

**Authors**: Haitao Lin, Minsi Ren, Yang Li, Xingtong Ge, Zheyuan Hu, Peiyan Hu, Tailin Wu  
**Category**: cs.LG  
**Published**: 2026-11-06  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2611.08240v1  

#### Abstract
Reinforcement learning (RL) has become a standard post-training method for diffusion models, but most RL objectives roll out the current policy at every iteration, and with a many-step sampler these rollouts typically dominate the cost of training. Few-step models cut this cost, yet they model an av...

---

### 17. [BlockLock: Reproducible LLM Inference with Fixed Batch Geometry](https://arxiv.org/abs/2611.06977)

**Authors**: Xiuyuan Qi, Pedro Reviriego, Javier Conde, Shanshan Liu  
**Category**: cs.AI  
**Published**: 2026-11-06  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2611.06977v1  

#### Abstract
Reproducible LLM inference requires that a request produces the same output regardless of the other requests in the batch. Existing solutions pay for this in different ways: batch-invariant kernels constrain every operator, and isolating prefill with fixed-shape decode ({our CoRun-style baseline}) g...

---

### 18. [Recursive Sparse Attention: Million-Token Length Extrapolation at Logarithmic Cost](https://arxiv.org/abs/2611.07232)

**Authors**: Ruitian Wang, Xinyu Yang, Minshen Zhang, Hao Zhang  
**Category**: cs.CL  
**Published**: 2026-11-06  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2611.07232v1  

#### Abstract
At context lengths of about 1M tokens, dense attention is costly in prefill and decoding, and training typically relies on long documents, which become scarcer as the target length increases. Most sparse attention methods score all preceding blocks or tokens, so their selection cost per query is $O(...

---

### 19. [Automated Video-to-QA Pipeline: End-to-End Question Generation and Verification Using Large Language Models](https://arxiv.org/abs/2611.07755)

**Authors**: Jaival Chauhan, Vikram Singh, Pruthwik Mishra  
**Category**: cs.CL  
**Published**: 2026-11-06  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2611.07755v1  

#### Abstract
The exponential growth of educational and informational video content on platforms like YouTube presents a challenge for efficient knowledge retrieval and assessment. This paper proposes an end-to-end automated pipeline for extracting structured summaries and generating high-fidelity Question-Answer...

---

### 20. [Agent-R$^2$: Scaling Agents for Realistic Productivity Work via Real-Source Data Synthesis](https://arxiv.org/abs/2611.08219)

**Authors**: Teng Pan, Zixuan Wang, Yuchen Yan, Qifei Wu, Yao Chen, Yingqi Qu, Ruiqing Zhang, Yongliang Shen  
**Category**: cs.CL  
**Published**: 2026-11-06  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2611.08219v1  

#### Abstract
Large language models have made rapid progress in agentic reasoning and tool use, raising expectations for agents that can take over realistic productivity work from humans. However, current agents still struggle to finish such tasks, where success requires following long-horizon professional workfl...

---

### 21. [NUMA-Torch: Compiler-Driven NUMA-Aware LLM Prefill on Multi-Socket CPUs](https://arxiv.org/abs/2611.07105)

**Authors**: Thisara Rathnayaka Weerakoon (Department of Computer Science & Engineering, University of Moratuwa, Sri Lanka), Ishan Dewpura (Department of Computer Science & Engineering, University of Moratuwa, Sri Lanka), Niman Ransindu (Department of Computer Science & Engineering, University of Moratuwa, Sri Lanka), Shavin Anjitha (Department of Computer Science & Engineering, University of Moratuwa, Sri Lanka), Tharindu R. Patabandi (Independent Researcher, San Jose, CA, USA), Sanath Jayasena (Department of Computer Science & Engineering, University of Moratuwa, Sri Lanka)  
**Category**: cs.DC  
**Published**: 2026-11-06  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2611.07105v1  

#### Abstract
For sufficiently large transformer-based Large Language Models (LLMs), whose parameters cannot be retained in on-chip caches, prefill throughput on multi-socket CPUs is often limited by sustainable DRAM bandwidth rather than peak floating-point throughput. In the standard PyTorch compilation (i.e., ...

---

### 22. [DictLLM: Post-training Compression of Large Language Model with Dictionary Kernels](https://arxiv.org/abs/2611.08082)

**Authors**: Jinho Park, Se Young Chun, Mingoo Seok  
**Category**: cs.AI  
**Published**: 2026-11-06  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2611.08082v1  

#### Abstract
Efficient deployment of large language models (LLMs) on mobile devices requires balancing performance with compute and memory constraints. Post-training compression is an effective approach to managing this trade-off, while requiring much less computation than in-training counterparts. We propose Di...

---

### 23. [SwarmEvolve: Latent Agent Swarms for Evolutionary Discovery and Optimization](https://arxiv.org/abs/2611.08483)

**Authors**: Qile Jiang, George Em Karniadakis  
**Category**: cs.AI  
**Published**: 2026-11-06  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2611.08483v1  

#### Abstract
LLM-guided evolutionary search uses AI agents to improve programs for mathematical and computational problems by iteratively refining solutions and combining high-scoring methods. However, the process is often slow because the LLM generates a long textual chain of thought for each round and must rep...

---

### 24. [Scalable Bayesian Gaussian Process Classification for Cancer Histopathology: Calibrated Uncertainty Under Distribution Shift](https://arxiv.org/abs/2611.07127)

**Authors**: Yunru Zheng, Wun Ting Chan, Sampras Dsouza, Zhenyu Hong  
**Category**: cs.LG  
**Published**: 2026-11-06  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2611.07127v1  

#### Abstract
Reliable uncertainty quantification is essential for clinical deployment of cancer diagnosis models, yet neural networks and variational Gaussian process methods can produce overconfident predictions, particularly under the distribution shifts that arise when models trained at one hospital are appli...

---

### 25. [How Do AI Agents Waste Your Money?](https://arxiv.org/abs/2611.07238)

**Authors**: Huimin Zeng, Yuxuan Lu, Jesse Cai, Eric Hanchen Jiang, Di Wu, Po-Nien Kung, Wenbo Hu, Ziyi Wang, Dakuo Wang, Jiaxin Pei, Kai-Wei Chang  
**Category**: cs.CL  
**Published**: 2026-11-06  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2611.07238v1  

#### Abstract
AI agents may consume massive tokens across a wide range of tasks, but how many of those tokens are really necessary? Although prior work has shown that agents can incur unexpectedly high token costs, we still lack a systematic way to identify and measure token waste. In this work, we introduce a un...

---

### 26. [Energy Measurement and Optimization of TinyML in Resource-Constrained Embedded Systems: A Systematic Mapping Study](https://arxiv.org/abs/2611.07264)

**Authors**: Clariele Almeida, Renan Beserra, Rafael Moura, Danilo Ara\'ujo, Bruno Nogueira, S\'ergio Chevtchenko, Ermeson Andrade  
**Category**: cs.LG  
**Published**: 2026-11-06  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2611.07264v1  

#### Abstract
The rapid expansion of the Internet of Things (IoT) has increased the volume of data generated at the network edge, where many devices operate under severe memory, processing, and energy constraints. In this context, Tiny Machine Learning (TinyML) enables machine learning (ML) tasks to be executed o...

---

### 27. [PromptOps: Iterative Prompt Optimization for LLM-as-a-Judge Systems](https://arxiv.org/abs/2611.07123)

**Authors**: Sven Kleinkop, Tam\'as Janusko, Ricardo Bochnia, Gunnar Hempel, Steffen Tomschke, J\"urgen Anke, Maik Thiele  
**Category**: cs.AI  
**Published**: 2026-11-06  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2611.07123v1  

#### Abstract
Large language models (LLMs) are increasingly used as automated judges for evaluating the behavior of chatbots and other AI systems. In deployment, however, their agreement with the intended human evaluation policy may degrade as user interactions and score distributions change. Existing automatic p...

---

### 28. [Who Should Know What? Empirical Information Design for Multi-Agent LLM Economies](https://arxiv.org/abs/2611.07667)

**Authors**: Mete Erdogan, Mirac Suzgun, Aneesh Pappu, Guowei Xu, Batu El, Andrew A. Schoen, Furong Huang, John P. Dickerson, Mert Pilanci, James Zou  
**Category**: cs.AI  
**Published**: 2026-11-06  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2611.07667v1  

#### Abstract
The performance of multi-agent LLM systems depends not only on the capabilities of the underlying models but also on information-sharing policies, which govern what information each agent observes and at what level of detail. But does providing AI agents with more detailed information always improve...

---

### 29. [LLM-Assisted Label Noise Correction for EEG-Based Sleep Stage Classification](https://arxiv.org/abs/2611.08311)

**Authors**: Elmira Salari, Shruti Kshirsagar  
**Category**: cs.CL  
**Published**: 2026-11-06  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2611.08311v1  

#### Abstract
Automatic sleep stage classification from electroencephalography (EEG) signals is a critical task in clinical sleep medicine. While deep learning models have shown promising results in this domain, their performance is fundamentally constrained by the quality of training labels, which human experts ...

---

### 30. [SpecCTMC: Exact Speculative Simulation of Diffusion Language Model Continuous-Time Markov Chains](https://arxiv.org/abs/2611.08362)

**Authors**: Siva Rajesh Kasa, Yasong Dai, Sumit Negi, Hongdong Li  
**Category**: cs.CL  
**Published**: 2026-11-06  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2611.08362v1  

#### Abstract
Masked diffusion language models offer parallel text generation, but independently sampling multiple tokens can disrupt their dependencies and change the output distribution. Exact continuous-time sampling avoids this approximation, yet typically requires one sequential model evaluation per token. W...

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
