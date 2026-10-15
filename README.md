# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-10-15 10:16:54 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Scaling LLM Inference via Parallelism: Throughput-Latency Trade-off](https://arxiv.org/abs/2610.19405)

**Authors**: Javad Mirzaei, Jeebak Mitra  
**Category**: cs.DC  
**Published**: 2026-10-15  
**Score**: 15.5  
**Type**: new  
**ArXiv ID**: 2610.19405v1  

#### Abstract
Large Language Model (LLM) serving systems must balance latency and throughput while efficiently utilizing distributed GPU resources. This challenge arises because inference consists of two fundamentally different phases: prefill, which determines time-to-first-token (TTFT), and decode, which determ...

---

### 2. [Echo: Error-Compensated 4-Bit Tensor-Parallel Communication for LLM Training and Inference](https://arxiv.org/abs/2610.14277)

**Authors**: Xingyu Xie, Xiaoyong Xie, Zhouchen Lin, Kim-Chuan Toh  
**Category**: cs.AI  
**Published**: 2026-10-15  
**Score**: 13.5  
**Type**: new  
**ArXiv ID**: 2610.14277v1  

#### Abstract
Tensor parallelism ends every layer with an all-reduce that sits between two dependent matrix multiplications, so its cost cannot be hidden and grows as compute outpaces interconnect bandwidth. Sending 4-bit instead of 16-bit values would cut this cost by nearly 4x, but the all-reduce must add the v...

---

### 3. [The Cost of Context: Profiling the Energy Footprint of Input Tokens in Large Language Models](https://arxiv.org/abs/2610.19819)

**Authors**: Boris Ruf, Marcin Detyniecki  
**Category**: cs.DC  
**Published**: 2026-10-15  
**Score**: 13.0  
**Type**: new  
**ArXiv ID**: 2610.19819v1  

#### Abstract
The rapid adoption of Retrieval-Augmented Generation (RAG) and long-context LLMs has shifted the computational burden toward the prefill phase---the initial stage where the model processes all input tokens in parallel before generating a response---yet its energy impact remains poorly characterised....

---

### 4. [Stable Token-level Scheduling for Serverless LLM Inference](https://arxiv.org/abs/2610.17689)

**Authors**: Hardi Ashishbhai Desai, Balaji N K, Sourav Kanti Addya  
**Category**: cs.AI  
**Published**: 2026-10-15  
**Score**: 12.5  
**Type**: new  
**ArXiv ID**: 2610.17689v1  

#### Abstract
Serverless inference is emerging as an efficient deployment model for Large Language Models (LLMs) due to its ability to elastically scale resources, improve GPU utilization through resource sharing, and reduce infrastructure costs. However, existing approaches primarily optimize time to first token...

---

### 5. [FleetPilot: Fine-Grained Steering Empowers Real-World LLM Personalization](https://arxiv.org/abs/2610.14970)

**Authors**: Kounianhua Du, Jianxing Liu, Kangning Zhang, Wenxiang Jiao, Yuan Lu, Jiarui Jin, Jianghao Lin, Weiwen Liu, Yong Yu, Weinan Zhang  
**Category**: cs.AI  
**Published**: 2026-10-15  
**Score**: 12.0  
**Type**: new  
**ArXiv ID**: 2610.14970v1  

#### Abstract
With the rapid evolution of large language models (LLMs), there is an increasing demand for effective personalization techniques that can adapt model behavior to individual user preferences. Personalized LLMs must infer a user's query-dependent preferences from limited, heterogeneous, and evolving i...

---

### 6. [Joint Subspace Sparse Attention for Efficient Long-Context Decoding](https://arxiv.org/abs/2610.19026)

**Authors**: Jiwon Song, Yoongon Kim, Jae-Joon Kim  
**Category**: cs.CL  
**Published**: 2026-10-15  
**Score**: 12.0  
**Type**: new  
**ArXiv ID**: 2610.19026v1  

#### Abstract
Sparse attention reduces the KV-cache traffic that dominates long-context LLM decoding, but practical methods largely operate at coarse granularity. Token-level selection offers finer control, yet presents two challenges: low-dimensional scoring may lose query--key structure needed for accurate rank...

---

### 7. [TreeSpark: Learning to Draft the Trees You Verify for Speculative Decoding](https://arxiv.org/abs/2610.20175)

**Authors**: Jihao Zhao, Qianyu Jiang, Shuaishuai Zu, Pengfei Li, Guanghuan Fang, Ben Feng, Biao Qin, Zhibin Wang, Baochuan Yang  
**Category**: cs.CL  
**Published**: 2026-10-15  
**Score**: 12.0  
**Type**: new  
**ArXiv ID**: 2610.20175v1  

#### Abstract
Parallel drafters such as DFlash and DSpark reduce proposal latency, but early mismatches limit how much of each block can be accepted. Retaining alternatives introduces parent states and candidate choices that reference-conditioned training does not directly supervise. We introduce TreeSpark, a uni...

---

### 8. [Chain Dimension Scaling: Communication-Efficient Loop-Style Scaling for Industrial Short-Video Ranking](https://arxiv.org/abs/2610.16415)

**Authors**: Zhimeng Jiang, Yuening Li, Ruixiao Sun, Yuanzhen Lin, Kun Jin, Chuan He, Liang Liu  
**Category**: cs.AI  
**Published**: 2026-10-15  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2610.16415v1  

#### Abstract
Industrial recommendation rankers keep scaling in width and depth to billions of parameters, but this increasingly incurs costs in parameter growth, distributed communication overhead, and training inefficiency. We introduce \underline{C}hain \underline{D}imension \underline{S}caling (CDS), a scalin...

---

### 9. [Queue-Aware Token Communications: Bridging AI Representation and Wireless Service](https://arxiv.org/abs/2610.16730)

**Authors**: Yuquan Xiao, Qinghe Du, Chaoyang Zhang, Wenchi Cheng, Jingqing Wang, Tao Chen, Jiangzhou Wang  
**Category**: cs.AI  
**Published**: 2026-10-15  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2610.16730v1  

#### Abstract
The rapid advancement of foundation models and edge intelligence is transforming wireless networks from data-oriented communication systems into artificial intelligence (AI)-native service platforms. Token communications (TokenCom) has emerged as a promising paradigm that directly transmits task-ori...

---

### 10. [Titan-Llama: Retrofitting Pre-trained LLMs with External Neural Memory Modules](https://arxiv.org/abs/2610.19210)

**Authors**: Sarah Pan, Ali Behrouz  
**Category**: cs.CL  
**Published**: 2026-10-15  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2610.19210v1  

#### Abstract
Transformers are efficient to pre-train at scale because attention permits parallel computation across sequence positions, but long-context inference requires increasingly large KV caches. Efficient recurrent alternatives reduce the asymptotic complexity of inference, but their sequential dependence...

---

### 11. [ForkGRPO: Efficient Exploration and Fine-Grained Credit Assignment for Diffusion Language Models](https://arxiv.org/abs/2610.19408)

**Authors**: Zhaoyan Jia, Chengwei Zhou, Gourav Datta, Massoud Pedram  
**Category**: cs.CL  
**Published**: 2026-10-15  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2610.19408v1  

#### Abstract
Diffusion large language models (dLLMs) have advanced rapidly, yet strengthening their reasoning through reinforcement learning (RL) remains an open challenge. Any-order generation can suffer from entropy degradation, which narrows the reasoning paths available for exploration. Recent work mitigates...

---

### 12. [SAT-RL: Training Language Models for Logical Reasoning with Symbolic Solvers](https://arxiv.org/abs/2610.14402)

**Authors**: Jialiang Sun, Kuldeep Meel  
**Category**: cs.AI  
**Published**: 2026-10-15  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2610.14402v1  

#### Abstract
Large language models (LLMs) have achieved strong performance in mathematical and logical reasoning, yet they still suffer from the curse of complexity and struggle as problem difficulty increases with the number of constraint interactions and the size of the search space. In contrast, satisfiabilit...

---

### 13. [Hardware-Friendly Token-Group Activation Quantization for Low-Bit Mamba Super-Resolution](https://arxiv.org/abs/2610.16261)

**Authors**: Jinwoo Chung, Jangho Kim  
**Category**: cs.AI  
**Published**: 2026-10-15  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2610.16261v1  

#### Abstract
Mamba-based super-resolution models are efficient alternatives to Transformer-based restoration backbones, but their low-bit post-training quantization remains challenging because selective scan, Mamba's input-dependent state-space operator, induces different activation ranges across spatial tokens....

---

### 14. [Inverse Reasoning Models for Efficient Post-Training and Verification from Answer-Only Data](https://arxiv.org/abs/2610.18878)

**Authors**: Eric O. Korman  
**Category**: cs.LG  
**Published**: 2026-10-15  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2610.18878v1  

#### Abstract
Post-training reasoning LLMs often requires making a tradeoff between costly supervision and compute inefficiency: supervised finetuning typically requires reasoning traces, which is data inefficient since in many applications one only has prompt-answer tuples and lacks the groundtruth chain-of-thou...

---

### 15. [EvoQuantLab: Training-Free Dual-Loop Evolution for Auditable Quantitative Research](https://arxiv.org/abs/2610.17130)

**Authors**: Haolin Yang, Zhiqiang Qian, Chutian Chen, Jipeng Zhang, Zhenghao Zhu, Tangjia Deng, Jian Xie, Shuaishuai Gong, Yike Guo, Sirui Han  
**Category**: cs.AI  
**Published**: 2026-10-15  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2610.17130v2  

#### Abstract
Automated alpha factor discovery is critical for modern quantitative research. While large language models (LLMs) have increasingly been integrated into quantitative workflows, existing agent-based approaches remain highly susceptible to severe backtest overfitting. To address this challenge, we pro...

---

### 16. [Qwen-MT-Lumen at WMT2026: Instruction-Specialized Translation via Quality-Aware Training and On-Policy Distillation](https://arxiv.org/abs/2610.19758)

**Authors**: Hao Wang, Yangyang Liu, Xiaohu Zhao, Heng Liu, Zifu Shang, Tianhao Li, Ruize Gao, Jialong Tang, Liangying Shao, Haoran Wei, Baosong Yang, Linlong Xu, Longyue Wang, Weihua Luo  
**Category**: cs.CL  
**Published**: 2026-10-15  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2610.19758v1  

#### Abstract
We present Lumen, The open-source model is available at, our submission to the WMT 2026 General Machine Translation Shared Task. We focus on instruction-specialized translation, where systems must produce accurate translations while satisfying fine-grained requirements on style, terminology, and str...

---

### 17. [ReaDy: Recurrent States as Dynamic Context for Speculative Drafting in Hybrid Linear Attention Large Language Models](https://arxiv.org/abs/2610.20503)

**Authors**: Bohan Sun, Zunhai Su, Xialie Zhuang, Jianjin Li, Ngai Wong, Chuan-Wei Kuo  
**Category**: cs.CL  
**Published**: 2026-10-15  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2610.20503v1  

#### Abstract
Speculative decoding accelerates autoregressive generation by drafting multiple tokens per step, yet feature-based drafters rely on target hidden features whose contextual emphasis may not align with the needs of subsequent draft positions. Hybrid linear attention (HLA) models maintain compact recur...

---

### 18. [PairIT: Autoregressive Transformers for Low-Data Molecule Optimization](https://arxiv.org/abs/2610.19123)

**Authors**: Hannah Lawrence, Bodhi P. Vani, Ji Won Park, Natasa Tagasovska, Samuel Don Stanton, Andrew Martin Watkins, Michael Maser, Ewa M. Nowara  
**Category**: cs.LG  
**Published**: 2026-10-15  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2610.19123v1  

#### Abstract
Generating novel molecules with desired properties is a central challenge in drug discovery. Although foundational work focused on de novo generation, practical use cases require a more complex setting: conditioning on an initial seed molecule, optimizing multiple properties, and operating in data-s...

---

### 19. [Towards Training and Inference of Graph Neural Networks in Graph Databases](https://arxiv.org/abs/2610.19142)

**Authors**: Victor Pekkari, Harry Wik, Alfred Clemedtson, Borun Shi, Xuan-Son Vu  
**Category**: cs.LG  
**Published**: 2026-10-15  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2610.19142v1  

#### Abstract
Graph Neural Networks on large graphs commonly rely on in-memory graph storage with ad hoc data processing pipelines for both training and inference. Graph databases provide scalable storage and efficient traversal for graphs that can evolve over time, but their applicability in GNN training and inf...

---

### 20. [Optimizing Sparse Boolean Rule Networks using Mixed Integer Linear Programming](https://arxiv.org/abs/2610.20170)

**Authors**: Thomas Pammer, Johannes F\"urnkranz, Markus Sinnl  
**Category**: cs.LG  
**Published**: 2026-10-15  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2610.20170v1  

#### Abstract
We study the problem of learning accurate and cognitively simple binary classifiers that are directly interpretable as Boolean rules. We propose mixed integer linear programming (MILP) formulations for training sparse disjunctive normal form (DNF) networks, viewed as small networks with a conjunctiv...

---

### 21. [Bridging Risk Approximation Gaps in Model Predictive Task Sampling via In-Context Modeling](https://arxiv.org/abs/2610.14913)

**Authors**: Jiarong Wen, Qi Tao, Kaiyu Zhang, Yun Qu, Lizhou Cai, Heming Zou, Yuhang Jiang, Qi Wang  
**Category**: cs.AI  
**Published**: 2026-10-15  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2610.14913v1  

#### Abstract
Reinforcement learning (RL) is essential for adaptive decision-making, e.g., robust Meta-RL and domain randomization (DR), and efficient post-training of large language models (LLMs), where tasks are commonly structured as Markov decision processes or prompts. A central bottleneck in all these setti...

---

### 22. [FARL: A Fully Automated Pipeline of RL Interface and Solver Design](https://arxiv.org/abs/2610.15691)

**Authors**: Tianyu Jin, Liuyu Xiang, Chaowen Yang, Xingyang Chang, Pei Pei Li, Zhaofeng He  
**Category**: cs.AI  
**Published**: 2026-10-15  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2610.15691v1  

#### Abstract
Existing reinforcement learning (RL) largely focuses on optimizing learning algorithms within a pre-defined task formulation, while applying RL to new tasks still relies heavily on human expertise to design the learning pipeline: what an agent observes, how it is rewarded, which algorithm it uses, a...

---

### 23. [Retrieve What Helps, Only When It Helps: Agentic Reinforcement Learning for Selective Retrieval-Augmented Machine Translation](https://arxiv.org/abs/2610.19528)

**Authors**: Shijun Li, Yang Liu, Najmeh Sadoughi, Dongmei Wang, Xinjie Li, Zhu Liu, Rohith Mysore Vijaya Kumar, Yuzong Liu  
**Category**: cs.CL  
**Published**: 2026-10-15  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2610.19528v1  

#### Abstract
Retrieving external knowledge can improve large language model (LLM) translation by providing supplementary information such as idiom interpretations and terminology. However, existing retrieval-augmented machine translation systems typically apply predefined retrieval strategies uniformly across al...

---

### 24. [MemDuet: Learning Dual-Process Memory for Long-Context Reasoning](https://arxiv.org/abs/2610.19928)

**Authors**: Jiangcheng Song, Haodong Jing, Yulin Mu, Lesong Tao, Jixin Wang, Jin Cui, Tianjing Ren, Runxuan Xie, Yongqiang Ma, Nanning Zheng  
**Category**: cs.CL  
**Published**: 2026-10-15  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2610.19928v1  

#### Abstract
Reasoning over infinitely long documents is becoming a reality with recurrent memory agents. However, increasing context length exacerbates the trade-off between reasoning performance and inference efficiency. Inspired by Kahneman's account of fast and slow thinking, we propose MemDuet, a dual-proce...

---

### 25. [FOLD: Position-Free Far Memory with Local Order for Long-Context Speculative Decoding](https://arxiv.org/abs/2610.20392)

**Authors**: Shuxuan Li, Jialiang Kang, Yanghao Zhou, Ziye Zhu, Xiao Yang  
**Category**: cs.CL  
**Published**: 2026-10-15  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2610.20392v1  

#### Abstract
Speculative decoding accelerates LLM generation through lightweight drafting and parallel verification. It is particularly promising for long inputs, where KV-cache access costs can be amortized over multiple accepted tokens. Yet most drafters are trained on short sequences and degrade on long input...

---

### 26. [Backpropagating Verifier Rewards Through Continuous Flow Language Models](https://arxiv.org/abs/2610.19080)

**Authors**: Xu Han, Jiajing Hu, Li-Ping Liu  
**Category**: cs.LG  
**Published**: 2026-10-15  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2610.19080v1  

#### Abstract
Reinforcement learning (RL) has substantially improved the reasoning of autoregressive and discrete-diffusion language models. Language models that generate continuous token embeddings are emerging as a strong alternative, but RL post-training for them remains understudied. We propose \method{}, a p...

---

### 27. [Reinforcement Learning via Advantage Guidance](https://arxiv.org/abs/2610.19783)

**Authors**: Zhiyuan Zhou, Sergey Levine  
**Category**: cs.LG  
**Published**: 2026-10-15  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2610.19783v1  

#### Abstract
Reinforcement learning (RL) holds the promise of learning a policy that can outperform what is shown in the data. However, in practice, such systems suffer from instability and hyperparameter sensitivity. In contrast, supervised learning provides a stable and scalable training objective, but is fund...

---

### 28. [Comparative Evaluation of Deep-Learning based DoA Estimators for Real-Time ISAC Beamforming](https://arxiv.org/abs/2610.20326)

**Authors**: Anjali K. S., Manoj Kumar Panda, Chandramani Singh  
**Category**: cs.LG  
**Published**: 2026-10-15  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2610.20326v1  

#### Abstract
Integrated Sensing and Communication (ISAC) systems rely on accurate Direction-of-Arrival (DoA) estimation for effective beamforming and spectral-efficiency enhancement. While numerous classical and deep-learning-based DoA estimation methods have been reported in the literature, their suitability fo...

---

### 29. [When Does Momentum Help in Reinforcement Learning with Verifiable Rewards?](https://arxiv.org/abs/2610.20413)

**Authors**: Kaifei Wang, Zikang Shan, Liwei Wang, Yinyu Ye, Han Zhong  
**Category**: cs.LG  
**Published**: 2026-10-15  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2610.20413v1  

#### Abstract
Momentum is a classical tool for accelerating convex optimization and a standard component of modern large language model training. Yet recent empirical work has questioned the usefulness of first-moment momentum in reinforcement learning with verifiable rewards (RLVR). We study this question throug...

---

### 30. [An FPGA-Based Spiking Neural Network for Spatiotemporal Pattern Learning and Classification](https://arxiv.org/abs/2610.17463)

**Authors**: Jingang Jin, Yue Ma, Zhenhang Zhang, Qinru Qiu  
**Category**: cs.AI  
**Published**: 2026-10-15  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2610.17463v1  

#### Abstract
Spiking Neural Networks (SNNs) are promising for energy-efficient spatiotemporal pattern recognition at the edge, but efficient deployment and on-device adaptation require specialized hardware. In this work, we present an FPGA implementation of multilayer SNNs for spatiotemporal pattern classificati...

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
