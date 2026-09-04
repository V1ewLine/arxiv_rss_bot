# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-09-04 09:55:26 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [DE-Venus: A Data-Efficient RLVR Framework for Large Language Models](https://arxiv.org/abs/2609.03324)

**Authors**: Shenzhi Yang, Guangcheng Zhu, Kai Tang, Zhengqing Zang, Xing Zheng, Haobo Wang, Yingfan Ma, Bowen Song, Bo Han, Bo An, Lei Feng, Weiqiang Wang, Junbo Zhao, Gang Chen  
**Category**: cs.LG  
**Published**: 2026-09-04  
**Score**: 13.0  
**Type**: new  
**ArXiv ID**: 2609.03324v1  

#### Abstract
Reinforcement learning with verifiable rewards (RLVR) improves large language model reasoning, but its practical scaling is constrained by expensive on-policy rollouts and the cost of obtaining reliable targets at scale. Existing methods address sample selection, incomplete supervision, or noisy lab...

---

### 2. [LeanStream: A Speculate-and-Refine Streaming Framework for Efficient on-Device LLM Inference](https://arxiv.org/abs/2609.03079)

**Authors**: Renyuan Liu (Richard), Yuyang Leng (Richard), Kaiyan Liu (Richard), Yuzhou Zhong (Richard), Shaohan Hu (Richard),  Chun-Fu (Richard),  Chen, Peijun Zhao, Heechul Yun, Shuochao Yao  
**Category**: cs.LG  
**Published**: 2026-09-04  
**Score**: 12.5  
**Type**: new  
**ArXiv ID**: 2609.03079v1  

#### Abstract
On-device LLM inference is attractive for privacy and responsiveness, but remains challenging on mobile and embedded devices because model weights far exceed available DRAM. Prior systems exploit activation sparsity and offload weights to SSD or flash storage, but face a fundamental systems trade-of...

---

### 3. [Para-Pipe: Exploiting Hierarchical Operator Parallelism of ML Computational Graphs on SoCs](https://arxiv.org/abs/2609.04168)

**Authors**: Yujie Zhang, Huiying Lan, Ehsan Aghapour, Zhiyuan Ning, Peng Zan, Weidong Shao, Anuj Pathania, Tulika Mitra  
**Category**: cs.DC  
**Published**: 2026-09-04  
**Score**: 12.0  
**Type**: new  
**ArXiv ID**: 2609.04168v1  

#### Abstract
As edge-based deep learning applications become more complex, optimizing performance on heterogeneous System-on-Chips (SoCs) presents unique challenges. Traditional pipelining techniques distributing the computation across different on-chip processing units, while effective for throughput, do not ad...

---

### 4. [Why Gated DeltaNet Survives 4-Bit Quantization: NVFP4 W4A4 for the Recurrent Half of a Hybrid 27B LLM](https://arxiv.org/abs/2609.04098)

**Authors**: Sergii Kozyrev, Davyd Maiboroda  
**Category**: cs.AI  
**Published**: 2026-09-04  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2609.04098v1  

#### Abstract
Hybrid LLMs pair softmax attention with linear-attention layers such as Gated DeltaNet (GDN), whose recurrent state summarizes the context in fixed size. Early community 4-bit quantizations of Qwen3.8-27B (48 GDN layers, 16 attention layers) left the GDN block in 8- or 16-bit precision -- especially...

---

### 5. [Margins, Not Windows: Training-Free Per-Step Lossy Speculative Decoding](https://arxiv.org/abs/2609.02897)

**Authors**: Oszk\'ar Urb\'an, Young D. Kwon, Stylianos I. Venieris, Cecilia Mascolo  
**Category**: cs.CL  
**Published**: 2026-09-04  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2609.02897v1  

#### Abstract
Speculative decoding accelerates LLM inference by drafting candidate tokens and verifying them in parallel. Tree-attention drafters such as EAGLE-3 are widely adopted, yet typically hold two decisions fixed: (1) a strict token-match verification rule and (2) a static draft-tree shape. Prior work rel...

---

### 6. [Hardware-Aware FP4 FlashAttention-4](https://arxiv.org/abs/2609.04105)

**Authors**: Robert Hu  
**Category**: cs.LG  
**Published**: 2026-09-04  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2609.04105v1  

#### Abstract
Blackwell's 4-bit floating-point (FP4) tensor cores do not automatically make attention faster because softmax conversion and on-chip dependencies dominate once its matrix products shrink. We address this with \emph{Direct-P} for noncausal inference and a causal path that passes the forward quantiza...

---

### 7. [BASP: Communication-Efficient Batch-Aware Sequence Parallelism for LLM Training](https://arxiv.org/abs/2609.03151)

**Authors**: Bigyan Ghimire, Jon C. Calhoun  
**Category**: cs.DC  
**Published**: 2026-09-04  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2609.03151v1  

#### Abstract
Long-context reasoning for large language models (LLMs) is becoming increasingly important, but training over long sequences remains challenging due to massive memory and communication requirements. Sequence parallelism has emerged as an essential technique for addressing bottlenecks in long sequenc...

---

### 8. [Unlocking Lossless Speedups in LLMs via Discrete Diffusion](https://arxiv.org/abs/2609.04010)

**Authors**: Subham Sekhar Sahoo, Lingjie Chen, Khiem Pham, Jonathan Geuter, Chaitanya Dwivedi, Varad Pimpalkhute, Yash Akhauri, Alexander Moreno, Mikhail Yurochkin, Zhenting Wang, Mostafa Elhoushi, Nolan Dey, Shane Bergsma, Joel Hestness, John Thickstun, Eric Xing, Zhengzhong Liu  
**Category**: cs.LG  
**Published**: 2026-09-04  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2609.04010v1  

#### Abstract
Large Language Models (LLMs) owe much of their success to next-token prediction (NTP), but their autoregressive (AR) structure requires slow, sequential token generation. To overcome this bottleneck, we introduce diffusion-augmented LLMs, a new class of models that defines an AR model distribution w...

---

### 9. [Two-Stage Reinforcement Learning for Sound and Adversarial Test Generation in Code LLMs](https://arxiv.org/abs/2609.03955)

**Authors**: Jiacheng Xu, Wentao Zhang, Zhiyi Lyu, Fuxiang Zhang, Chaojie Wang, Yang Liu, Bo An  
**Category**: cs.CL  
**Published**: 2026-09-04  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2609.03955v1  

#### Abstract
Reinforcement learning (RL) has substantially advanced code generation with large language models (LLMs) through executable feedback. The feedback for coding problems mainly comes from specific test cases, where high-quality test cases are often scarce since they should be both sound and discriminat...

---

### 10. [SVG-Score: Human-Aligned Evaluation of Text-to-SVG Generation](https://arxiv.org/abs/2609.03806)

**Authors**: Marco Cipriano, Leonardo Zini, Alexandra Schild, Valentin Teutschbein, Afsana Mimi, Marcella Cornia, Lorenzo Baraldi, Gerard de Melo  
**Category**: cs.AI  
**Published**: 2026-09-04  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2609.03806v1  

#### Abstract
Scalable Vector Graphics (SVG) generation is attracting increasing attention as generative models improve in expressiveness and controllability. Progress, however, is held back by the lack of domain-specific evaluation protocols: current practice relies on metrics designed for natural images, most n...

---

### 11. [RL-ADA: A World-Feedback Framework for Adversarially Robust Enterprise Dialogue Agents](https://arxiv.org/abs/2609.02902)

**Authors**: Ram Narayanan, Harshit Rajgarhia, Abhishek Mukherji  
**Category**: cs.CL  
**Published**: 2026-09-04  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2609.02902v1  

#### Abstract
Deploying task-oriented dialogue agents in enterprise customer support faces a persistent annotation bottleneck: robust training requires labelled interaction data at scale, yet enterprise conversational logs are privacy-sensitive and expensive to annotate, while user behaviour evolves faster than l...

---

### 12. [GrowPage: On-Demand KV Budgeting for Efficient LLM Reasoning Serving](https://arxiv.org/abs/2609.03494)

**Authors**: Qiankun Ma, Yanjiang Zhou, Zinan Xiong, Haofei Wang, Zhen Song, Yang Xiang, Ziyao Zhang, Hairong Zheng  
**Category**: cs.AI  
**Published**: 2026-09-04  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2609.03494v1  

#### Abstract
Long-output reasoning has made the key--value (KV) cache a critical memory bottleneck for efficient LLM serving. Existing KV compression methods usually rely on a predefined per-request budget and adjust only which KV states are retained, leaving the total capacity fixed throughout decoding. However...

---

### 13. [SGD-KV: Summarization Guided KV Cache Compression](https://arxiv.org/abs/2609.03235)

**Authors**: Zeyu Liu, Woomin Song, Xuandi Fu, Sai Muralidhar Jayanthi, Vivek Govindan, Aram Galstyan, Sravan Babu Bodapati, Srikanth Ronanki  
**Category**: cs.CL  
**Published**: 2026-09-04  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2609.03235v1  

#### Abstract
Large language models (LLMs) face severe memory bottlenecks in long-context inference due to the linearly growing size of key-value (KV) caches. Existing KV cache compression techniques typically rely on simple heuristics, overlooking the distinct functional roles of different attention heads. We pr...

---

### 14. [Jina-OCR-v1: Efficient Document Parsing with Speculative Decoding and Dense Verifiable Rewards](https://arxiv.org/abs/2609.03181)

**Authors**: Alejandro Bar\'on Garc\'ia, Feng Wang, Emilia Garcia Casademont, Han Xiao  
**Category**: cs.CL  
**Published**: 2026-09-04  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2609.03181v1  

#### Abstract
We present Jina-OCR-v1, an end-to-end document parsing model built to serve on low-budget GPUs. It combines the compressed-vision encoder and the 3B mixture-of-experts decoder of DeepSeek-OCR, which activates about 570M parameters per token, with a FastMTP speculative decoding head that shares a sin...

---

### 15. [Risk and Anomaly Identification for Distribution Network Optimal Operation Based on Reinforcement Learning and Uncertainty Quantification](https://arxiv.org/abs/2609.03308)

**Authors**: Ziqi Zhang  
**Category**: cs.LG  
**Published**: 2026-09-04  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2609.03308v1  

#### Abstract
Reliable operation of modern distribution networks requires timely identification of operational risks and anomalous events under pervasive uncertainty. In practice, operators must identify risks that are inherent in stochastic yet in-distribution conditions, and anomalies that correspond to out-of-...

---

### 16. [High-Dimensional Learning Dynamics of Attention-Indexed Models](https://arxiv.org/abs/2609.03858)

**Authors**: Yizhou Xu, Margarita Sagitova, Lenka Zdeborov\'a, Florent Krzakala  
**Category**: cs.LG  
**Published**: 2026-09-04  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2609.03858v1  

#### Abstract
Attention mechanisms are central to modern foundation models, yet their training dynamics remain poorly understood, especially when the attention matrices have extensive rank. In this work, we study attention-indexed models, a broad framework that can represent multi-layer and multi-head attention a...

---

### 17. [Speculative Macro Commit for Faster Tool-Using Agents](https://arxiv.org/abs/2609.03236)

**Authors**: Zeyu Liu, Souvik Kundu, Peter A. Beerel  
**Category**: cs.AI  
**Published**: 2026-09-04  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.03236v1  

#### Abstract
Tool-using LLM agents spend wall-clock time not only on model inference but also in serial action--observation turns, where each tool call, environment transition, and observation can delay subsequent decisions. We introduce \textbf{Speculative Macro Commit} (SMC), a runtime mechanism for a two-tier...

---

### 18. [LLM4CKD: Large Language Models for Early Stage Chronic Kidney Disease Screening](https://arxiv.org/abs/2609.04013)

**Authors**: Muhammad Ashad Kabir, Sirajam Munira  
**Category**: cs.AI  
**Published**: 2026-09-04  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.04013v1  

#### Abstract
Early screening of chronic kidney disease (CKD) is critical for timely intervention, yet most machine learning (ML) and deep learning (DL) approaches require labeled data and model training, limiting their use in real-world screening settings. This study evaluates the effectiveness of large language...

---

### 19. [Less Is Moral: A CHARMing Framework for Moral Foundations Detection in Endorsement Behaviour](https://arxiv.org/abs/2609.03330)

**Authors**: Huixiang Fu, Marian-Andrei Rizoiu  
**Category**: cs.CL  
**Published**: 2026-09-04  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.03330v1  

#### Abstract
Moral language plays a central role in shaping online endorsement and the diffusion of information, yet existing moral foundation detection systems often suffer from poor cross-domain generalization, weak rationale grounding, and reliance on costly prompting-based large language models (LLMs). We in...

---

### 20. [Alignment-Free Text-Audiobox for Voice Dubbing and Full-Duplex Dialogue Synthesis](https://arxiv.org/abs/2609.03992)

**Authors**: Sanyuan Chen, Min-Jae Hwang, Sho Inoue, Anna Sun, Bokai Yu, David Kant, Dongmin Hyun, Dorian Desblancs, Gregory Antonovsky, Oleg Repin, Peng-Jen Chen, Xutai Ma, Zehai Tu, Juan Pino, Wei-Ning Hsu  
**Category**: cs.CL  
**Published**: 2026-09-04  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.03992v1  

#### Abstract
We present Alignment-Free Text-Audiobox (Text-AB), a unified framework for high-quality voice dubbing and full-duplex dialogue synthesis. Building on a Diffusion Transformer trained with a flow-matching objective, Text-AB departs from the Audiobox system along three dimensions. First, it operates in...

---

### 21. [Skywing: A Platform for Decentralized Mathematical Computing in Unreliable Environments](https://arxiv.org/abs/2609.03145)

**Authors**: Alyson Fox, Colin Ponce, Annika Mauro, Wayne Mitchell, Sarah Osborn, Tom Benson, Shayna Kapadia  
**Category**: cs.DC  
**Published**: 2026-09-04  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.03145v1  

#### Abstract
Emerging edge, autonomous, and cyber-physical systems increasingly require mathematical computation across heterogeneous devices connected by unreliable communication networks. Traditional high-performance computing and distributed data-processing frameworks provide powerful abstractions for managed...

---

### 22. [Equation Recast for Canonical Operator Learning Across Parametric PDEs](https://arxiv.org/abs/2609.02982)

**Authors**: Qiyun Cheng, Valentin Duruisseaux, Cesar F. Clauser, Md Hossain Sahadath, Huihua Yang, Shaowu Pan, Nathaniel Ferraro, Anima Anandkumar, Wei Ji, Cristina Rea  
**Category**: cs.LG  
**Published**: 2026-09-04  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.02982v1  

#### Abstract
Learning solution operators across broad parameter ranges can require substantial coverage of both input functions and physical parameters, particularly for purely data-driven parametric models. In addition, the resulting models may fail silently outside the training distribution. We introduce equat...

---

### 23. [Tail-Likelihood Reinforcement Learning](https://arxiv.org/abs/2609.02987)

**Authors**: Shrinivas Ramasubramanian, Daman Arora, Fahim Tajwar, Guanning Zeng, Qingyang Wu, Zhongzhu Zhou, Chenfeng Xu, Haiwen Feng, Yuda Song, Aarti Singh, Ruslan Salakhutdinov, J. Andrew Bagnell, Jeff Schneider, Andrea Zanette  
**Category**: cs.LG  
**Published**: 2026-09-04  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.02987v1  

#### Abstract
Reinforcement learning typically optimizes average reward. For generative policies, the average can hide an important distinction: two policies can achieve the same mean reward while having very different chances of producing a rare but high-reward rollout. This matters as sampling increases during ...

---

### 24. [Coupled Scaling: A Representational Accessibility Framework for Neural Scaling Laws](https://arxiv.org/abs/2609.03533)

**Authors**: Jie Wang  
**Category**: cs.LG  
**Published**: 2026-09-04  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.03533v1  

#### Abstract
Existing theories derive neural scaling from data geometry or a specified data-model spectrum, but systems trained on the same data can scale differently when architecture or optimization changes the representations they can efficiently reach. We introduce Coupled Scaling, a task-conditioned framewo...

---

### 25. [R$^{2}$Adapter: A Routing and Rewriting Adapter for Efficient Hybrid RAG](https://arxiv.org/abs/2609.02894)

**Authors**: Yucan Guo, Miao Su, Saiping Guan, Long Bai, Zhongni Hou, Zixuan Li, Xiaolong Jin, Jiafeng Guo, Xueqi Cheng  
**Category**: cs.CL  
**Published**: 2026-09-04  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2609.02894v1  

#### Abstract
Retrieval-Augmented Generation (RAG) has become a prevailing paradigm for enhancing Large Language Models (LLMs) with non-parametric knowledge. Vanilla RAG efficiently handles simple queries but struggles with relational or multi-hop reasoning. Graph-based RAG alleviates this issue but incurs higher...

---

### 26. [Every Kernel Is a Join: Automatic Multi-GPU Parallelism for AI Computations in Einsummable](https://arxiv.org/abs/2609.03905)

**Authors**: Zhimin Ding, Chen-Kuan Liao, Chima Adiole, Brianna Barrow, Fangzhou Du, Yu Hsiao, Ge Huang, Yicheng Jin, Ismail Syed, Chris Jermaine  
**Category**: cs.DC  
**Published**: 2026-09-04  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2609.03905v1  

#### Abstract
Distributing an AI computation across the GPUs of a multi-GPU server is one of the central problems in systems-for-AI. We present Einsummable, a prototype system that accepts a PyTorch-like description of an AI computation and automatically distributes it across a multi-GPU server, with no device as...

---

### 27. [AutoGraphForge: Towards Automated Graph Theory Discovery](https://arxiv.org/abs/2609.03478)

**Authors**: J\'an Pastorek  
**Category**: cs.AI  
**Published**: 2026-09-04  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.03478v1  

#### Abstract
We report on our ongoing project to develop a computational pipeline, AutoGraphForge, for an automated graph-theoretic conjecturing-refuting-formalizing-proving system. Conjecture generation is counterexample-guided and runs in rounds: a Graffiti3 generator proposes conjectures over a small, evolvin...

---

### 28. [NeoRed: A Knowledge-Logic-Alignment Multimodal Large Language Model for Neonatal Respiratory Disease Diagnosis](https://arxiv.org/abs/2609.03527)

**Authors**: Yinan Liu, Hongtai Xia, Haoran Xu, Jiankang Hong, Jingkuan Song, Ye Luo  
**Category**: cs.AI  
**Published**: 2026-09-04  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.03527v1  

#### Abstract
Neonatal respiratory diseases are a major cause of neonatal morbidity and mortality, posing substantial challenges in clinical practice. Despite recent advances, existing Multimodal Large Language Models (MLLMs) face two key limitations in neonatal diagnosis: (1) domain gap arising from predominantl...

---

### 29. [Synthetic Semantic Supervision for Contrastive Code Representation Learning in Small Transformers: An Empirical Study](https://arxiv.org/abs/2609.03702)

**Authors**: Kenneth Paulsen, Florian Tambon, Mike Papadakis, Shin Yoo  
**Category**: cs.AI  
**Published**: 2026-09-04  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.03702v1  

#### Abstract
General-purpose code embeddings power tools for code search, classification, and retrieval. Compact transformer encoders for code typically rely on either human-written docstrings (labor-intensive and inconsistent) or mined structural signals such as execution traces (setting-specific and costly to ...

---

### 30. [Mesh-Native Physics-Informed Graph Surrogates for TCAD-in-the-Loop Design Space Exploration](https://arxiv.org/abs/2609.02988)

**Authors**: Leonid Popryho, Ayoub Sadeghi, Inna Partin-Vaisband  
**Category**: cs.LG  
**Published**: 2026-09-04  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.02988v1  

#### Abstract
High-fidelity TCAD simulation of drift-diffusion transport remains the workhorse of emerging FinFET device design, but it is computationally expensive, especially for 3D structures where runtime escalates steeply with mesh complexity. This sharply limits multi-objective design space exploration. Exi...

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
