# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-11-09 14:18:33 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [ResServe: Accurate and Efficient FP4 LLM Inference with Residue Compensation](https://arxiv.org/abs/2611.08713)

**Authors**: Shuo Rong, Ruibo Fan, Jiawen Wen, Xiaowen Chu  
**Category**: cs.DC  
**Published**: 2026-11-09  
**Score**: 15.0  
**Type**: new  
**ArXiv ID**: 2611.08713v1  

#### Abstract
Block-scaled 4-bit floating-point (FP4) quantization reduces large language model inference costs but can degrade task quality. Higher-precision computation can mitigate this loss at the expense of some of the efficiency gains from FP4 inference. Narrow matrix computations common in decoding often l...

---

### 2. [RL-ATTACK: A Generalized Black-Box Adversarial Attack on Language Models via Reinforcement Learning](https://arxiv.org/abs/2611.08711)

**Authors**: Taikun Chen, Yue Zhang, Wei Qian, Hui Fang  
**Category**: cs.CL  
**Published**: 2026-11-09  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2611.08711v1  

#### Abstract
Large language models (LLMs) exhibit vulnerabilities to adversarial perturbations that can induce unintended responses. These vulnerabilities can be identified via intentional attacks and subsequently used to improve model safety. In black-box settings where gradients are inaccessible, existing meth...

---

### 3. [Adaptive Scheduling Pipeline for Multi-Instance Asynchronous Reinforcement Learning](https://arxiv.org/abs/2611.08864)

**Authors**: Salah Chikhi, Luis Zuin, Entong Li, Zeng Li  
**Category**: cs.DC  
**Published**: 2026-11-09  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2611.08864v1  

#### Abstract
We present the Adaptive Scheduling Pipeline (ASP), an offline rollout scheduler for multi-instance asynchronous reinforcement learning (RL) of large language models (LLMs). ASP jointly reduces three sources of rollout underutilization - padding bubbles from intra-minibatch length heterogeneity, skew...

---

### 4. [AutoParallel: Composable, PyTorch-Native Automatic Parallelism for Production Training](https://arxiv.org/abs/2611.08660)

**Authors**: Kaijian Wang, Sanket Purandare, Aditya Venkataraman, Will Constable, Ivan Kobzarev, Elias Ellison, Simon Fan, Edward Z. Yang, Davide Italiano, Patrick Labatut, Yuke Wang, Francisco Massa  
**Category**: cs.DC  
**Published**: 2026-11-09  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2611.08660v1  

#### Abstract
Training large models requires multiple interacting forms of parallelism. As models and clusters scale, finding and maintaining efficient parallelization plans becomes essential for training performance. Diverse model architectures, workloads, and hardware topologies demand flexible strategies, yet ...

---

### 5. [DUET: Dual-Sparse Inference for Block Diffusion Language Models](https://arxiv.org/abs/2611.09239)

**Authors**: Siyuan Yan, Shen Gu, Xiaoxing Ma, Chun Cao, Jingwei Xu  
**Category**: cs.CL  
**Published**: 2026-11-09  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2611.09239v1  

#### Abstract
Diffusion language models (DLMs) enable parallel token generation through iterative denoising, yet their inference remains costly due to repeated attention over a growing historical prefix and repeated processing of the entire current generation block. Historical attention becomes increasingly expen...

---

### 6. [CodeQuant2: Enhanced Outlier Mitigation and Adaptive Clustering for Low-Precision LRMs](https://arxiv.org/abs/2611.09868)

**Authors**: Xingyu Liu, Xiangyang Yin, Tianhua Xia, Haiyu Wang, Sai Qian Zhang  
**Category**: cs.CL  
**Published**: 2026-11-09  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2611.09868v1  

#### Abstract
Large reasoning models (LRMs) improve task performance by generating longer intermediate reasoning traces, but this substantially increases inference costs and makes low-precision quantization increasingly important. However, aggressively quantized models often perform poorly on reasoning tasks. We ...

---

### 7. [More with Less: Efficient Parallel Scientific Code Translation via an Agent-Skill Workflow](https://arxiv.org/abs/2611.08947)

**Authors**: Jianuo Shen, Pengyi Xu, Bo Zhang  
**Category**: cs.DC  
**Published**: 2026-11-09  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2611.08947v1  

#### Abstract
Translating parallel scientific codes between HPC programming frameworks is a critical but labor-intensive task. Existing LLM-based approaches rely on fixed sequential pipelines with undifferentiated repair strategies and correctness criteria that do not match scientific computing practice. We prese...

---

### 8. [DARIC: A Position-Independent KV Cache Reuse System with Distribution-Aware Recomputation](https://arxiv.org/abs/2611.09109)

**Authors**: Weikang Wang, Haiyang Liu, Xin Zhou, Jun Liu, Weifeng Zhang  
**Category**: cs.DC  
**Published**: 2026-11-09  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2611.09109v1  

#### Abstract
The context lengths of large language models (LLMs) are rapidly increasing in real-world applications such as agentic systems and retrieval-augmented generation (RAG). To accelerate LLM inference in these scenarios, reusing position-independent key-value (KV) caches across requests has emerged as a ...

---

### 9. [ISER-MoE: Dynamic In-Switch Replication of Hot Experts for Load Balancing in Expert-Parallel](https://arxiv.org/abs/2611.09008)

**Authors**: Jintao Peng, Ethan Zhang, Kefeng Duan, Bangyu Sheng, Min Li, Itamar Rosenfeld Rauch, Jack Yang, June Yang  
**Category**: cs.DC  
**Published**: 2026-11-09  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2611.09008v1  

#### Abstract
Expert-parallel mixture-of-experts (MoE) layers run at the pace of the most loaded GPU, and a few hot experts carry most of the excess load. Replicating them at every MoE layer balances the load only if their weights arrive within the layer, and broadcasting from the owner is the bottleneck: point-t...

---

### 10. [Nebula: Distributed Position-Independent Caching for LLM Serving](https://arxiv.org/abs/2611.09474)

**Authors**: Yifei Liu, Junbin Pan, Junhao Hu, Yang Liu, Junyi Chen, Juntong Wu, Chen Chen, Haiyuan Zhao, Xiaoxu Chen, Weihang Chen  
**Category**: cs.DC  
**Published**: 2026-11-09  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2611.09474v1  

#### Abstract
Modern LLM workflows, such as retrieval-augmented generation (RAG) and LLM agents, build long prompts from multiple independent text segments, making prefill a major source of computation. Reusing KV caches is therefore important for reducing serving costs and latency: distributed caching systems sp...

---

### 11. [Steerable Multi-Objective Group-Relative Preference Optimization with Jointly Trained Low-Rank Adapters](https://arxiv.org/abs/2611.09634)

**Authors**: Constantinos Karouzos, Nikolaos Aletras  
**Category**: cs.CL  
**Published**: 2026-11-09  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2611.09634v1  

#### Abstract
Language model deployment often requires different trade-offs between competing objectives as user preferences change, yet standard post-training methods typically produce a single policy for a fixed objective weighting. We introduce LambdaMix, a new parameter efficient method for learning a prefere...

---

### 12. [The Invisible Gap: Romanized Indic Generation Fails Where Automatic Metrics Succeed](https://arxiv.org/abs/2611.09085)

**Authors**: Madhu Sudhan Rao Beesetty, Surbhi Sharma, Devesh Maheshwari, Lokesh Meesala  
**Category**: cs.CL  
**Published**: 2026-11-09  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2611.09085v1  

#### Abstract
Hundreds of millions of Indians communicate with digital systems in romanized Latin characters -- typing "aap kaise hain" rather than native script -- yet romanized Indic generation is absent from NLP benchmarks and evaluation frameworks. We demonstrate that automatic metrics (BGE-M3, BERTScore) sys...

---

### 13. [MARS: Training Multilingual Search Agents for Retriever-Free Evidence Acquisition](https://arxiv.org/abs/2611.09090)

**Authors**: Rui Qi, Yunlong Liang, Yufeng Chen, Jinan Xu, Fandong Meng, Kaiyu Huang  
**Category**: cs.CL  
**Published**: 2026-11-09  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2611.09090v1  

#### Abstract
Existing multilingual retrieval-augmented generation relies heavily on learned retrievers and similarity-based ranking from multilingual collections, which are vulnerable to cross-lingual retrieval errors and indexing overhead. In this paper, we explore retriever-free multilingual search, where the ...

---

### 14. [KLEAR: Knowledge-Matched LLM Evaluation in Korean across Application Requirements via Controlled Benchmark Transformation](https://arxiv.org/abs/2611.09223)

**Authors**: Seoyeon Kim, Jeonghyun Kang, Hamin Koo, Barom Kang, Heondeuk Lee, Jaehyung Kim  
**Category**: cs.CL  
**Published**: 2026-11-09  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2611.09223v1  

#### Abstract
Existing Korean benchmarks for Large Language Models (LLMs) evaluate performance across diverse knowledge domains, but their accuracy scores cannot distinguish whether incorrect responses arise from a lack of required knowledge or a failure to apply that knowledge correctly. Our analysis of existing...

---

### 15. [The GPU Needs an Operating System](https://arxiv.org/abs/2611.08720)

**Authors**: Reza Nourai, Behnam Bastani  
**Category**: cs.DC  
**Published**: 2026-11-09  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2611.08720v1  

#### Abstract
AI chips sit mostly idle while serving models, because replies must be fast and each step waits on memory. On two large open models, the math units were busy only about 6-11% of the time. We argue the GPU needs an operating system that keeps urgent work on schedule and fills idle capacity with other...

---

### 16. [Inside the Trap: A Renewal Law for Prefix-Cache Collapse in LLM Agent Serving](https://arxiv.org/abs/2611.08902)

**Authors**: Akshat Valse, Ali Jannesari  
**Category**: cs.DC  
**Published**: 2026-11-09  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2611.08902v1  

#### Abstract
Prefix caching lets an LLM agent's follow-up turns reuse cached KV, but a burst can tip an engine into a trap in which every turn misses and the backlog does not drain, at a load it served before. In registered paired tests on two engines, two GPUs and two models (passed at 9 of 10 rates), a 200-ses...

---

### 17. [DataHelix: A CPU-GPU Co-Scheduling Data Plane for Multimodal Model Training](https://arxiv.org/abs/2611.09411)

**Authors**: Yuanhe Li, Jiazhen Wu, Dian Ding, Guangtao Xue, Yiming Zhang  
**Category**: cs.DC  
**Published**: 2026-11-09  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2611.09411v1  

#### Abstract
Multimodal model training breaks the loose coupling between data loading and model execution assumed by conventional training pipelines. Large per-sample variation in decoding cost, token/latent count, and tensor size causes both CPU preprocessing time and GPU compute time to vary substantially acro...

---

### 18. [Syn2DRE: Synergistic Dual Teacher-Student Framework for Document-level Relation Extraction](https://arxiv.org/abs/2611.09342)

**Authors**: Khai Phan Tran, Xue Li  
**Category**: cs.CL  
**Published**: 2026-11-09  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2611.09342v1  

#### Abstract
Document-level Relation Extraction (DocRE) aims to identify relations between entities across a document but typically relies on costly large-scale annotations. While pseudo-labelling enables the use of low-quality data, existing methods often suffer from inefficiency and confirmation bias. To addre...

---

### 19. [RuACT: Recovering Inactive Rubrics as Adaptive Critics and Teachers](https://arxiv.org/abs/2611.09597)

**Authors**: Yijun Li, Zhengxi Lu, Zhipiao Liu, Jinzheng Yu, Xutao Weng, Jian Shao, Yongliang Shen  
**Category**: cs.CL  
**Published**: 2026-11-09  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2611.09597v1  

#### Abstract
Rubric-based reinforcement learning (RL) extends RL with verifiable rewards to open-ended tasks by scoring each response against query-specific rubrics. Under group-relative optimization, however, a rubric that every sampled response fails (unreachable) or passes (saturated) gives all responses the ...

---

### 20. [Capital Efficiency, Isolation, Risk, and Ruin in Generative AI Inference Serving Infrastructure](https://arxiv.org/abs/2611.08999)

**Authors**: Ren\'ee St. Amant, Anjaly Parayil, Ankur Mallick, Victor R\"uhle  
**Category**: cs.DC  
**Published**: 2026-11-09  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2611.08999v1  

#### Abstract
Generative AI serving is an extremely capital-intensive endeavor that demands rapid, step-change improvements in capital efficiency to prevent industry wide increases in the cost of capital from limiting growth plans. Keeping pace with the economic necessities will increasingly require infrastructur...

---

### 21. [Towards Sustainable Federated Learning via Carbon-Aware Client Selection](https://arxiv.org/abs/2611.09552)

**Authors**: Wentao Gao, Omid Tavallaie, Albert Y. Zomaya  
**Category**: cs.DC  
**Published**: 2026-11-09  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2611.09552v1  

#### Abstract
The Information/Communication Technology (ICT) industry is severely challenging the global environment due to the sharply rising carbon emissions generated by machine learning (ML) workloads. Federated Learning (FL) is one of the prominent distributed ML systems used in applications with billions of...

---

### 22. [Concurrent Use of FP32 and FP64 Computing Units on GPUs and Mixed-Precision Placement in WENO Reconstruction](https://arxiv.org/abs/2611.09651)

**Authors**: Jun Hatayama, Daichi Mukunoki, Yusuke Onishi, Atsushi Sekimoto, Toshinori Kouchi  
**Category**: cs.DC  
**Published**: 2026-11-09  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2611.09651v1  

#### Abstract
Scientific computing often requires FP64 arithmetic, although FP64 performance is limited on NVIDIA GPUs. Double-float (DF) emulation, which represents a value using two FP32 values with about 48 bits of significand, outperforms native FP64 where FP64 units are limited. On the other hand, mixed-prec...

---

### 23. [Query-Guided RoPE Modulation for Lost-in-the-Middle Mitigation in Long-Context LLMs](https://arxiv.org/abs/2611.09554)

**Authors**: Peiyu Zhang, Heng Ping, Shinibali Bhattacharyya, Anwesan Pal, Nikos Kanakaris, Kevin Cheang, Shixuan Li, Wei Yang, Paul Bogdan  
**Category**: cs.CL  
**Published**: 2026-11-09  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2611.09554v1  

#### Abstract
Long-context language models often neglect information in the middle of the input, a phenomenon known as "lost-in-the-middle". Part of this gap may stem from the long-range decay of rotary position embedding (RoPE), which weakens the positional signal between distant tokens. Existing input compressi...

---

### 24. [ReKV: Overlapping Quantized KV Transfer and Suffix Recomputation](https://arxiv.org/abs/2611.08770)

**Authors**: Guangjian Dong  
**Category**: cs.DC  
**Published**: 2026-11-09  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2611.08770v1  

#### Abstract
Prefill--decode disaggregation moves a prompt's key--value (KV) cache between workers before the first token can be generated. Reducing this handoff to either cache compression or recomputation misses a useful choice: which layers to transmit, at what precision, and which layers to execute again on ...

---

### 25. [EventHOI: Event-Driven Resource-Efficient Hand-Object Interaction Logging on Smartglasses](https://arxiv.org/abs/2611.09089)

**Authors**: Le Zhang, Kailai Cui, Hao Chen, Vlad Roznyatovskiy, Jianzhong Zhang, Anhong Guo, Kang G. Shin, Ke Sun  
**Category**: cs.DC  
**Published**: 2026-11-09  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2611.09089v1  

#### Abstract
Hand-object interactions (HOIs) can be captured by smartglasses to learn how people manipulate objects in daily life. Automatic logging of these interactions from an egocentric perspective provides time-stamped evidence of daily actions for applications like memory assistance. However, continuous RG...

---

### 26. [Fast and Actionable Classification of HPC I/O Time Series](https://arxiv.org/abs/2611.09646)

**Authors**: Maxime Gonthier (RIKEN CCS), Luan Teylo (RIKEN CCS), Francesco Antici (RIKEN CCS), Mihail Popov (LaBRI, UB), Francieli Boito (LaBRI, UB)  
**Category**: cs.DC  
**Published**: 2026-11-09  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2611.09646v1  

#### Abstract
Input/output (I/O) is a major bottleneck for highperformance computing (HPC) performance. One of the challenges is the contention between jobs when sharing the parallel file system. However, tackling this issue depends on knowing the temporal behavior of jobs. In this paper, we propose a clusteringb...

---

### 27. [Failure Profiling and Reachable Trajectory Selection for Reasoning Distillation](https://arxiv.org/abs/2611.09042)

**Authors**: Yijun Zhu, Jianxin Wang, Chengchao Shen  
**Category**: cs.CL  
**Published**: 2026-11-09  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2611.09042v1  

#### Abstract
Training smaller language models on reasoning solutions generated by a larger teacher model is the prevailing approach for transferring mathematical reasoning capabilities. However, the student tends to overfit the output distribution of the teacher and generalizes poorly when its own behavior at in...

---

### 28. [ROSE: Risk-Aware Orthogonal Subspace Navigation for Lifelong Knowledge Editing in Multimodal Large Language Models](https://arxiv.org/abs/2611.09770)

**Authors**: Lingyun Song, Ziyao Chen, Kang Pan, Xiaolin Han, Yifei Zhang, Xiaoqi Wang, Yudai Pan, Xuequn Shang  
**Category**: cs.CL  
**Published**: 2026-11-09  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2611.09770v1  

#### Abstract
Multimodal Large Language Models (MLLMs) require continuous factual updates to stay current, yet lifelong knowledge editing remains a daunting challenge. The primary obstacle is the severe Locality Erosion and Catastrophic Forgetting, where sequential edits inevitably interfere with pre-trained capa...

---

### 29. [Contextuality and Contradictions: Exploring Real-World Identifications of AI-Generated Writing](https://arxiv.org/abs/2611.08704)

**Authors**: Michelle Morgenstern, Julia Witte Zimmerman, Tabia Tanzin Prama, Kristine Harootunian, June Clark, Bryn Loftness, Calla Beauregard, Juniper Lovato  
**Category**: cs.CL  
**Published**: 2026-11-09  
**Score**: 3.5  
**Type**: new  
**ArXiv ID**: 2611.08704v1  

#### Abstract
As AI-generated content spreads into nearly every domain of human interaction, practices and discourses of AI detection, both formal and informal, have become an everyday part of how people interact with text and with each other. In this paper, we explore the real-world practices and discourses of A...

---

### 30. [Guardrail Erosion in Domain Fine-Tuned Large Language Models: A Paired Audit of Knowledge-Centric Abstention in Biomedical LLMs](https://arxiv.org/abs/2611.08780)

**Authors**: Stephen Peng, Vineet Jagadeesan Nair, Noah Broestl, Anne Friederike Kleppe, Steven Mills, Adi Zolotov  
**Category**: cs.CL  
**Published**: 2026-11-09  
**Score**: 3.5  
**Type**: new  
**ArXiv ID**: 2611.08780v1  

#### Abstract
The number of domain fine-tuned models, particularly in the biomedical domain, is rapidly expanding. Fine-tuning promises high-performing models at a fraction of the cost of training a base model, but it can also produce unintended behavioral changes. Multiple studies have shown that safety guardrai...

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
