# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-04-30 07:52:49 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [When Hidden States Drift: Can KV Caches Rescue Long-Range Speculative Decoding?](https://arxiv.org/abs/2604.26412)

**Authors**: Tianyu Liu, Yuhao Shen, Xinyi Hu, Baolin Zhang, Hengxin Zhang, Jun Dai, Jun Zhang, Shuang Ge, Lei Chen, Yue Li, MingCheng Wan  
**Category**: cs.CL  
**Published**: 2026-04-30  
**Score**: 12.0  
**Type**: new  
**ArXiv ID**: 2604.26412v1  

#### Abstract
Speculative decoding accelerates LLM inference, but SOTA hidden-state-based drafters suffer from long-range decay: draft accuracy degrades as the speculative step increases. Existing work attributes this decay to train-inference mismatch and proposes test-time training (TTT) as a remedy, yet we obse...

---

### 2. [Folding Tensor and Sequence Parallelism for Memory-Efficient Transformer Training & Inference](https://arxiv.org/abs/2604.26294)

**Authors**: Vasu Shyam, Anna Golubeva, Quentin Anthony  
**Category**: cs.CL  
**Published**: 2026-04-30  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2604.26294v1  

#### Abstract
We present tensor and sequence parallelism (TSP), a parallel execution strategy that folds tensor parallelism and sequence parallelism onto a single device axis. In conventional multi-dimensional parallelism layouts, tensor parallelism (TP) shards model weights while sequence parallelism (SP) shards...

---

### 3. [Adaptive and Fine-grained Module-wise Expert Pruning for Efficient LoRA-MoE Fine-Tuning](https://arxiv.org/abs/2604.26340)

**Authors**: Weihang Li, Jianchun Liu, Hongli Xu  
**Category**: cs.LG  
**Published**: 2026-04-30  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2604.26340v1  

#### Abstract
LoRA-MoE has emerged as an effective paradigm for parameter-efficient fine-tuning, combining the low training cost of LoRA with the increased adaptation capacity of Mixture-of-Experts (MoE). However, existing LoRA-MoE frameworks typically adopt a fixed and uniform expert configuration across heterog...

---

### 4. [COPUS: Co-adaptive Parallelism and Batch Size Selection in Large Language Model Training](https://arxiv.org/abs/2604.26687)

**Authors**: Akhmed Sakip, Erland Hilman Fuadi, Omar Sayedelahl, Zonghang Li, Jianshu She, Alham Fikri Aji, Steve Liu, Eric Xing, Qirong Ho  
**Category**: cs.DC  
**Published**: 2026-04-30  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2604.26687v1  

#### Abstract
Training large language models requires jointly configuring two interdependent aspects of the system: the global batch size, which governs statistical efficiency, and the 3D parallelism strategy, which governs hardware throughput. Existing approaches make these decisions independently: optimization ...

---

### 5. [EvoSelect: Data-Efficient LLM Evolution for Targeted Task Adaptation](https://arxiv.org/abs/2604.26170)

**Authors**: Ting-Wei Li, Sirui Chen, Jiaru Zou, Yingbing Huang, Tianxin Wei, Jingrui He, Hanghang Tong  
**Category**: cs.CL  
**Published**: 2026-04-30  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.26170v1  

#### Abstract
Adapting large language models (LLMs) to a targeted task efficiently and effectively remains a fundamental challenge. Such adaptation often requires iteratively improving the model toward a targeted task, yet collecting high-quality human-labeled data to support this process is costly and difficult ...

---

### 6. [Efficient, VRAM-Constrained xLM Inference on Clients](https://arxiv.org/abs/2604.26334)

**Authors**: Aditya Ukarande, Deep Shekhar, Marc Blackstein, Ram Rangan  
**Category**: cs.DC  
**Published**: 2026-04-30  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.26334v1  

#### Abstract
To usher in the next round of client AI innovation, there is an urgent need to enable efficient, lossless inference of high-accuracy large language models (LLMs) and vision language models (VLMs), jointly referred to as xLMs, on client systems. To address this, we present pipelined sharding, a novel...

---

### 7. [FaaSMoE: A Serverless Framework for Multi-Tenant Mixture-of-Experts Serving](https://arxiv.org/abs/2604.26881)

**Authors**: Minghe Wang, Trever Schirmer, Mohammadreza Malekabbasi, David Bermbach  
**Category**: cs.DC  
**Published**: 2026-04-30  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.26881v1  

#### Abstract
Mixture-of-Experts (MoE) models offer high capacity with efficient inference cost by activating a small subset of expert models per input. However, deploying MoE models requires all experts to reside in memory, creating a gap between the resource used by activated experts and the provisioned resourc...

---

### 8. [DAK: Direct-Access-Enabled GPU Memory Offloading with Optimal Efficiency for LLM Inference](https://arxiv.org/abs/2604.26074)

**Authors**: Shouxu Lin, Zhiyuan Guo, Jiaxin Lin  
**Category**: cs.DC  
**Published**: 2026-04-30  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.26074v1  

#### Abstract
LLM inference is constrained by GPU memory capacity and bandwidth. Tiered memory architectures mitigate this by allowing the GPU to offload memory to the remote tier. However, existing memory offloading frameworks rely on prefetching data into local GPU HBM. This approach underutilizes system resour...

---

### 9. [FloatSOM: GPU-Accelerated, Distributed, Topology-Flexible Self-Organizing Maps](https://arxiv.org/abs/2604.26555)

**Authors**: Tony Xu, Sarah Klamt, Katherine Turner, Anne Brustle, Felix Marsh-Wakefield, Givanna Putri  
**Category**: cs.DC  
**Published**: 2026-04-30  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.26555v1  

#### Abstract
GPU-accelerated Self-Organizing Map (SOM) implementations are among the most competitive options for large-scale SOM analysis, but growing dataset sizes increasingly challenge their practical use because workloads no longer fit cleanly within device-memory limits. We introduce FloatSOM, a SOM framew...

---

### 10. [Who Trains Matters: Federated Learning under Enrollment and Participation Selection Biases](https://arxiv.org/abs/2604.26604)

**Authors**: Gota Morishita  
**Category**: cs.LG  
**Published**: 2026-04-30  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.26604v1  

#### Abstract
Federated learning (FL) trains a shared model from updates contributed by distributed clients, often implicitly assuming that contributing clients are representative of the target population. In practice, this representativeness assumption can fail at two distinct stages, inducing selection bias. Fi...

---

### 11. [SpecTr-GBV: Multi-Draft Block Verification Accelerating Speculative Decoding](https://arxiv.org/abs/2604.25925)

**Authors**: Yijun Lin, Jinhao Sheng, Qingyue Cai, Feng Zhou  
**Category**: cs.CL  
**Published**: 2026-04-30  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.25925v1  

#### Abstract
Autoregressive language models suffer from high inference latency due to their sequential decoding nature. Speculative decoding (SD) mitigates this by employing a lightweight draft model to propose candidate tokens, which are selectively verified by a larger target model. While existing methods eith...

---

### 12. [HealthNLP_Retrievers at ArchEHR-QA 2026: Cascaded LLM Pipeline for Grounded Clinical Question Answering](https://arxiv.org/abs/2604.26880)

**Authors**: Md Biplob Hosen, Md Alomgeer Hussein, Md Akmol Masud, Omar Faruque, Tera L Reynolds, Lujie Karen Chen  
**Category**: cs.CL  
**Published**: 2026-04-30  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.26880v1  

#### Abstract
Patient portals now give individuals direct access to their electronic health records (EHRs), yet access alone does not ensure patients understand or act on the complex clinical information contained in these records. The ArchEHR-QA 2026 shared task addresses this challenge by focusing on grounded q...

---

### 13. [Unifying Sparse Attention with Hierarchical Memory for Scalable Long-Context LLM Serving](https://arxiv.org/abs/2604.26837)

**Authors**: Zihan Zhao, Baotong Lu, Shengjie Lin, Yizou Chen, Jing Liu, Yanqi Zhang, Ziming Miao, Ming-Chang Yang, Haiying Shen, Qi Chen, Fan Yang  
**Category**: cs.LG  
**Published**: 2026-04-30  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.26837v1  

#### Abstract
Long-context LLM serving is bottlenecked by the cost of attending over ever-growing KV caches. Dynamic sparse attention promises relief by accessing only a small, query-dependent subset of the KV state per decoding step and extending the KV storage to CPU memory. In practice, however, these algorith...

---

### 14. [SplitFT: An Adaptive Federated Split Learning System For LLMs Fine-Tuning](https://arxiv.org/abs/2604.26388)

**Authors**: Yimeng Shan, Zhaorui Zhang, Sheng Di, Yu Liu, Xiaoyi Lu, Benben Liu  
**Category**: cs.DC  
**Published**: 2026-04-30  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.26388v1  

#### Abstract
Federated Split Learning has been identified as an efficient approach to address the computational resource constraints of clients in classical federated learning, while guaranteeing data privacy for distributed model training across data owners. However, it faces some critical challenges when such ...

---

### 15. [Bian Que: An Agentic Framework with Flexible Skill Arrangement for Online System Operations](https://arxiv.org/abs/2604.26805)

**Authors**: Bochao Liu, Zhipeng Qian, Yang Zhao, Xinyuan Jiang, Zihan Liang, Yufei Ma, Junpeng Zhuang, Ben Chen, Shuo Yang, Hongen Wan, Yao Wu, Chenyi Lei, Xiao Liang  
**Category**: cs.AI  
**Published**: 2026-04-30  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.26805v1  

#### Abstract
Operating and maintaining (O&amp;M) large-scale online engine systems (search, recommendation, advertising) demands substantial human effort for release monitoring, alert response, and root cause analysis. While LLM-based agents are a natural fit for these tasks, the deployment bottleneck is not rea...

---

### 16. [Shorthand for Thought: Compressing LLM Reasoning via Entropy-Guided Supertokens](https://arxiv.org/abs/2604.26355)

**Authors**: Zhenyu Zhao, Sander Land, Dan Bikel, Waseem Alshikh  
**Category**: cs.CL  
**Published**: 2026-04-30  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.26355v1  

#### Abstract
Reasoning in Large Language Models incurs significant inference-time compute, yet the token-level information structure of reasoning traces remains underexplored. We observe that reasoning tokens split into two functional types: low-entropy \textit{structural} tokens (recurring phrases that scaffold...

---

### 17. [SAGE: A Strategy-Aware Graph-Enhanced Generation Framework For Online Counseling](https://arxiv.org/abs/2604.26630)

**Authors**: Eliya Naomi Aharon, Meytal Grimland, Avi Segal, Loona Ben Dayan, Inbar Shenfeld, Yossi Levi Belz, Kobi Gal  
**Category**: cs.CL  
**Published**: 2026-04-30  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.26630v1  

#### Abstract
Effective mental health counseling is a complex, theory-driven process requiring the simultaneous integration of psychological frameworks, real-time distress signals, and strategic intervention planning. This level of clinical reasoning is critical for safety and therapeutic effectiveness but is oft...

---

### 18. [Hierarchical adaptive control for real-time dynamic inference at the edge](https://arxiv.org/abs/2604.26470)

**Authors**: Francesco Daghero, Mahyar Tourchi Moghaddam, Mikkel Baun Kj{\ae}rgaard  
**Category**: cs.LG  
**Published**: 2026-04-30  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.26470v1  

#### Abstract
Industrial systems increasingly depend on Machine Learning (ML), and operate on heterogeneous nodes that must satisfy tight latency, energy, and memory constraints. Dynamic ML models, which reconfigure their computational footprint at runtime, promise high energy efficiency and lower average latency...

---

### 19. [PAINT: Partial-Solution Adaptive Interpolated Training for Self-Distilled Reasoners](https://arxiv.org/abs/2604.26573)

**Authors**: Zhiquan Tan, Yinrong Hong  
**Category**: cs.LG  
**Published**: 2026-04-30  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.26573v1  

#### Abstract
Improving large language model (LLM) reasoning requires supervision that is both aligned with the model's own test-time states and informative at the token level. Reinforcement learning with verifiable rewards provides on-policy exploration but offers sparse, high-variance credit; supervised fine-tu...

---

### 20. [Uncertainty-Aware Predictive Safety Filters for Probabilistic Neural Network Dynamics](https://arxiv.org/abs/2604.26836)

**Authors**: Bernd Frauenknecht, Lukas Kesper, Daniel Mayfrank, Henrik Hose, Sebastian Trimpe  
**Category**: cs.LG  
**Published**: 2026-04-30  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.26836v1  

#### Abstract
Predictive safety filters (PSFs) leverage model predictive control to enforce constraint satisfaction during deep reinforcement learning (RL) exploration, yet their reliance on first-principles models or Gaussian processes limits scalability and broader applicability. Meanwhile, model-based RL (MBRL...

---

### 21. [Hierarchical Multi-Persona Induction from User Behavioral Logs: Learning Evidence-Grounded and Truthful Personas](https://arxiv.org/abs/2604.26120)

**Authors**: Nayoung Choi, Haeyu Jeong, Changbong Kim, Hongjun Lim, Jinho D. Choi  
**Category**: cs.AI  
**Published**: 2026-04-30  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.26120v1  

#### Abstract
Behavioral logs provide rich signals for user modeling, but are noisy and interleaved across diverse intents. Recent work uses LLMs to generate interpretable natural-language personas from user logs, yet evaluation often emphasizes downstream utility, providing limited assurance of persona quality i...

---

### 22. [AGEL-Comp: A Neuro-Symbolic Framework for Compositional Generalization in Interactive Agents](https://arxiv.org/abs/2604.26522)

**Authors**: Mahnoor Shahid, Hannes Rothe  
**Category**: cs.AI  
**Published**: 2026-04-30  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.26522v1  

#### Abstract
Large Language Model (LLM)-based agents exhibit systemic failures in compositional generalization, limiting their robustness in interactive environments. This work introduces AGEL-Comp, a neuro-symbolic AI agent architecture designed to address this challenge by grounding actions of the agent. AGEL-...

---

### 23. [FutureWorld: A Live Environment for Training Predictive Agents with Real-World Outcome Rewards](https://arxiv.org/abs/2604.26733)

**Authors**: Zhixin Han, Yanzhi Zhang, Chuyang Wei, Maohang Gao, Xiawei Yue, Kefei Chen, Yu Zhuang, Haoxiang Guan, Jiyan He, Jian Li, Yitong Duan, Yu Shi, Mengting Hu, Shuxin Zheng  
**Category**: cs.AI  
**Published**: 2026-04-30  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.26733v1  

#### Abstract
Live future prediction refers to the task of making predictions about real-world events before they unfold. This task is increasingly studied using large language model-based agent systems, and it is important for building agents that can continually learn from real-world. Just as interactive enviro...

---

### 24. [CogRAG+: Cognitive-Level Guided Diagnosis and Remediation of Memory and Reasoning Deficiencies in Professional Exam QA](https://arxiv.org/abs/2604.25928)

**Authors**: Xudong Wang, Zilong Wang, Zhaoyan Ming  
**Category**: cs.CL  
**Published**: 2026-04-30  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.25928v1  

#### Abstract
Professional domain knowledge underpins human civilization, serving as both the basis for industry entry and the core of complex decision-making and problem-solving. However, existing large language models often suffer from opaque inference processes in which retrieval and reasoning are tightly enta...

---

### 25. [FlowBot: Inducing LLM Workflows with Bilevel Optimization and Textual Gradients](https://arxiv.org/abs/2604.26258)

**Authors**: Hongyeon Yu, Young-Bum Kim, Yoon Kim  
**Category**: cs.CL  
**Published**: 2026-04-30  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.26258v1  

#### Abstract
LLM workflows, which coordinate structured calls to individual LLMs (each augmented with varying instructions and tools) to achieve a particular goal, offer a promising path towards extending the capabilities of LLMs and building powerful systems that can tackle diverse tasks. However, existing appr...

---

### 26. [Select to Think: Unlocking SLM Potential with Local Sufficiency](https://arxiv.org/abs/2604.26940)

**Authors**: Wenxuan Ye, Yangyang Zhang, Xueli An, Georg Carle, Yunpu Ma  
**Category**: cs.CL  
**Published**: 2026-04-30  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.26940v1  

#### Abstract
Small language models (SLMs) offer computational efficiency for scalable deployment, yet they often fall short of the reasoning power exhibited by their larger counterparts (LLMs). To mitigate this gap, current approaches invoke an LLM to generate tokens at points of reasoning divergence, but these ...

---

### 27. [BioGraphletQA: Knowledge-Anchored Generation of Complex QA Datasets](https://arxiv.org/abs/2604.26048)

**Authors**: Richard A. A. Jonker, B\'arbara Maria Ribeiro de Abreu Martins, S\'ergio Matos  
**Category**: cs.CL  
**Published**: 2026-04-30  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.26048v1  

#### Abstract
This paper presents a principled and scalable framework for systematically generating complex Question Answering (QA) data. In the core of this framework is a graphlet-anchored generation process, where small subgraphs from a Knowledge Graph (KG) are used in a structured prompt to control the comple...

---

### 28. [MoRFI: Monotonic Sparse Autoencoder Feature Identification](https://arxiv.org/abs/2604.26866)

**Authors**: Dimitris Dimakopoulos, Shay B. Cohen, Ioannis Konstas  
**Category**: cs.CL  
**Published**: 2026-04-30  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.26866v1  

#### Abstract
Large language models (LLMs) acquire most of their factual knowledge during the pre-training stage, through next token prediction. Subsequent stages of post-training often introduce new facts outwith the parametric knowledge, giving rise to hallucinations. While it has been demonstrated that supervi...

---

### 29. [MPI Malleability Validation under Replayed Real-World HPC Conditions](https://arxiv.org/abs/2604.26576)

**Authors**: S. Iserte, M. Madon, G. Da, J. Pierson, A. J. Pe\~na  
**Category**: cs.DC  
**Published**: 2026-04-30  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.26576v1  

#### Abstract
Dynamic Resource Management (DRM) techniques can be leveraged to maximize throughput and resource utilization in computational clusters. Although DRM has been extensively studied through analytical workloads and simulations, skepticism persists among end administrators and users regarding their feas...

---

### 30. [A Test Taxonomy and Continuous Integration Ecosystem for Dynamic Resource Management in HPC](https://arxiv.org/abs/2604.26824)

**Authors**: Petter Sand{\aa}s, \'I\~nigo Ar\'ejula-A\'isa, Sergio Iserte, Antonio J. Pe\~na  
**Category**: cs.DC  
**Published**: 2026-04-30  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.26824v1  

#### Abstract
High-performance computing (HPC) systems are increasingly exploring dynamic resource management and malleable MPI applications to better adapt to heterogeneous architectures, fluctuating workloads, and energy constraints. However, the correctness of the libraries that support these techniques is oft...

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
