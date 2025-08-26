# Awesome Video DiT Distillation Papers 🎬🚀

A curated list of papers on Video Diffusion Transformer (DiT) distillation, focusing on reducing inference steps from N-step to 1-step generation.

## Table of Contents
- [Technical Challenges & Insights](#technical-challenges--insights)
- [One-Step Inference](#one-step-inference)
- [Few-Step Inference (2-4 steps)](#few-step-inference-2-4-steps)
- [Mobile & Real-Time Generation](#mobile--real-time-generation)
- [Quantization & Optimization](#quantization--optimization)
- [Open Source Implementations](#open-source-implementations)
- [Industry Applications](#industry-applications)
- [Video Restoration & Enhancement](#video-restoration--enhancement)
- [Dataset Distillation](#dataset-distillation)
- [Consistency Models & Rectified Flow](#consistency-models--rectified-flow)
- [Autonomous Driving Applications](#autonomous-driving-applications)
- [Acceleration Methods](#acceleration-methods)

---

## Technical Challenges & Insights

### Computational Complexity
Video generation is computationally expensive, taking ~10 minutes for 129 frames on H100 GPU with quadratic attention scaling.

**Solution Approaches**:
- **Step Reduction**: Single-step generation, few-step distillation (4-8 steps), consistency models
- **Attention Optimization**: Sparse attention patterns, attention tiling, asymmetric reduction and restoration ([AsymRnR](#acceleration-methods))
- **Architectural Efficiency**: Compressed representations, quantization ([S²Q-VDiT](#quantization--optimization)), model pruning and acceleration

### Temporal Consistency
Maintaining coherent motion and identity across video frames, especially in long sequences, remains challenging.

**Solution Approaches**:
- **Enhanced Attention**: View-inflated attention ([DiVE](#autonomous-driving-applications)), consistent self-attention ([StoryDiffusion](#consistency-models--rectified-flow)), cross-frame attention
- **Latent Methods**: Latent shifting (Mobius), motion-compensated processing
- **Training Strategies**: Motion consistency models ([MCM](#few-step-inference-2-4-steps)), cycle consistency learning
- **Frequency Domain**: Time-frequency analysis (TiARA), spectral attention for temporal modeling

### Quality-Speed Trade-off
Reducing inference steps for speed often leads to significant quality degradation and detail loss.

**Solution Approaches**:
- **Advanced Distillation**: Distribution matching (DMD), trajectory distribution matching (TDM), score implicit matching (SIM)
- **Adaptive Mechanisms**: Block-sparse attention ([Video-BLADE](#few-step-inference-2-4-steps)), multiscale rendering ([LTX-Video](#mobile--real-time-generation))
- **Multi-Step Strategies**: Consistency models, rectified flow matching, few-step generation

### Training Instability
Distillation training suffers from mode collapse, adversarial instability, and memory constraints.

**Solution Approaches**:
- **Stabilization**: Spectral normalization, gradient penalty, weak-to-strong distillation (W2SVD)
- **Memory Optimization**: LoRA fine-tuning, gradient checkpointing, mixed precision training
- **Training Strategies**: Denoising diffusion GANs, mini-batch discrimination, data-free methods

### Key Distillation Techniques
- **Distribution Matching Distillation (DMD)**: Minimizes KL divergence
- **Adversarial Diffusion Distillation (ADD)**: Combines score distillation with adversarial loss for 1-4 step generation
- **Score Implicit Matching (SIM)**: Maintains sample generation ability
- **Multi-Step Consistency Distillation (MCD)**: Generates student with fewer steps
- **Latent Consistency Models (LCM)**: Enable 2-4 step generation in latent space
- **Trajectory Distribution Matching (TDM)**: Unifies distribution and trajectory matching
- **Concept Distillation**: Transfers T2V model understanding using synthesized training data (Vivid-VR)
- **Weak-to-Strong Video Distillation (W2SVD)**: Mitigates training memory issues
- **Salient Data & Sparse Token Distillation**: Focuses on influential tokens for quantization
- **Motion-Appearance Disentangled Distillation**: Separates motion and appearance for efficient video generation
- **Multi-Control Auxiliary Branch Distillation (MAD)**: Eliminates CFG selection for controllable generation
- **Rectified Flow Matching (RFM)**: Builds straight paths from noise to samples
- **Latent Adversarial Diffusion Distillation**: FLUX-style single-step generation
- **Progressive Training Distillation**: Multi-resolution training for efficient video generation (LTX-Video)

### Architecture Innovations
- **Adaptive Block-Sparse Attention (ASA)**: Dynamic, content-aware attention
- **Attention Tile**: Efficient attention mechanisms for video transformers
- **High-Compression VAE**: Reduces computational overhead
- **Expert Transformer**: Expert adaptive LayerNorm for text-video fusion
- **3D Variational Autoencoder**: Compresses videos along spatial and temporal dimensions
- **Chain of Diffusion Experts**: Multiple specialized models for long video generation
- **AsymmDiT (Asymmetric DiT)**: Novel architecture with asymmetric spatial/temporal processing
- **AsymmVAE**: Causal video compression with 8x8 spatial and 6x temporal compression
- **Multiscale Rendering**: Progressive detail generation for speed and quality
- **Unified Full Attention**: Single attention mechanism for superior performance
- **View-Inflated Attention**: Parameter-free cross-view consistency for multi-view generation
- **Asymmetric Reduction and Restoration**: Training-free token reduction based on redundancy

---

## One-Step Inference

### 2025

- **Diffusion Adversarial Post-Training for One-Step Video Generation** (Jan 2025)  
  [[ArXiv](https://arxiv.org/html/2501.08316v1)]  
  *Achieves single-step high-resolution video generation using pre-trained DiT as initialization. Demonstrates ability to surpass teacher model in realism and fine details.*

- **MagicDistillation: Weak-to-Strong Video Distillation for Large-Scale Portrait Few-Step Synthesis** (Mar 2025)  
  [[ArXiv](https://arxiv.org/html/2503.13319v1)]  
  *Achieves 1/4-step video synthesis, surpassing standard Euler, LCM, DMD and 28-step sampling in FID/FVD and VBench metrics.*

- **VideoScene: Distilling Video Diffusion Model to Generate 3D Scenes in One Step** (Apr 2025)  
  [[ArXiv](https://arxiv.org/html/2504.01956v1)]  
  *Focuses on recovering 3D scenes from sparse views using single-step generation.*

- **FLUX.1-schnell: Adversarial Diffusion Distillation for Single-Step Generation** (2025)  
  [[Hugging Face](https://huggingface.co/black-forest-labs/FLUX.1-schnell)]  
  *Trained using latent adversarial diffusion distillation, can generate high-quality images in 1-4 steps. Extended to video applications via APT method.*

### 2024

- **One-Step Diffusion with Distribution Matching Distillation** (CVPR 2024)  
  [[CVPR](https://openaccess.thecvf.com/content/CVPR2024/papers/Yin_One-step_Diffusion_with_Distribution_Matching_Distillation_CVPR_2024_paper.pdf)] [[ResearchGate](https://www.researchgate.net/publication/384215207_One-Step_Diffusion_with_Distribution_Matching_Distillation)]  
  *Minimizes KL divergence between generator output and true data distribution for single-step inference.*

- **One-Step Diffusion Distillation through Score Implicit Matching** (NeurIPS 2024)  
  [[NeurIPS](https://neurips.cc/virtual/2024/poster/93608)]  
  *Achieves aesthetic score of 6.42 for text-to-image generation, outperforming SDXL-TURBO (5.33), SDXL-LIGHTNING (5.34) and HYPER-SDXL (5.85).*

---

## Few-Step Inference (2-4 steps)

### 2025

- **From Slow Bidirectional to Fast Causal Video Generators** (Dec 2024)  
  [[ArXiv](https://arxiv.org/html/2412.07772v1)]  
  *Extends Distribution Matching Distillation (DMD) to videos, distilling 50-step diffusion model into a 4-step generator.*

- **Efficient-vDiT: Efficient Video Diffusion Transformers with Attention Tile** (Feb 2025)  
  [[ArXiv](https://arxiv.org/html/2502.06155)]  
  *Adopts multi-step consistency distillation (MCD) technique to generate student model with fewer sampling steps than teacher.*

- **GVD: Guiding Video Diffusion Model for Scalable Video Distillation** (Jul 2025)  
  [[ArXiv](https://arxiv.org/abs/2507.22360)]  
  *First diffusion-based video distillation method. Achieves 78.29% performance using only 1.98% of frames in MiniUCF.*

- **Training-free Long Video Generation with Chain of Diffusion Model Experts** (Aug 2025)  
  [[ArXiv](https://arxiv.org/abs/2408.13423)] [[Hugging Face](https://huggingface.co/papers/2408.13423)]  
  *Novel approach for generating long videos using multiple diffusion experts without additional training.*

- **Motion Consistency Model: Accelerating Video Diffusion** (Jun 2025)  
  [[ArXiv](https://arxiv.org/html/2406.06890v1)]  
  *Single-stage video diffusion distillation method with disentangled motion-appearance distillation. Enables few-step sampling and achieves SOTA video distillation performance.*

### 2024-2025 (Updated)

- **Video-BLADE: Block-Sparse Attention Meets Step Distillation for Efficient Video Generation** (Aug 2025)  
  [[ArXiv](https://arxiv.org/html/2508.10774)]  
  *BLADE framework with Adaptive Block-Sparse Attention (ASA) and Trajectory Distribution Matching (TDM) distillation process. Achieves significant inference acceleration without sacrificing generation quality.*

- **Multi-Student Diffusion Distillation for Better One-step Generators** (2025)  
  [[NVIDIA Research](https://research.nvidia.com/publication/2025-03_multi-student-diffusion-distillation-better-one-step-generators)]  
  *Using 4 same-sized students, achieves FID 1.20 on ImageNet-64×64 and 8.20 on zero-shot COCO2014.*

---

## Mobile & Real-Time Generation

### 2025

- **Taming Diffusion Transformer for Real-Time Mobile Video Generation** (Jul 2025)  
  [[ArXiv](https://arxiv.org/html/2507.13343)]  
  *Achieves 10+ FPS on iPhone 16 Pro Max using only 4 denoising steps (4 seconds for 49 frames). Uses distillation-guided, sensitivity-aware pruning with new discriminator design for DiTs.*

- **LTX-Video: Real-Time Video Generation 30x Faster** (2025)  
  [[GitHub](https://github.com/Lightricks/LTX-Video)] [[Hugging Face](https://huggingface.co/Lightricks/LTX-Video)]  
  *13B-parameter DiT model with multiscale rendering. Distilled version runs in 4-8 steps (9.5 seconds generation) on consumer GPUs like RTX 4090. Achieves 30x speed improvement over comparable models.*

- **Optimizing Transformer-Based Diffusion Models for Video Generation with NVIDIA TensorRT**  
  [[NVIDIA Blog](https://developer.nvidia.com/blog/optimizing-transformer-based-diffusion-models-for-video-generation-with-nvidia-tensorrt/)]  
  *Technical guide for optimizing DiT models for efficient inference.*

---

## Quantization & Optimization

### 2025

- **S²Q-VDiT: Accurate Quantized Video Diffusion Transformer with Salient Data and Sparse Token Distillation** (Aug 2025)  
  [[ArXiv](https://arxiv.org/abs/2508.04016)]  
  *Post-training quantization framework using Hessian-aware Salient Data Selection and Attention-guided Sparse Token Distillation. Achieves 3.9× model compression and 1.3× inference acceleration under W4A6 quantization.*

### 2024

- **Q-VDiT: Towards Accurate Quantization and Distillation of Video-Generation Diffusion Transformers**  
  [[ArXiv](https://arxiv.org/html/2505.22167)]  
  *Addresses unique challenges of video DiT quantization, as video generation requires modeling temporal dimensions with significantly increased information density.*

- **Time-Aware One Step Diffusion Network for Real-World Image Super-Resolution** (Aug 2024)  
  [[ArXiv](https://arxiv.org/abs/2508.16557)]  
  *Focuses on single-step diffusion for image enhancement applications.*

---

## Open Source Implementations

### 2025

- **Open-Sora 2.0: Training a Commercial-Level Video Generation Model in $200k** (Mar 2025)  
  [[ArXiv](https://arxiv.org/html/2503.09642v1)] [[GitHub](https://github.com/hpcaitech/Open-Sora)]  
  *Commercial-level model trained for only $200k. Uses Flux initialization and latent space distillation. Reduces gap with OpenAI Sora from 4.52% to 0.69% on VBench.*

- **CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer** (Aug 2024)  
  [[ArXiv](https://arxiv.org/abs/2408.06072)] [[GitHub](https://github.com/zai-org/CogVideo)] [[Hugging Face](https://huggingface.co/papers/2408.06072)]  
  *Large-scale T2V model generating 10-second 768×1360 videos at 16fps. Features 3D VAE compression and expert transformer architecture. Used in ADM distillation experiments and concept distillation research.*

- **Diffusers Library: State of Open Video Generation Models** (Jan 2025)  
  [[Hugging Face Blog](https://huggingface.co/blog/video_gen)]  
  *Comprehensive overview of video generation models in 🤗 Diffusers library. Plans for 2025 include Control LoRAs, Distillation Algorithms, ControlNets, and Adapters.*

- **Mochi 1: State-of-the-Art Open-Source Video Generation** (Oct 2024)  
  [[GitHub](https://github.com/genmoai/mochi)] [[Hugging Face](https://huggingface.co/genmo/mochi-1-preview)]  
  *10B parameter model with novel AsymmDiT architecture. Features 128x spatial compression, 6x temporal compression, and 30fps generation. Uses single T5-XXL language model and requires ~60GB VRAM.*

- **HunyuanVideo: Large-Scale Open-Source Video Generation** (Dec 2024)  
  [[ArXiv](https://arxiv.org/html/2412.03603v1)] [[GitHub](https://github.com/Tencent-Hunyuan/HunyuanVideo)] [[Hugging Face](https://huggingface.co/tencent/HunyuanVideo)]  
  *13B+ parameter model from Tencent with unified Full Attention mechanism. Features model distillation to solve inference limitations and uses MLLM as text encoder. Includes FP8 model weights to save GPU memory.*

### 2024

- **Diffusion Transformers for Image and Video Generation** (CVPR 2024)  
  [[CVPR](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_GenTron_Diffusion_Transformers_for_Image_and_Video_Generation_CVPR_2024_paper.pdf)]  
  *GenTron: Comprehensive framework for image and video generation using diffusion transformers.*

---

## Industry Applications

### 2024

- **Adobe Firefly Video Generation**  
  *Achieved time to market in under 4 months, generating 70+ million images in first month, powering 20+ billion assets.*

- **OpenAI Sora** (Dec 2024)  
  [[Official](https://openai.com/sora/)] [[System Card](https://openai.com/index/sora-system-card/)]  
  *Released Sora Turbo - significantly faster than February preview model. Available for ChatGPT Plus/Pro users.*

### 2025 Industry Developments

- **FLUX Ecosystem Expansion**  
  [[Hugging Face](https://huggingface.co/black-forest-labs)] [[GitHub Issues](https://github.com/black-forest-labs/flux/issues/15)]  
  *FLUX.1-schnell demonstrates 1-4 step generation via latent adversarial diffusion distillation. Community discussions on model distillation techniques and acceleration methods.*

---

## Video Restoration & Enhancement

### 2025

- **Vivid-VR: Distilling Concepts from Text-to-Video Diffusion Transformer for Photorealistic Video Restoration** (Aug 2025)  
  [[ArXiv](https://arxiv.org/html/2508.14483)]  
  *DiT-based generative video restoration using CogVideoX1.5-5B foundation model. Introduces concept distillation training strategy to preserve texture and temporal quality.*

---

## Dataset Distillation

### 2025

- **The Evolution of Dataset Distillation: Toward Scalable and Generalizable Solutions** (Feb 2025)  
  [[ArXiv](https://arxiv.org/html/2502.05673v3)]  
  *Comprehensive survey covering dataset distillation advances including applications to video diffusion models.*

- **Seedance 1.0: Exploring the Boundaries of Video Generation Models** (Jun 2025)  
  [[Hugging Face](https://huggingface.co/papers/2506.09113)]  
  *Explores video generation model capabilities and boundaries, with implications for distillation approaches.*

---

## Consistency Models & Rectified Flow

### 2025

- **VideoLCM: Video Latent Consistency Model** (Dec 2023/2025 Updates)  
  [[ArXiv](https://arxiv.org/abs/2312.09109)] [[OpenReview](https://openreview.net/forum?id=TpshckO3g4)] [[Hugging Face](https://huggingface.co/papers/2312.09109)]  
  *Extends latent consistency models to video generation. Enables high-fidelity video synthesis in 4-6 steps (vs 50 for DDIM). Can produce visually satisfactory results with even 1 step for compositional video synthesis.*

- **OpenAI sCM: Simplified Consistency Models** (2025)  
  [[OpenAI Blog](https://openai.com/index/simplifying-stabilizing-and-scaling-continuous-time-consistency-models/)]  
  *Achieves comparable sample quality to diffusion models using only 2 sampling steps, resulting in ~50x wall-clock speedup through simplified theoretical formulation.*

- **Rectified Flow for Video Generation** (2025)  
  [[ArXiv Multiple](https://arxiv.org/abs/2403.03206)]  
  *FiVE benchmark for evaluating rectified flow models in video editing. V2SFlow uses Rectified Flow Matching for video-to-speech generation. RF-Edit framework for video editing with structural preservation.*

- **Frieren: Video-to-Audio with Rectified Flow** (Jun 2025)  
  [[ArXiv](https://arxiv.org/abs/2406.00320)]  
  *V2A model based on rectified flow matching, regresses conditional transport vector fields from noise to spectrogram latent with straight paths. Outperforms autoregressive and score-based models.*

- **StoryDiffusion: Consistent Self-Attention for Long-Range Image and Video Generation** (NeurIPS 2024)  
  [[ArXiv](https://arxiv.org/abs/2405.01434)] [[GitHub](https://github.com/HVision-NKU/StoryDiffusion)] [[Project](https://storydiffusion.github.io/)]  
  *Proposes Consistent Self-Attention that significantly boosts consistency between generated images and extends to long-range video generation. Includes Semantic Motion Predictor for smooth transitions and consistent subjects.*

---

## Autonomous Driving Applications

### 2025

- **DiVE: Efficient Multi-View Driving Scenes Generation** (Apr 2025)  
  [[ArXiv](https://arxiv.org/html/2504.19614)] [[OpenReview](https://openreview.net/forum?id=cvDB1QAYUu)]  
  *First DiT-based framework for multi-view driving scenario videos. Features Multi-Control Auxiliary Branch Distillation (MAD) eliminating CFG selection. Achieves 2.62x speedup with Resolution Progressive Sampling.*

---

## Acceleration Methods

### 2025

- **AsymRnR: Asymmetric Reduction and Restoration for Video DiT Acceleration** (Dec 2024/ICML 2025)  
  [[ArXiv](https://arxiv.org/abs/2412.11706)] [[ICML](https://icml.cc/virtual/2025/poster/46432)]  
  *Training-free and model-agnostic method for accelerating video DiTs. Asymmetrically reduces redundant tokens based on their redundancy across model blocks, denoising steps, and feature types. Reduces running time by nearly 1/3.*

- **Adversarial Distribution Matching for Video Synthesis** (Jul 2025)  
  [[ArXiv](https://arxiv.org/html/2507.18569)]  
  *Applies multi-step ADM distillation on CogVideoX. Demonstrates 8-step ADM distillation for efficient video generation with maintained quality.*

---

## Citation

```bibtex
@misc{awesome-video-dit-distillation-2025,
  title={Awesome Video DiT Distillation Papers},
  author={Community Curated},
  year={2025},
  note={A curated list of Video Diffusion Transformer distillation papers}
}
```

---

## Contributing

Feel free to contribute by:
1. Adding new papers with proper formatting
2. Updating existing entries with new information
3. Fixing any errors or broken links
4. Suggesting new categories

## License

This compilation is available under Creative Commons license. Individual papers retain their original copyrights.

---

*Last Updated: August 27, 2025*  
*Total Papers: 35+ cutting-edge research works*