# Awesome Video DiT Distillation Papers 🎬🚀

A curated list of papers on Video Diffusion Transformer (DiT) distillation, focusing on reducing inference steps from N-step to 1-step generation.

## Table of Contents
- [One-Step Inference](#one-step-inference)
- [Few-Step Inference (2-4 steps)](#few-step-inference-2-4-steps)
- [Mobile & Real-Time Generation](#mobile--real-time-generation)
- [Quantization & Optimization](#quantization--optimization)
- [Open Source Implementations](#open-source-implementations)
- [Industry Applications](#industry-applications)
- [Video Restoration & Enhancement](#video-restoration--enhancement)
- [Dataset Distillation](#dataset-distillation)

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

- **CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer** (Aug 2025)  
  [[ArXiv](https://arxiv.org/abs/2408.06072)] [[GitHub](https://github.com/zai-org/CogVideo)] [[Hugging Face](https://huggingface.co/papers/2408.06072)]  
  *Large-scale T2V model generating 10-second 768×1360 videos at 16fps. Features 3D VAE compression and expert transformer architecture. Used in ADM distillation experiments and concept distillation research.*

- **Diffusers Library: State of Open Video Generation Models** (Jan 2025)  
  [[Hugging Face Blog](https://huggingface.co/blog/video_gen)]  
  *Comprehensive overview of video generation models in 🤗 Diffusers library. Plans for 2025 include Control LoRAs, Distillation Algorithms, ControlNets, and Adapters.*

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

## Technical Challenges & Insights

### Computational Complexity
- **Standard Performance**: 129-frame video generation takes ~10 minutes on single H100 GPU
- **Training Cost**: Large-scale VDMs require ~10 minutes for 28-step video on H100 GPU
- **Memory Requirements**: H200 GPUs with 141GB memory enable more effective data parallelism

### Key Distillation Techniques
1. **Distribution Matching Distillation (DMD)**: Minimizes KL divergence
2. **Adversarial Step Distillation**: Combines adversarial training with step reduction
3. **Score Implicit Matching (SIM)**: Maintains sample generation ability
4. **Multi-Step Consistency Distillation (MCD)**: Generates student with fewer steps
5. **Latent Space Distillation**: Aligns latents with foundation models
6. **Trajectory Distribution Matching (TDM)**: Unifies distribution and trajectory matching
7. **Concept Distillation**: Transfers conceptual understanding from T2V models
8. **Weak-to-Strong Video Distillation (W2SVD)**: Mitigates training memory issues
9. **Salient Data & Sparse Token Distillation**: Focuses on influential tokens for quantization

### Architecture Innovations
- **Adaptive Block-Sparse Attention (ASA)**: Dynamic, content-aware attention
- **Attention Tile**: Efficient attention mechanisms for video transformers
- **High-Compression VAE**: Reduces computational overhead
- **Expert Transformer**: Expert adaptive LayerNorm for text-video fusion
- **3D Variational Autoencoder**: Compresses videos along spatial and temporal dimensions
- **Chain of Diffusion Experts**: Multiple specialized models for long video generation

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
*Total Papers: 25+ cutting-edge research works*