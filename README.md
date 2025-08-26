# Awesome Video DiT Distillation Papers 🎬🚀

A curated list of papers on Video Diffusion Transformer (DiT) distillation, focusing on reducing inference steps from N-step to 1-step generation.

## Table of Contents
- [One-Step Inference](#one-step-inference)
- [Few-Step Inference (2-4 steps)](#few-step-inference-2-4-steps)
- [Mobile & Real-Time Generation](#mobile--real-time-generation)
- [Quantization & Optimization](#quantization--optimization)
- [Open Source Implementations](#open-source-implementations)
- [Industry Applications](#industry-applications)

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

### 2024

- **Video-BLADE: Block-Sparse Attention Meets Step Distillation for Efficient Video Generation** (Aug 2024)  
  [[ArXiv](https://arxiv.org/html/2508.10774)]  
  *Proposes BLADE framework with Adaptive Block-Sparse Attention (ASA) and sparsity-aware step distillation.*

- **Multi-Student Diffusion Distillation for Better One-step Generators** (2025)  
  [[NVIDIA Research](https://research.nvidia.com/publication/2025-03_multi-student-diffusion-distillation-better-one-step-generators)]  
  *Using 4 same-sized students, achieves FID 1.20 on ImageNet-64×64 and 8.20 on zero-shot COCO2014.*

---

## Mobile & Real-Time Generation

### 2024

- **Taming Diffusion Transformer for Real-Time Mobile Video Generation**  
  [[ArXiv](https://arxiv.org/html/2507.13343)]  
  *Achieves 10+ FPS on iPhone 16 Pro Max using only 4 denoising steps (4 seconds for 49 frames). Combines high-compression VAE, latency-aware pruning, and adversarial step distillation.*

- **Optimizing Transformer-Based Diffusion Models for Video Generation with NVIDIA TensorRT**  
  [[NVIDIA Blog](https://developer.nvidia.com/blog/optimizing-transformer-based-diffusion-models-for-video-generation-with-nvidia-tensorrt/)]  
  *Technical guide for optimizing DiT models for efficient inference.*

---

## Quantization & Optimization

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

### Architecture Innovations
- **Adaptive Block-Sparse Attention (ASA)**: Dynamic, content-aware attention
- **Attention Tile**: Efficient attention mechanisms for video transformers
- **High-Compression VAE**: Reduces computational overhead

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

*Last Updated: August 2025*  
*Total Papers: 15+ cutting-edge research works*