---
id: 16_visao_computacional
title: "Módulo 16 — Visão Computacional"
sidebar_position: 16
---

# Módulo 16 — Visão Computacional

> **Objetivo**: dominar visão computacional moderna — das CNNs clássicas aos Vision Transformers, foundation models visuais (DINOv2, CLIP, SAM), e os modelos visão-linguagem que conectam com o módulo [18](18_multimodal.md).
>
> **Pré-requisitos**: Módulo [05](05_deep_learning.md) (Deep Learning), Módulo [07](07_transformers.md) (Transformers).
>
> **Tempo de referência**: 4–6 semanas.

---

## Por que isso importa

Mesmo em uma trilha centrada em LLMs, **VC é parte do tronco**: modelos multimodais (GPT-4V, Claude, Gemini, LLaVA) integram visão; agentes que usam computer-use precisam parsear screenshots; OCR/extração de PDFs de documentos é essencial em RAG profissional. E muitos conceitos (embeddings, transferência, transformers) ressoam.

---

## 16.1 Fundamentos de imagens digitais

### Conceitos
- Pixels, canais (RGB, RGBA, grayscale, HSV, LAB).
- Amostragem, quantização.
- Convolução clássica (filtros: Sobel, Gaussian, Laplacian).
- Operações morfológicas (erosão, dilatação).
- Detecção de bordas (Canny), keypoints (SIFT, ORB) — **clássicos pré-DL**, ainda úteis.

### Ferramentas
- 🛠 **OpenCV** (Python e bindings TS limitados via opencv.js / opencv4nodejs).
- 🛠 **Pillow / Pillow-SIMD**.
- 🛠 **scikit-image**.

### Por que estudar o clássico
Pipelines reais combinam DL com pré-/pós-processamento clássico (cropping, augmentation, calibração de cor). Saber clássico dobra produtividade.

---

## 16.2 CNNs revisitadas (com profundidade)

Revisar mod. [05](05_deep_learning.md), agora aprofundado.

### Arquiteturas
- **LeNet-5** (didática).
- **AlexNet** (virada de chave 2012).
- **VGG** (simplicidade).
- **GoogLeNet / Inception**.
- **ResNet** (skip connections).
- **DenseNet**.
- **MobileNet / EfficientNet** (eficiência mobile).
- **ConvNeXt** — ResNet "modernizada" com truques de Transformer. 📄 https://arxiv.org/abs/2201.03545

### Truques de treinamento
- **Data augmentation**: random crop, flip, color jitter, RandAugment, AutoAugment.
- **Mixup**, **CutMix**, **CutOut**.
- **Label smoothing**.
- **EMA** (exponential moving average de pesos).
- **Cosine learning rate**.
- **Stochastic depth**.

### Referências
- 📄 **A ConvNet for the 2020s (ConvNeXt)** — Liu et al. (2022). https://arxiv.org/abs/2201.03545
- 📄 **EfficientNet** — Tan & Le (2019). https://arxiv.org/abs/1905.11946
- 🎓 **Stanford CS231N — Convolutional Neural Networks for Visual Recognition**. http://cs231n.stanford.edu/

---

## 16.3 Vision Transformer (ViT) e variantes

### ViT
Aplica Transformer puro a patches de imagem.
- Imagem 224×224 → 196 patches 16×16 → tokens (com pos embedding) → Transformer encoder.

📄 **An Image is Worth 16×16 Words (ViT)** — Dosovitskiy et al. (2020). https://arxiv.org/abs/2010.11929

### Variantes importantes
- **DeiT** — ViT com training tricks, sem precisar de JFT-300M. 📄 https://arxiv.org/abs/2012.12877
- **Swin Transformer** — hierarchical, sliding window. 📄 https://arxiv.org/abs/2103.14030
- **MLP-Mixer** (alternativa sem attention). 📄 https://arxiv.org/abs/2105.01601
- **MaxViT**, **CoAtNet** — híbridos CNN+Transformer.

### Quando preferir ViT vs CNN
- **Datasets grandes**: ViT escala melhor.
- **Datasets pequenos**: CNN com inductive bias pode ganhar.
- **Resolução variável**: ViT lida bem.
- **Inferência em edge**: CNN mobile (MobileNet, EfficientNet) ainda dominam.

---

## 16.4 Self-Supervised Learning em VC

Recapitulando do mod. [04](04_ml_moderno.md), agora com modelos de ponta.

### Contrastive
- **SimCLR**, **MoCo v3**.

### Cluster-based / Self-distillation
- **DINO** — self-distillation com no labels. 📄 https://arxiv.org/abs/2104.14294
- **DINOv2** (Meta, 2023) — backbone universal, embeddings de alta qualidade. 📄 https://arxiv.org/abs/2304.07193

### Masked Image Modeling
- **MAE (Masked Autoencoders)** — He et al. (2021). 📄 https://arxiv.org/abs/2111.06377
- **BEiT**, **iBOT**.

### Por que DINOv2 importa
É o **"BERT da visão"** atual: backbone pré-treinado que serve para classificação, segmentação, retrieval, depth estimation, sem fine-tuning ou com pouco. Use-o.

---

## 16.5 Tarefas clássicas de VC (e o estado da arte 2024–2025)

### Classificação
- ImageNet — benchmark histórico.
- Atual: ConvNeXt v2, ViT-22B, EVA, InternImage.

### Detecção de objetos
- **R-CNN, Fast/Faster R-CNN** (clássicos two-stage).
- **YOLO** (one-stage, real-time) — versões evoluem rapidamente: YOLOv8, YOLOv9, YOLOv10, YOLOv11. https://github.com/ultralytics/ultralytics
- **DETR** — detecção como set-prediction com Transformer. 📄 https://arxiv.org/abs/2005.12872
- **Grounding DINO** — detecção open-vocabulary com texto. 📄 https://arxiv.org/abs/2303.05499

### Segmentação
- **U-Net** (clássica, ainda forte em medical imaging). 📄 https://arxiv.org/abs/1505.04597
- **Mask R-CNN**.
- **DeepLabV3+**.
- **SAM (Segment Anything Model)** — Meta, foundation model para segmentação. 📄 https://arxiv.org/abs/2304.02643
- **SAM 2** — vídeo. 📄 https://arxiv.org/abs/2408.00714
- **Mask2Former** — unifica panoptic, instance, semantic.

### Estimativa de profundidade (depth)
- **Depth Anything**, **Depth Anything v2**. 📄 https://arxiv.org/abs/2406.09414
- **MiDaS**.

### Pose estimation
- **OpenPose**, **HRNet**, **MMPose**.

### OCR
- **Tesseract** (clássico).
- **PaddleOCR**, **EasyOCR**, **TrOCR**.
- Modelos VLM modernos (LLaVA, Qwen-VL) frequentemente fazem OCR razoável end-to-end.
- **Marker**, **Docling** — extração de PDF estrutural com VC.

### Referências unificadoras
- 🎓 **Stanford CS231N**. http://cs231n.stanford.edu/
- 🎓 **MIT 6.819/6.869 Advances in Computer Vision**. https://www.mit.edu/~vondrick/courses.html
- 🛠 **MMDetection, MMSegmentation** (OpenMMLab). https://github.com/open-mmlab

---

## 16.6 Modelos visão-linguagem (preview do mod. [18](18_multimodal.md))

- **CLIP** — alinhamento visão-texto via contrastive learning. 📄 https://arxiv.org/abs/2103.00020
- **SigLIP** — versão melhorada do CLIP com sigmoid loss. 📄 https://arxiv.org/abs/2303.15343
- **BLIP / BLIP-2** — bootstrapping captioning. 📄 https://arxiv.org/abs/2301.12597
- **Florence-2** — Microsoft, geral-purpose. 📄 https://arxiv.org/abs/2311.06242
- **OpenCLIP** (LAION) — reproduções abertas.

### Aplicações
- Zero-shot classification.
- Image-text retrieval.
- Embeddings universais para imagens.
- Inicialização de VLMs (mod. [18](18_multimodal.md)).

---

## 16.7 Modelos generativos visuais (preview do mod. [19](19_topicos_avancados.md))

- **GANs** (StyleGAN family).
- **Diffusion Models**: DDPM, Stable Diffusion, SDXL, FLUX.1. Detalhe profundo em mod. [19](19_topicos_avancados.md).
- **Autoregressive image** (Parti, MaskGIT).
- **Video generation**: Sora-style, Mochi, CogVideoX, LTX-Video.

---

## 16.8 Vision foundation models (2024–2025)

- **DINOv2** — backbone universal.
- **SAM 2** — segmentação universal.
- **Depth Anything v2** — depth universal.
- **Grounding DINO 1.5** — detecção open-vocab.
- **CoTracker, TAPIR** — point tracking.

### Ecossistema HF
- 🛠 **timm** (PyTorch Image Models) — Ross Wightman, repositório de modelos. https://github.com/huggingface/pytorch-image-models
- 🛠 **transformers** suporta a maioria dos modelos VLM/VC.

---

## 16.9 Visão em produção

### TS / browser
- **transformers.js** — modelos VC em ONNX no browser.
- **WebGL/WebGPU** acceleration.
- **MediaPipe** — modelos visuais Google rodando local.

### Mobile
- **CoreML** (iOS), **TensorFlow Lite** / **MediaPipe** (Android).
- Modelos quantizados em FP16/INT8.

### Servidor
- **Triton Inference Server** (NVIDIA).
- **ONNX Runtime**.
- **TorchServe**.

---

## 🧪 Projetos práticos

### Projeto 16.1 — Pipeline clássico vs CNN vs ViT
- Mesma tarefa: classificação de plantas (ou outro dataset 5–10 classes).
- Pipeline 1: features clássicas (SIFT/HOG) + SVM.
- Pipeline 2: CNN (transfer learning de ResNet50).
- Pipeline 3: fine-tuning de ViT pequeno.
- Compare accuracy, tempo de treino, tamanho.

### Projeto 16.2 — Fine-tune YOLO em domínio próprio
- Coletar/anotar 200–500 imagens (ou usar dataset público).
- Use Ultralytics YOLOv8/v11.
- Treine, avalie em mAP, exporte ONNX.

### Projeto 16.3 — Embeddings com DINOv2
- Backbone DINOv2 (sem fine-tune).
- Tarefa downstream: image retrieval em coleção pessoal.
- Compare embeddings DINOv2 vs CLIP.
- Frontend simples (Streamlit ou Next.js).

### Projeto 16.4 — Segmentação com SAM 2
- Use SAM 2 para segmentar objetos em imagens próprias.
- Implemente "click to segment" UI.
- Bônus: aplicar em vídeo (point/box tracking).

### Projeto 16.5 — OCR + extração estruturada de PDFs
- Use Marker, Docling ou pipeline próprio (tabula + Tesseract + LLM).
- Extraia tabelas + texto preservando estrutura.
- Compare com VLM (LLaVA, Qwen2-VL) fazendo extração end-to-end.

### Projeto 16.6 — Zero-shot classification com CLIP
- Use OpenCLIP.
- Construa classificador zero-shot para categorias arbitrárias.
- Compare com modelo supervisionado fine-tunado (precisa de labels) na mesma tarefa.

### Projeto 16.7 — Inferência em browser
- transformers.js: classificação ou detecção rodando no browser.
- Compare latência com inferência no servidor.

---

## ⚠️ Erros comuns

- **Augmentation errado**: rotação livre num dataset onde orientação importa (texto, faces).
- **Normalização inconsistente** entre treino e inferência (mean/std diferentes).
- **Overfitting silencioso** em datasets pequenos sem augmentation.
- **Confundir mAP@50 com mAP@50:95** — métricas COCO têm convenções específicas.
- **Esquecer de fixar data leakage** quando há mesma "instância" em splits diferentes (ex: vários frames do mesmo vídeo).
- **Ignorar resolução** — a maioria dos modelos VC tem ponto-doce de input específico.

---

## Conexão com módulos seguintes

| Conceito daqui | Aparece em |
|---|---|
| ViT | Multimodal (mod. [18](18_multimodal.md)) |
| CLIP, SigLIP | Multimodal (mod. [18](18_multimodal.md)) |
| Diffusion preview | Tópicos avançados (mod. [19](19_topicos_avancados.md)) |
| DINOv2 embeddings | Agentes com visão (mod. [13](13_agentes_tools_protocolos.md) estendido) |

---

## Checklist de saída

- [ ] Treinei CNN do zero e fiz transfer learning com ResNet/ViT.
- [ ] Apliquei DINOv2 como feature extractor em problema próprio.
- [ ] Usei SAM 2 e Grounding DINO em pipeline real.
- [ ] Treinei YOLO em domínio customizado.
- [ ] Tenho pipeline OCR/extração de PDF em produção.
- [ ] Construí app de browser com inferência VC client-side.
