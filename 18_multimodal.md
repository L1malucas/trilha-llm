# Módulo 18 — Modelos Multimodais

> **Objetivo**: dominar modelos que combinam texto + imagem + áudio (+ vídeo). VLMs (LLaVA, Qwen-VL, InternVL), modelos de fala (Whisper, voice models), modelos visão-linguagem-ação, omni-modal.
>
> **Pré-requisitos**: Módulos 07, 08, 16.
>
> **Tempo de referência**: 4–6 semanas.

---

## Por que isso importa

A interface humana com IA está se tornando inerentemente multimodal: foto + pergunta, voz + tela, vídeo + comando. Os modelos de ponta de 2024–2025 (Claude, GPT-4o, Gemini, Qwen-VL, LLaVA-OneVision) são todos multimodais. Agentes profissionais leem screenshots, escutam comandos, processam documentos com diagramas. Sem este módulo, você está limitado a texto puro.

---

## 18.1 Padrões arquiteturais multimodais

### Encoders separados + projetor (padrão LLaVA-style)
```
imagem → vision encoder (CLIP/SigLIP) → projetor MLP/Q-Former → tokens visuais
texto → tokenizer → tokens textuais
[tokens visuais ⊕ tokens textuais] → LLM → resposta
```

Este é o padrão dominante: **encoder visual congelado** + **projetor treinado** para mapear features visuais ao espaço de embedding do LLM.

### Cross-attention layered (Flamingo-style)
LLM tem camadas extras de cross-attention que consultam features visuais. Mais complexo, melhor para vídeo longo.

### Native multimodal / unified tokenizer
Modelo treinado from scratch com tokens unificados (texto + imagem patches + áudio frames). Tendência atual:
- **Chameleon** (Meta). 📄 https://arxiv.org/abs/2405.09818
- **Gemini** (treinamento conjunto desde o início).
- **GPT-4o** (omni-modal nativo).

### Mixture of Modality Experts
MoE onde experts se especializam por modalidade.

### Referências
- 📄 **Flamingo: a Visual Language Model for Few-Shot Learning** — Alayrac et al. (DeepMind, 2022). https://arxiv.org/abs/2204.14198
- 📄 **A Survey on Multimodal Large Language Models**. https://arxiv.org/abs/2306.13549

---

## 18.2 CLIP e modelos contrastivos visão-texto

### CLIP
Treina encoder visual + encoder textual com **contrastive loss**: pares (imagem, caption) próximos no espaço; não-pares afastados.

📄 **Learning Transferable Visual Models From Natural Language Supervision (CLIP)** — Radford et al. (2021). https://arxiv.org/abs/2103.00020

### Por que CLIP foi um marco
- **Zero-shot classification**: classifique sem treinar — só descreva as classes em texto.
- **Embeddings universais**: imagem e texto no mesmo espaço.
- **Backbone para muitas coisas**: Stable Diffusion, LLaVA, etc.

### Variantes melhoradas
- **OpenCLIP** (LAION) — reproduções abertas em vários tamanhos. https://github.com/mlfoundations/open_clip
- **SigLIP / SigLIP 2** — sigmoid loss em vez de softmax, melhor escalabilidade. 📄 https://arxiv.org/abs/2303.15343
- **EVA-CLIP** — escala maior. 📄 https://arxiv.org/abs/2303.15389
- **Jina-CLIP** — multimodal embeddings unificados.

### Aplicações
- Image-text retrieval (em RAG multimodal).
- Zero-shot classification.
- Filtragem de datasets (LAION usou CLIP).
- Inicialização de VLMs.

---

## 18.3 BLIP, BLIP-2 e modelos de captioning

- 📄 **BLIP** — Li et al. (2022). https://arxiv.org/abs/2201.12086
- 📄 **BLIP-2** — uses Q-Former para conectar encoder visual congelado a LLM congelado, treinando só ponte. https://arxiv.org/abs/2301.12597
- 📄 **InstructBLIP** — instruct-tuning do BLIP-2. https://arxiv.org/abs/2305.06500

### Q-Former
Módulo aprendível que extrai número fixo de "query tokens" do encoder visual, comprimindo informação para o LLM. Padrão em vários VLMs.

---

## 18.4 LLaVA e a família de VLMs open

📄 **Visual Instruction Tuning (LLaVA)** — Liu et al. (2023). https://arxiv.org/abs/2304.08485

### LLaVA recipe
1. Pegue LLM forte (Vicuna/LLaMA).
2. Pegue vision encoder (CLIP).
3. Treine **só um projetor MLP** com pares (imagem, descrição).
4. Faça SFT com instruções visuais.

Resultado: VLM competitivo com fração do compute.

### Família LLaVA
- **LLaVA-1.5**, **LLaVA-NeXT** (high-res). https://llava-vl.github.io/
- **LLaVA-OneVision** — single-image, multi-image, vídeo. 📄 https://arxiv.org/abs/2408.03326
- **LLaVA-CoT / LLaVA-o1** — reasoning visual.

### Outros VLMs open de 2024–2025
- **Qwen-VL / Qwen2-VL / Qwen2.5-VL** (Alibaba). 📄 https://arxiv.org/abs/2409.12191
- **InternVL / InternVL 2 / InternVL 2.5** (OpenGVLab). 📄 https://arxiv.org/abs/2312.14238
- **MiniCPM-V** (OpenBMB) — VLM eficiente para edge. 📄 https://arxiv.org/abs/2408.01800
- **Pixtral** (Mistral). https://mistral.ai/news/pixtral-12b
- **Molmo** (Allen AI). 📄 https://arxiv.org/abs/2409.17146
- **Phi-3.5-Vision / Phi-4 Multimodal** (Microsoft).
- **PaliGemma** (Google) — modelo educacional aberto, ótima base. https://github.com/google-research/big_vision
- **NVLM** (NVIDIA). 📄 https://arxiv.org/abs/2409.11402
- **Idefics 2 / 3** (Hugging Face). https://arxiv.org/abs/2405.02246

---

## 18.5 Capacidades modernas de VLMs

### High-resolution / dynamic resolution
Modelos modernos (Qwen2-VL, InternVL 2.5) processam imagens em resolução nativa, sem downsampling agressivo. Crítico para OCR, leitura de documentos, UI.

### Multi-image e vídeo
- Múltiplas imagens em um prompt.
- Vídeo como sequência de frames com pos embeddings temporais.
- LLaVA-OneVision, Qwen2-VL, InternVL 2 cobrem.

### Document understanding
- **Donut**, **Pix2Struct**, **DocOwl**, **mPLUG-DocOwl**.
- VLMs modernos fazem OCR + layout + pergunta sobre documento end-to-end.

### Grounding e referring
- "Aponte onde está X na imagem" — modelo retorna bounding box.
- **Grounding DINO**, **Florence-2**, **Qwen2-VL** com grounding.

### Computer use / GUI agents
- Modelos especializados em ler screenshots e gerar ações (click, type).
- **CogAgent**, **ShowUI**, **Anthropic Claude Computer Use**, **OS-Copilot**.

### Referências
- 📄 **Qwen2-VL Technical Report**. https://arxiv.org/abs/2409.12191
- 📄 **CogAgent**. https://arxiv.org/abs/2312.08914

---

## 18.6 Modelos de áudio e fala

### ASR (Automatic Speech Recognition)
- 📄 **Whisper** — Radford et al. (OpenAI, 2022). https://arxiv.org/abs/2212.04356
  - Aberto, multilíngue, robusto.
  - Variantes otimizadas: **faster-whisper** (CTranslate2), **whisper.cpp**, **Distil-Whisper**.
- **SeamlessM4T** (Meta) — speech-to-text + speech-to-speech multilíngue.
- **NVIDIA Parakeet, Canary** — alta precisão, otimizados.
- **wav2vec 2.0** — self-supervised foundation. 📄 https://arxiv.org/abs/2006.11477

### TTS (Text-to-Speech)
- **XTTS / Coqui TTS** (descontinuado, mas códigos abertos). https://github.com/coqui-ai/TTS
- **Piper** — TTS local rápido. https://github.com/rhasspy/piper
- **F5-TTS**, **MeloTTS**, **StyleTTS 2**, **Kokoro TTS** (small-but-good 2024–2025).
- **Bark** — gerativo expressivo (Suno).
- **Tortoise TTS** — qualidade alta, lento.

### Modelos de áudio gerais (música, sons)
- **MusicGen** (Meta). https://arxiv.org/abs/2306.05284
- **Stable Audio**, **AudioLDM**.
- **CLAP** — CLIP para áudio.

### Speech-to-Speech / voice
- **GPT-4o** (proprietário).
- **Moshi** (Kyutai, open). https://arxiv.org/abs/2410.00037
- **VITA** (Tencent).
- **Mini-Omni**, **GLM-4-Voice**.

### Referências
- 🎓 **Hugging Face Audio Course**. https://huggingface.co/learn/audio-course

---

## 18.7 Modelos de vídeo

### Compreensão (video → text/answer)
- **VideoChat / VideoChat2**.
- **Video-LLaVA**, **LLaVA-NeXT-Video**.
- **InternVideo, InternVideo 2**. https://arxiv.org/abs/2403.15377
- **Qwen2-VL** com vídeos.
- **Gemini** (long-context vídeo até horas).

### Geração (text → vídeo)
- **Sora** (OpenAI, proprietário).
- **Veo** (Google).
- **Mochi 1** (Genmo, open). https://www.genmo.ai/play
- **CogVideoX** (open). https://github.com/THUDM/CogVideo
- **HunyuanVideo** (Tencent, open). https://arxiv.org/abs/2412.03603
- **LTX-Video** (Lightricks, open).
- **Stable Video Diffusion** (Stability AI).
- **AnimateDiff**.

### Referências
- 📄 **Sora technical report** (OpenAI, 2024) — não há paper formal, mas blog post detalhado. https://openai.com/research/video-generation-models-as-world-simulators

---

## 18.8 Modelos visão-linguagem-ação (VLA)

Para robótica e agentes embodied.

- **RT-1, RT-2** (Google) — modelos que produzem ações de robô a partir de visão + instrução. 📄 https://robotics-transformer2.github.io/
- **OpenVLA** (open). https://openvla.github.io/
- **π0** (Physical Intelligence). https://www.physicalintelligence.company/blog/pi0
- **Octo**.

---

## 18.9 Treinamento e fine-tuning de VLMs

### Estágios
1. **Pretrain do projetor**: dados (imagem, caption) — LAION, COYO, CC12M.
2. **SFT visual**: dados (imagem, instrução, resposta) — LLaVA-Instruct, ShareGPT4V, etc.
3. **(opcional) DPO/RLHF visual**.

### Fine-tuning
- **LoRA** sobre o LLM + projetor.
- **Hugging Face TRL** suporta SFT e DPO multimodal.
- **PaliGemma** é referência educacional (recipe completa pública).

### Datasets visuais úteis
- **LAION-2B / LAION-COCO**.
- **COYO-700M**.
- **DataComp**.
- **ShareGPT4V** — captions ricas geradas por GPT-4V.
- **ALLaVA**, **Cambrian-7M**.
- Documentos: **DocVQA**, **ChartQA**, **InfographicVQA**, **AI2D**.

### Referências práticas
- 🎓 **Hugging Face — Fine-tuning VLMs**. https://huggingface.co/learn/cookbook/fine_tuning_vlm_trl

---

## 18.10 Avaliação de multimodal

### Benchmarks
- **MMMU** — Massive Multi-discipline Multimodal Understanding. 📄 https://arxiv.org/abs/2311.16502
- **MMBench** — checklists granulares. https://arxiv.org/abs/2307.06281
- **MM-Vet**. https://arxiv.org/abs/2308.02490
- **OCRBench**, **DocVQA**, **ChartQA**, **AI2D**.
- **MathVista**, **MathVerse** — raciocínio matemático visual.
- **POPE** — hallucination em VLMs.
- **VideoMME**, **MVBench** — vídeo.
- **Open VLM Leaderboard** (HF). https://huggingface.co/spaces/opencompass/open_vlm_leaderboard

### Métricas específicas
- **Hallucination rate** em imagens.
- **Grounding accuracy** (IoU de bounding boxes).
- **OCR accuracy** em documentos.

---

## 18.11 Multimodal RAG e embeddings

### RAG com imagens
- Embeddings unificados (CLIP, SigLIP, Jina-CLIP, **CLIP-Math**, **Cohere Multimodal**).
- Busca: query texto → recupera imagens; query imagem → recupera textos.
- **ColPali** — RAG sobre PDFs com layout via visão. 📄 https://arxiv.org/abs/2407.01449

### Document RAG moderno
ColPali e variantes pulam OCR: embeddam **imagem da página** diretamente. Excelente para layouts complexos (finance, science, legal).

---

## 🧪 Projetos práticos

### Projeto 18.1 — Zero-shot com CLIP
- Use OpenCLIP em Python e em TS (transformers.js).
- Tarefa: classificar imagens em categorias arbitrárias só com texto.
- Compare com SigLIP.

### Projeto 18.2 — VLM local rodando
- Rode Qwen2.5-VL 7B ou InternVL 2.5 8B via vLLM ou Ollama.
- Tarefa: análise de screenshots, OCR de PDFs, captioning.
- Compare com LLaVA-NeXT.

### Projeto 18.3 — RAG multimodal de fato
- Corpus: PDFs com diagramas (papers, relatórios).
- Use ColPali ou pipeline com VLM + embeddings de página.
- Compare com pipeline tradicional (OCR + texto-only RAG).

### Projeto 18.4 — Fine-tune mini-VLM
- Use PaliGemma-3B ou MiniCPM-V.
- Domínio próprio: classificação visual, descrição estilizada, ou Q&A sobre imagens específicas.
- Use HF TRL com LoRA.

### Projeto 18.5 — Pipeline de fala completo
- ASR (Whisper / faster-whisper).
- LLM (modelo local).
- TTS (Piper / Kokoro).
- Loop voz-para-voz local com latência baixa.

### Projeto 18.6 — Agente que vê tela
- Use Qwen2-VL ou Anthropic Computer Use.
- Tarefa: navegação simples em um app.
- Documente falhas e custos.

### Projeto 18.7 — Geração de vídeo
- Use CogVideoX ou Mochi 1 (open) localmente ou em cloud.
- Compare prompts e resultados.

---

## ⚠️ Erros comuns

- **Misturar tokenizer/processor errado**: VLMs têm processors específicos (image_processor + tokenizer); usar errado quebra silenciosamente.
- **Resolução errada**: forçar 224×224 em modelos high-res = qualidade despenca.
- **OCR delegado a VLM** sem fallback: VLM pode alucinar texto; valide com OCR clássico em paralelo.
- **Custo subestimado**: vídeo é **caro** (muitos tokens); planeje.
- **Eval só em inglês**: VLMs degradam visivelmente em PT-BR e em diagramas em PT.
- **Hallucination visual** silenciosa: modelo descreve algo que não está na imagem; valide com perguntas de "tem ou não tem".

---

## Conexão com outros módulos

| Conceito daqui | Aparece em |
|---|---|
| VLMs | Agentes computer-use (mod. 13) |
| Multimodal embeddings | RAG (mod. 12) |
| Whisper / TTS | Aplicações de produção (mod. 15) |
| Diffusion models | Tópicos avançados (mod. 19) |
| Avaliação multimodal | Avaliação (mod. 14) |

---

## Checklist de saída

- [ ] Rodei pelo menos 3 VLMs open localmente, comparando-os.
- [ ] Construí pipeline de RAG multimodal (com ColPali ou equivalente).
- [ ] Fine-tunei um VLM pequeno em domínio próprio.
- [ ] Tenho pipeline voz→voz funcionando localmente.
- [ ] Avaliei VLMs em benchmark próprio em PT-BR.
- [ ] Sei distinguir modelos com encoder separado vs nativos multimodais.
