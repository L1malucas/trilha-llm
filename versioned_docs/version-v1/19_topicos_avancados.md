---
id: 19_topicos_avancados
title: "Módulo 19 — Tópicos Avançados"
sidebar_position: 19
---

# Módulo 19 — Tópicos Avançados

> **Objetivo**: cobrir frentes de pesquisa ativas em 2024–2025 — Mixture of Experts, State Space Models (Mamba), modelos de difusão, world models, neuro-symbolic, interpretability profundo, e fronteiras emergentes.
>
> **Pré-requisitos**: Módulos 07, 08, 09. Conforto com leitura de papers.
>
> **Tempo de referência**: 6–10 semanas (não-linear; escolha sub-tópicos).

---

## Por que isso importa

Aqui você sai de "engenheiro que aplica" para "engenheiro que acompanha pesquisa". Nenhum desses tópicos é "obrigatório" para construir produtos hoje, mas todos definirão como serão os modelos de 2026–2027. Ler papers em fluência é o resultado deste módulo.

---

## 19.1 Mixture of Experts (MoE) em profundidade

### Conceito
Em vez de FFN denso, ter N FFNs (experts). Um **router** escolhe top-k experts por token. Compute por token cresce sublinearmente em parâmetros totais.

### Papers fundamentais
- 📄 **Outrageously Large Neural Networks (Sparse MoE)** — Shazeer et al. (2017). https://arxiv.org/abs/1701.06538
- 📄 **Switch Transformers** — Fedus et al. (2021). https://arxiv.org/abs/2101.03961
- 📄 **GShard** — Lepikhin et al. (2020). https://arxiv.org/abs/2006.16668
- 📄 **Mixtral of Experts** — Jiang et al. (2024). https://arxiv.org/abs/2401.04088
- 📄 **DeepSeek-V3 Technical Report** — MoE em escala extrema. https://arxiv.org/abs/2412.19437

### Desafios
- **Load balancing** entre experts (auxiliary loss).
- **Comunicação** em treinamento distribuído (all-to-all).
- **Memória total** alta para inferência.
- **Fine-tuning** mais delicado.

### Inferência eficiente de MoE
- Modelos como Mixtral 8×7B usam ~13B params ativos por token, mas precisam dos 47B em VRAM.
- Otimizações: expert offloading (CPU/disk), expert caching.

---

## 19.2 State Space Models (SSMs) e Mamba

### Motivação
Transformers têm complexidade O(n²) em attention. RNNs são O(n) mas têm gradientes problemáticos. SSMs prometem o melhor dos dois mundos.

### Papers
- 📄 **HiPPO: Recurrent Memory with Optimal Polynomial Projections** — Gu et al. (2020). https://arxiv.org/abs/2008.07669
- 📄 **Efficiently Modeling Long Sequences with Structured State Spaces (S4)** — Gu et al. (2021). https://arxiv.org/abs/2111.00396
- 📄 **Mamba: Linear-Time Sequence Modeling with Selective State Spaces** — Gu & Dao (2023). https://arxiv.org/abs/2312.00752
- 📄 **Mamba-2: Transformers are SSMs** — Dao & Gu (2024). https://arxiv.org/abs/2405.21060
- 📄 **Jamba** — modelo híbrido Transformer + Mamba (AI21). https://arxiv.org/abs/2403.19887

### Por que importa
- **Linear scaling** em comprimento.
- Memória constante na geração.
- Competitivos com Transformers em escalas pequenas-médias.
- 2024+ vê experimentos híbridos (Jamba, Zamba, Hymba).

### Outras arquiteturas alternativas
- **RWKV** — RNN-Transformer híbrido. https://arxiv.org/abs/2305.13048
- **RetNet** (Microsoft). https://arxiv.org/abs/2307.08621
- **Hyena** (StanfordHazy). https://arxiv.org/abs/2302.10866
- **Liquid Foundation Models (LFMs)**.

---

## 19.3 Modelos de difusão

### Para imagens
A revolução pós-GAN. Aprendem a "des-ruidar" passo a passo.

### Papers fundamentais
- 📄 **Denoising Diffusion Probabilistic Models (DDPM)** — Ho et al. (2020). https://arxiv.org/abs/2006.11239
- 📄 **Score-Based Generative Modeling through SDEs** — Song et al. (2020). https://arxiv.org/abs/2011.13456
- 📄 **High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion)** — Rombach et al. (2021). https://arxiv.org/abs/2112.10752
- 📄 **Classifier-Free Guidance** — Ho & Salimans (2021). https://arxiv.org/abs/2207.12598
- 📄 **DiT: Diffusion Transformers** — Peebles & Xie (2022). https://arxiv.org/abs/2212.09748
- 📄 **Flow Matching for Generative Modeling** — Lipman et al. (2022). https://arxiv.org/abs/2210.02747
- 📄 **Rectified Flow** — Liu et al. (2022). https://arxiv.org/abs/2209.03003 (base do FLUX e SD3).

### Modelos atuais (open)
- **Stable Diffusion 3 / 3.5** (Stability AI).
- **FLUX.1** (Black Forest Labs) — estado da arte open. https://blackforestlabs.ai/
- **HunyuanDiT** (Tencent).
- **Sana** (NVIDIA, eficiente).
- **Lumina-Next**.

### Difusão para texto?
Pesquisa ativa, ainda longe de competir com autoregressivo.
- 📄 **Score-Based Continuous-Time Discrete Diffusion**.
- 📄 **DiffuSeq**, **SEDD**.

### Difusão para vídeo
Mod. 18 cobre. SDEs em alta dimensão, condicionamento temporal.

### Aplicações além de imagem
- Música (Stable Audio).
- Áudio (AudioLDM).
- Moléculas (drug discovery).
- Robótica (diffusion policy).

### Cursos
- 🎓 **Hugging Face Diffusion Course**. https://huggingface.co/learn/diffusion-course
- 🎓 **fast.ai — Stable Diffusion Deep Dive**.

---

## 19.4 World models e simuladores neurais

### Conceito
Aprender modelo do mundo (\(P(s' | s, a)\)) com rede neural; usar para planejamento, exploração, avaliação contrafactual.

### Papers
- 📄 **World Models** — Ha & Schmidhuber (2018). https://arxiv.org/abs/1803.10122
- 📄 **Dreamer V3** — Hafner et al. (2023). https://arxiv.org/abs/2301.04104
- 📄 **GameNGen** — DOOM rodando em modelo de difusão. https://arxiv.org/abs/2408.14837
- 📄 **Genie** (DeepMind, 2024) — gerar ambientes jogáveis a partir de imagens. https://arxiv.org/abs/2402.15391

---

## 19.5 Long context e infinite context

### Técnicas de extensão
- **Position interpolation**, **NTK-aware scaling**.
- **YaRN**. https://arxiv.org/abs/2309.00071
- **LongRoPE**. https://arxiv.org/abs/2402.13753
- **StreamingLLM** — sliding window com sink tokens. https://arxiv.org/abs/2309.17453

### Compressão de contexto
- **AutoCompressor**, **GIST tokens**.
- **RAG** (eterno trade-off).

### Memória externa
- **MemGPT** (visto em mod. 13).
- **Infini-attention** (Google). https://arxiv.org/abs/2404.07143
- **Titans: Learning to Memorize at Test Time** (Google, 2024). https://arxiv.org/abs/2501.00663

### Trade-off "Lost in the Middle"
- 📄 **Lost in the Middle: How Language Models Use Long Contexts** — Liu et al. (2023). https://arxiv.org/abs/2307.03172
- Long context não resolve tudo: modelos atendem mais ao início e fim do contexto.

---

## 19.6 Reasoning models a fundo

Mod. 09 introduziu; aqui aprofundamos.

### Linhas de pesquisa
- **CoT supervisionado** (treina em traces).
- **Process Reward Models** — recompensa em passos intermediários.
- **Outcome-only RL** (R1-style).
- **Search augmented** — tree search durante geração.
- **Test-time compute scaling** — gastar mais compute na inferência. 📄 https://arxiv.org/abs/2408.03314

### Papers
- 📄 **Let's Verify Step by Step** (PRM). https://arxiv.org/abs/2305.20050
- 📄 **DeepSeek-R1** — RL puro, comportamento emergente. https://arxiv.org/abs/2501.12948
- 📄 **Scaling LLM Test-Time Compute Optimally** — Snell et al. (2024). https://arxiv.org/abs/2408.03314
- 📄 **rStar-Math** — pequenos modelos com self-evolution. https://arxiv.org/abs/2501.04519

### Implicações
- Modelos pequenos + reasoning RL podem bater modelos grandes em tarefas verificáveis.
- Reasoning emergent é parcialmente entendido.

---

## 19.7 Neuro-symbolic e híbridos

Combinar redes neurais com raciocínio simbólico/lógico.

- **DeepProbLog**, **Logical Neural Networks**.
- **Neuro-Symbolic Concept Learner** (MIT). https://arxiv.org/abs/1904.12584
- **Tool-augmented LLMs** (mod. 13) é, em parte, neuro-symbolic na prática.
- **Theorem proving** com LLMs: AlphaProof, AlphaGeometry (DeepMind). 📄 https://www.nature.com/articles/s41586-023-06747-5

---

## 19.8 Interpretability profundo

Mod. 14 introduziu. Aqui, frentes ativas:

- **Sparse Autoencoders (SAEs)**: extrair features interpretáveis. Anthropic: Sonnet, Opus desencaixotados. https://transformer-circuits.pub/2024/scaling-monosemanticity/
- **Activation steering** — manipular comportamento via vetores no espaço de ativação.
- **Circuit-level interpretability** — descobrir "circuitos" para tarefas específicas.
- **Concept-based explanations**.

### Cursos / programas
- 🎓 **ARENA** (intensivo de alignment + interp). https://www.arena.education/
- 🎓 **Neel Nanda — TransformerLens tutorials**. https://www.neelnanda.io/

### Why this matters
Interpretability é ferramenta para:
- **Safety**: detectar capacidades latentes.
- **Debugging**: entender por que modelo erra.
- **Edição**: corrigir comportamentos sem retreino caro (model editing — ROME, MEMIT).

### Papers
- 📄 **Locating and Editing Factual Associations in GPT (ROME)** — Meng et al. (2022). https://arxiv.org/abs/2202.05262
- 📄 **Mass-Editing Memory in a Transformer (MEMIT)**. https://arxiv.org/abs/2210.07229

---

## 19.9 Federated learning e privacy-preserving ML

- **Federated learning** — treinar sem centralizar dados.
- **Differential privacy**.
- **Homomorphic encryption** + ML.
- **Secure multi-party computation**.
- **Confidential computing** (Intel SGX, AMD SEV).

### Por que importa
Healthcare, finance, on-device personalization.

### Referências
- 📄 **Communication-Efficient Learning of Deep Networks (FedAvg)** — McMahan et al. (2016). https://arxiv.org/abs/1602.05629

---

## 19.10 Continual learning

Aprender sequencialmente sem esquecer (catastrophic forgetting).

### Métodos
- **Elastic Weight Consolidation (EWC)**.
- **Replay-based** methods.
- **Progressive networks**.
- **Adapter-based** (relacionado a LoRA).

### Conexão com LLMs
- **Continued pretraining** (mod. 09) é uma instância prática.
- **Model merging** (TIES, DARE, MoE-merge) — combinar modelos sem retreino.
- 📄 **TIES-Merging**. https://arxiv.org/abs/2306.01708
- 📄 **Model Soups** — Wortsman et al. (2022). https://arxiv.org/abs/2203.05482
- 🛠 **mergekit** — kit prático. https://github.com/arcee-ai/mergekit

---

## 19.11 Geometric Deep Learning, GNNs

Para dados estruturados como grafos: redes sociais, moléculas, recommender systems.

- 📄 **Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges** — Bronstein et al. (2021). https://arxiv.org/abs/2104.13478
- 📚 **Graph Representation Learning Book** — Hamilton. https://www.cs.mcgill.ca/~wlh/grl_book/
- **GNNs**: GCN, GraphSAGE, GAT, MPNN, Graph Transformer.
- **AlphaFold 2 & 3** — proteínas. 📄 https://www.nature.com/articles/s41586-021-03819-2

---

## 19.12 AI for Science

- **AlphaFold 2/3** — proteínas. https://deepmind.google/discover/blog/alphafold-3-predicts-the-structure-and-interactions-of-all-of-life-s-molecules/
- **GNoME** — descoberta de materiais.
- **NeuralGCM** — modelagem climática.
- **Equiformer**, **SchNet** — química quântica.

### Por que importa
Talvez a aplicação mais impactante de ML moderno fora de tech.

---

## 19.13 Frontier safety e alignment

### Tópicos ativos
- **Sandbagging detection**.
- **Deceptive alignment**.
- **Power-seeking**.
- **Scalable oversight** (debate, recursive reward modeling).
- **Weak-to-strong generalization**. 📄 https://arxiv.org/abs/2312.09390 (OpenAI).
- **Constitutional AI v2** evolution.
- **AISI evaluations** (UK/US AI Safety Institutes).

### Leituras
- 📚 **Anthropic — Core Views on AI Safety**. https://www.anthropic.com/news/core-views-on-ai-safety
- 📚 **DeepMind — AGI Safety blog series**.
- 📚 **MIRI — Embedded Agency**.

---

## 🧪 Projetos práticos (escolha 2–3)

### Projeto 19.1 — MoE pequeno do zero
- Modifique nanoGPT/llama2.c para ter MoE no FFN.
- Implemente router com top-2 experts.
- Aux loss para load balancing.
- Treine em corpus pequeno.

### Projeto 19.2 — Mamba pequeno from scratch
- Use repositório oficial Mamba como referência.
- Treine em Tiny Shakespeare.
- Compare com nanoGPT do mesmo tamanho.

### Projeto 19.3 — DDPM no MNIST/CIFAR-10
- Implemente DDPM completo: forward (add ruído), reverse (treinar denoiser), sampling.
- Visualize trajetória de geração.
- Adicione classifier-free guidance condicional.

### Projeto 19.4 — Fine-tune de Stable Diffusion / FLUX com LoRA
- Use **diffusers** (Hugging Face).
- Conjunto de 20–50 imagens próprias.
- Compare estilos (DreamBooth-style vs LoRA-style).

### Projeto 19.5 — Reasoning RL pequeno
- Já antecipado em mod. 09 e 17.
- Aqui: aprofunde com process reward model.
- Compare outcome-only vs PRM.

### Projeto 19.6 — SAE em modelo pequeno
- Treine SAE sobre ativações de GPT-2 small.
- Identifique 5–10 features interpretáveis.
- Use SAELens como referência.

### Projeto 19.7 — Model merging
- Pegue 2–3 fine-tunes do mesmo base (LLaMA-3-8B).
- Combine com mergekit (TIES, DARE, model soup).
- Avalie em LM-Eval-Harness — frequentemente o merge bate cada um isolado.

### Projeto 19.8 — Long context experiment
- Avalie modelo seu (ou public) em "needle in haystack" em diferentes posições e tamanhos.
- Documente lost-in-the-middle empiricamente.

---

## ⚠️ Erros comuns

- **Tratar tudo aqui como "vai dominar logo"** — muitas dessas frentes são especulativas; mantenha ceticismo.
- **Confundir performance em paper com resultado em produção** — papers reportam o melhor caso.
- **Pular fundamentos** porque "é mais legal" — sem mod. 7–9, este módulo é mágica.
- **Treinar difusão sem GPU adequada** — é caro e lento; comece com modelos pequenos.

---

## Conexão com outros módulos

Este módulo recombina e estende todos os anteriores. Não há "próximo módulo natural"; vá para o 20 quando estiver pronto para projetos integradores.

---

## Checklist de saída

- [ ] Implementei (ou reproduzi) MoE pequeno from scratch.
- [ ] Implementei DDPM e gerei imagens com ele.
- [ ] Tive contato com Mamba/SSMs (ler paper + rodar código).
- [ ] Treinei SAE introdutório ou usei TransformerLens em experimentos próprios.
- [ ] Sei discutir com fluência: MoE, SSMs, difusão, reasoning RL, long context.
- [ ] Consigo ler papers da fronteira sem dependência de blog.
