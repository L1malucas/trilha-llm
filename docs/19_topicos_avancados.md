---
id: 19_topicos_avancados
title: "Módulo 19 — Tópicos Avançados"
sidebar_position: 19
---

# Módulo 19 — Tópicos Avançados

> **Objetivo**: cobrir frentes de pesquisa ativas em 2024–2025 — Mixture of Experts, State Space Models (Mamba), modelos de difusão, world models, neuro-symbolic, interpretability profundo, e fronteiras emergentes.
>
> **Pré-requisitos**: Módulos [07](07_transformers.mdx), [08](08_llms_arquiteturas.md), [09](09_treinamento_e_alinhamento.mdx). Conforto com leitura de papers.
>
> **Tempo de referência**: 6–10 semanas (não-linear; escolha sub-tópicos).

## Objetivos de aprendizagem

Ao final deste módulo você deve ser capaz de:

- Explicar por que SSMs (Mamba) escalam linearmente onde attention escala quadraticamente, e o que se perde nessa troca.
- Explicar o processo de difusão (forward de ruído, reverse de denoising) com intuição, não só o nome.
- Discutir com fluência pelo menos 3 destes tópicos o suficiente para ler o paper original sem depender de blog explicativo.
- Manter ceticismo calibrado: distinguir o que já é usado em produção do que ainda é pesquisa especulativa.

Este módulo é não-linear — escolha sub-tópicos por interesse, não precisa ler tudo em sequência.

---

## Por que isso importa

Aqui você sai de "engenheiro que aplica" para "engenheiro que acompanha pesquisa". Nenhum desses tópicos é "obrigatório" para construir produtos hoje, mas todos definirão como serão os modelos de 2026–2027. Ler papers em fluência é o resultado deste módulo.

---

## 19.1 Mixture of Experts (MoE) em profundidade

### Conceito
Em vez de FFN denso, ter N FFNs (experts). Um **router** escolhe top-k experts por token. Compute por token cresce sublinearmente em parâmetros totais.

### Papers fundamentais
- `Paper` **Outrageously Large Neural Networks (Sparse MoE)** — Shazeer et al. (2017). https://arxiv.org/abs/1701.06538
- `Paper` **Switch Transformers** — Fedus et al. (2021). https://arxiv.org/abs/2101.03961
- `Paper` **GShard** — Lepikhin et al. (2020). https://arxiv.org/abs/2006.16668
- `Paper` **Mixtral of Experts** — Jiang et al. (2024). https://arxiv.org/abs/2401.04088
- `Paper` **DeepSeek-V3 Technical Report** — MoE em escala extrema. https://arxiv.org/abs/2412.19437

### Desafios
- **Load balancing** entre experts (auxiliary loss).
- **Comunicação** em treinamento distribuído (all-to-all).
- **Memória total** alta para inferência.
- **Fine-tuning** mais delicado.

### Inferência eficiente de MoE
- Modelos como Mixtral 8×7B usam ~13B params ativos por token, mas precisam dos 47B em VRAM.
- Otimizações: expert offloading (CPU/disk), expert caching.

> A intuição central de MoE (roteador escolhendo especialistas, mais parâmetros totais sem custo proporcional por token) já foi construída no mod. [08](08_llms_arquiteturas.md#84-detalhes-arquiteturais-modernos-20242025). Aqui, o desafio novo é **load balancing**: sem incentivo explícito, o router tende a colapsar, favorecendo sempre os mesmos poucos experts (que ficam bem treinados) e ignorando os demais (que ficam subtreinados, criando um ciclo vicioso) — a auxiliary loss penaliza justamente essa distribuição desigual, empurrando o router a usar todos os experts de forma mais balanceada ao longo do treino.

---

## 19.2 State Space Models (SSMs) e Mamba

### Motivação
Transformers têm complexidade O(n²) em attention. RNNs são O(n) mas têm gradientes problemáticos. SSMs prometem o melhor dos dois mundos.

### Papers
- `Paper` **HiPPO: Recurrent Memory with Optimal Polynomial Projections** — Gu et al. (2020). https://arxiv.org/abs/2008.07669
- `Paper` **Efficiently Modeling Long Sequences with Structured State Spaces (S4)** — Gu et al. (2021). https://arxiv.org/abs/2111.00396
- `Paper` **Mamba: Linear-Time Sequence Modeling with Selective State Spaces** — Gu & Dao (2023). https://arxiv.org/abs/2312.00752
- `Paper` **Mamba-2: Transformers are SSMs** — Dao & Gu (2024). https://arxiv.org/abs/2405.21060
- `Paper` **Jamba** — modelo híbrido Transformer + Mamba (AI21). https://arxiv.org/abs/2403.19887

> **Intuição**: SSMs herdam a estrutura matemática de sistemas de controle clássicos — mantêm um "estado" comprimido de tamanho fixo que resume tudo relevante da sequência vista até agora, atualizado a cada passo por uma transformação (aprendida). Isso é estruturalmente parecido com o hidden state de uma RNN (mod. [05](05_deep_learning.md#55-redes-recorrentes-rnn-lstm-gru)) — e herda o benefício de custo linear (processar um passo a mais custa um incremento fixo, não uma comparação com todos os passos anteriores como em attention). A inovação do Mamba é tornar essa atualização de estado **seletiva**: em vez de uma transformação fixa igual pra todo input (como SSMs anteriores, S4), os parâmetros da atualização dependem do próprio input atual — o modelo pode "decidir" dinamicamente o que vale a pena reter no estado e o que descartar, mitigando o gargalo clássico de RNN (informação distante sendo progressivamente diluída) sem pagar o custo quadrático de attention completa.
>
> **Aplicação real**: o trade-off real é que SSMs processam sequencialmente durante *geração* (como RNN), mas o cálculo de treino pode ser paralelizado de forma eficiente (diferente de RNN clássica) — é essa combinação que torna Mamba competitivo com Transformers em benchmarks de linguagem, especialmente em contextos muito longos, onde o custo quadrático de attention se torna proibitivo (mod. [07](07_transformers.mdx#77-eficiência-de-attention)). Modelos híbridos (Jamba) apostam que combinar camadas de attention (melhores em recuperar informação específica e distante) com camadas Mamba (mais baratas, boas em resumir contexto) pode ser melhor que qualquer um dos dois puro.
>
> **Checkpoint**: sem olhar o texto, explique por que Mamba escala linearmente onde attention escala quadraticamente. Depois, explique o que "seletividade" adiciona sobre SSMs anteriores como S4.

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
- `Paper` **Denoising Diffusion Probabilistic Models (DDPM)** — Ho et al. (2020). https://arxiv.org/abs/2006.11239
- `Paper` **Score-Based Generative Modeling through SDEs** — Song et al. (2020). https://arxiv.org/abs/2011.13456
- `Paper` **High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion)** — Rombach et al. (2021). https://arxiv.org/abs/2112.10752
- `Paper` **Classifier-Free Guidance** — Ho & Salimans (2021). https://arxiv.org/abs/2207.12598
- `Paper` **DiT: Diffusion Transformers** — Peebles & Xie (2022). https://arxiv.org/abs/2212.09748
- `Paper` **Flow Matching for Generative Modeling** — Lipman et al. (2022). https://arxiv.org/abs/2210.02747
- `Paper` **Rectified Flow** — Liu et al. (2022). https://arxiv.org/abs/2209.03003 (base do FLUX e SD3).

> **Intuição**: o treino de um modelo de difusão tem duas metades. **Forward** (fixo, sem aprendizado): pegue uma imagem real e adicione ruído gaussiano gradualmente, passo a passo, até virar ruído puro — um processo simples e conhecido matematicamente. **Reverse** (o que se aprende): treine uma rede para prever, dado uma imagem ruidosa num passo qualquer, "o que foi adicionado" — ou seja, a rede aprende a **reverter** um passo de ruído por vez. Uma vez treinada, gerar uma imagem nova é começar de ruído puro (aleatório) e aplicar repetidamente o denoiser aprendido, passo a passo, até emergir uma imagem coerente — é literalmente esculpir uma imagem a partir de estática, removendo ruído iterativamente na direção que o modelo aprendeu ser "mais provável de gerar imagens reais".
>
> Classifier-free guidance é o mecanismo que permite controlar a geração por texto: durante o treino, o modelo aprende tanto a versão condicionada (com prompt) quanto a não-condicionada (sem prompt) da mesma tarefa; na geração, extrapolar na direção "condicionado menos não-condicionado" amplifica a influência do prompt, produzindo imagens mais fiéis à descrição (à custa de menos diversidade).
>
> **Checkpoint**: sem olhar o texto, explique o processo forward e reverse de um modelo de difusão com suas próprias palavras — o que exatamente a rede neural aprende a fazer?

### Modelos atuais (open)
- **Stable Diffusion 3 / 3.5** (Stability AI).
- **FLUX.1** (Black Forest Labs) — estado da arte open. https://blackforestlabs.ai/
- **HunyuanDiT** (Tencent).
- **Sana** (NVIDIA, eficiente).
- **Lumina-Next**.

### Difusão para texto?
Pesquisa ativa, ainda longe de competir com autoregressivo.
- `Paper` **Score-Based Continuous-Time Discrete Diffusion**.
- `Paper` **DiffuSeq**, **SEDD**.

### Difusão para vídeo
Mod. [18](18_multimodal.mdx) cobre. SDEs em alta dimensão, condicionamento temporal.

### Aplicações além de imagem
- Música (Stable Audio).
- Áudio (AudioLDM).
- Moléculas (drug discovery).
- Robótica (diffusion policy).

### Cursos
- `Curso` **Hugging Face Diffusion Course**. https://huggingface.co/learn/diffusion-course
- `Curso` **fast.ai — Stable Diffusion Deep Dive**.

---

## 19.4 World models e simuladores neurais

### Conceito
Aprender modelo do mundo (\(P(s' | s, a)\)) com rede neural; usar para planejamento, exploração, avaliação contrafactual.

### Papers
- `Paper` **World Models** — Ha & Schmidhuber (2018). https://arxiv.org/abs/1803.10122
- `Paper` **Dreamer V3** — Hafner et al. (2023). https://arxiv.org/abs/2301.04104
- `Paper` **GameNGen** — DOOM rodando em modelo de difusão. https://arxiv.org/abs/2408.14837
- `Paper` **Genie** (DeepMind, 2024) — gerar ambientes jogáveis a partir de imagens. https://arxiv.org/abs/2402.15391

> Mesmo conceito de model-based RL do mod. [17](17_aprendizado_reforco.md#175-model-based-rl-e-planning), levado ao extremo: em vez de um modelo simples do ambiente, treinar um simulador neural completo — GameNGen é o exemplo mais vívido, um modelo de difusão que aprendeu a simular DOOM jogável quadro a quadro, sem nenhum motor de jogo real por trás.

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
- **MemGPT** (visto em mod. [13](13_agentes_tools_protocolos.md)).
- **Infini-attention** (Google). https://arxiv.org/abs/2404.07143
- **Titans: Learning to Memorize at Test Time** (Google, 2024). https://arxiv.org/abs/2501.00663

### Trade-off "Lost in the Middle"
- `Paper` **Lost in the Middle: How Language Models Use Long Contexts** — Liu et al. (2023). https://arxiv.org/abs/2307.03172
- Long context não resolve tudo: modelos atendem mais ao início e fim do contexto.

> O trade-off "lost in the middle" e a comparação long context vs RAG já foram discutidos com intuição no mod. [12](12_rag.mdx#127-geração-com-contexto). As técnicas aqui (YaRN, LongRoPE) são extensões diretas do RoPE do mod. [07](07_transformers.mdx#73-posicionamento) — ajustam matematicamente como a rotação de posição se comporta além do comprimento visto no treino, permitindo extrapolar sem retreinar do zero.

---

## 19.6 Reasoning models a fundo

Mod. [09](09_treinamento_e_alinhamento.mdx) introduziu; aqui aprofundamos.

### Linhas de pesquisa
- **CoT supervisionado** (treina em traces).
- **Process Reward Models** — recompensa em passos intermediários.
- **Outcome-only RL** (R1-style).
- **Search augmented** — tree search durante geração.
- **Test-time compute scaling** — gastar mais compute na inferência. `Paper` https://arxiv.org/abs/2408.03314

### Papers
- `Paper` **Let's Verify Step by Step** (PRM). https://arxiv.org/abs/2305.20050
- `Paper` **DeepSeek-R1** — RL puro, comportamento emergente. https://arxiv.org/abs/2501.12948
- `Paper` **Scaling LLM Test-Time Compute Optimally** — Snell et al. (2024). https://arxiv.org/abs/2408.03314
- `Paper` **rStar-Math** — pequenos modelos com self-evolution. https://arxiv.org/abs/2501.04519

### Implicações
- Modelos pequenos + reasoning RL podem bater modelos grandes em tarefas verificáveis.
- Reasoning emergent é parcialmente entendido.

> Process Reward Models vs Outcome-only RL é uma escolha de *onde* colocar o sinal de recompensa: PRM recompensa cada passo intermediário do raciocínio (exige anotação mais granular, mas dá sinal mais denso); outcome-only (GRPO/R1-style, mod. [09](09_treinamento_e_alinhamento.mdx#96-reasoning-rl-estilo-r1)) recompensa só a resposta final, deixando o modelo "descobrir" que padrões de raciocínio levam a acertos — mais simples de implementar, mas exige mais exploração para o sinal esparso se propagar até os passos intermediários.

---

## 19.7 Neuro-symbolic e híbridos

Combinar redes neurais com raciocínio simbólico/lógico.

- **DeepProbLog**, **Logical Neural Networks**.
- **Neuro-Symbolic Concept Learner** (MIT). https://arxiv.org/abs/1904.12584
- **Tool-augmented LLMs** (mod. [13](13_agentes_tools_protocolos.md)) é, em parte, neuro-symbolic na prática.
- **Theorem proving** com LLMs: AlphaProof, AlphaGeometry (DeepMind). `Paper` https://www.nature.com/articles/s41586-023-06747-5

---

## 19.8 Interpretability profundo

Mod. [14](14_avaliacao_e_seguranca.md) introduziu. Aqui, frentes ativas:

- **Sparse Autoencoders (SAEs)**: extrair features interpretáveis. Anthropic: Sonnet, Opus desencaixotados. https://transformer-circuits.pub/2024/scaling-monosemanticity/
- **Activation steering** — manipular comportamento via vetores no espaço de ativação.
- **Circuit-level interpretability** — descobrir "circuitos" para tarefas específicas.
- **Concept-based explanations**.

### Cursos / programas
- `Curso` **ARENA** (intensivo de alignment + interp). https://www.arena.education/
- `Curso` **Neel Nanda — TransformerLens tutorials**. https://www.neelnanda.io/

### Why this matters
Interpretability é ferramenta para:
- **Safety**: detectar capacidades latentes.
- **Debugging**: entender por que modelo erra.
- **Edição**: corrigir comportamentos sem retreino caro (model editing — ROME, MEMIT).

### Papers
- `Paper` **Locating and Editing Factual Associations in GPT (ROME)** — Meng et al. (2022). https://arxiv.org/abs/2202.05262
- `Paper` **Mass-Editing Memory in a Transformer (MEMIT)**. https://arxiv.org/abs/2210.07229

> Activation steering estende diretamente a ideia de "features como direções no espaço de ativação" (mod. [14](14_avaliacao_e_seguranca.md#1411-mechanistic-interpretability)) de diagnóstico passivo para intervenção ativa: se uma direção corresponde a um conceito, somar (ou subtrair) essa direção às ativações durante a geração pode amplificar (ou suprimir) esse conceito no comportamento do modelo, sem retreinar nada — uma forma de "editar" comportamento diretamente na representação interna.

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
- `Paper` **Communication-Efficient Learning of Deep Networks (FedAvg)** — McMahan et al. (2016). https://arxiv.org/abs/1602.05629

---

## 19.10 Continual learning

Aprender sequencialmente sem esquecer (catastrophic forgetting).

### Métodos
- **Elastic Weight Consolidation (EWC)**.
- **Replay-based** methods.
- **Progressive networks**.
- **Adapter-based** (relacionado a LoRA).

### Conexão com LLMs
- **Continued pretraining** (mod. [09](09_treinamento_e_alinhamento.mdx)) é uma instância prática.
- **Model merging** (TIES, DARE, MoE-merge) — combinar modelos sem retreino.
- `Paper` **TIES-Merging**. https://arxiv.org/abs/2306.01708
- `Paper` **Model Soups** — Wortsman et al. (2022). https://arxiv.org/abs/2203.05482
- `Ferramenta` **mergekit** — kit prático. https://github.com/arcee-ai/mergekit

> Catastrophic forgetting já foi discutido com intuição no mod. [09](09_treinamento_e_alinhamento.mdx#97-continued-pretraining-cpt) (mistura de dados como mitigação). EWC ataca o mesmo problema de forma diferente: identifica quais pesos foram mais importantes para tarefas anteriores (via aproximação da curvatura da loss) e penaliza mudanças grandes nesses pesos específicos durante o treino da tarefa nova — uma forma de regularização direcionada, em vez de misturar dados.

---

## 19.11 Geometric Deep Learning, GNNs

Para dados estruturados como grafos: redes sociais, moléculas, recommender systems.

- `Paper` **Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges** — Bronstein et al. (2021). https://arxiv.org/abs/2104.13478
- `Livro` **Graph Representation Learning Book** — Hamilton. https://www.cs.mcgill.ca/~wlh/grl_book/
- **GNNs**: GCN, GraphSAGE, GAT, MPNN, Graph Transformer.
- **AlphaFold 2 & 3** — proteínas. `Paper` https://www.nature.com/articles/s41586-021-03819-2

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
- **Weak-to-strong generalization**. `Paper` https://arxiv.org/abs/2312.09390 (OpenAI).
- **Constitutional AI v2** evolution.
- **AISI evaluations** (UK/US AI Safety Institutes).

### Leituras
- `Livro` **Anthropic — Core Views on AI Safety**. https://www.anthropic.com/news/core-views-on-ai-safety
- `Livro` **DeepMind — AGI Safety blog series**.
- `Livro` **MIRI — Embedded Agency**.

---

## Projetos práticos (escolha 2–3)

### Projeto 19.1 — MoE pequeno do zero
- Modifique nanoGPT/llama2.c para ter MoE no FFN.
- Implemente router com top-2 experts.
- Aux loss para load balancing.
- Treine em corpus pequeno.

### Projeto 19.2 — Mamba pequeno from scratch
- Use repositório oficial Mamba como referência.
- Treine em Tiny Shakespeare.
- Compare com nanoGPT do mesmo tamanho.

> **Variante guiada**: meça o tempo de inferência de ambos conforme a sequência gerada cresce (100, 500, 1000 tokens) — deve evidenciar concretamente a diferença de escala linear (Mamba) vs quadrática (attention) discutida na seção 19.2.

### Projeto 19.3 — DDPM no MNIST/CIFAR-10
- Implemente DDPM completo: forward (add ruído), reverse (treinar denoiser), sampling.
- Visualize trajetória de geração.
- Adicione classifier-free guidance condicional.

> **Variante guiada**: visualize algumas imagens do processo forward (adicionando ruído progressivamente) antes de treinar o reverse — confirme visualmente que o processo forward é simples e correto antes de depurar o denoiser aprendido.

### Projeto 19.4 — Fine-tune de Stable Diffusion / FLUX com LoRA
- Use **diffusers** (Hugging Face).
- Conjunto de 20–50 imagens próprias.
- Compare estilos (DreamBooth-style vs LoRA-style).

### Projeto 19.5 — Reasoning RL pequeno
- Já antecipado em mod. [09](09_treinamento_e_alinhamento.mdx) e [17](17_aprendizado_reforco.md).
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

## Erros comuns

- **Tratar tudo aqui como "vai dominar logo"** — muitas dessas frentes são especulativas; mantenha ceticismo.
- **Confundir performance em paper com resultado em produção** — papers reportam o melhor caso.
- **Pular fundamentos** porque "é mais legal" — sem mod. [7](07_transformers.mdx)–[9](09_treinamento_e_alinhamento.mdx), este módulo é mágica.
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
