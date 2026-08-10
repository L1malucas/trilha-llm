---
id: 08_llms_arquiteturas
title: "Módulo 08 — Arquiteturas de LLMs"
sidebar_position: 8
---

# Módulo 08 — Arquiteturas de LLMs

> **Objetivo**: conhecer as famílias de LLMs, suas escolhas arquiteturais, scaling laws, e os modelos abertos de 2024–2025 que definem o estado da arte open-source.
>
> **Pré-requisitos**: Módulo [07](07_transformers.mdx).
>
> **Tempo de referência**: 3–4 semanas.

## Objetivos de aprendizagem

Ao final deste módulo você deve ser capaz de:

- Escolher entre encoder-only, decoder-only e encoder-decoder para uma tarefa dada, com justificativa.
- Explicar o resultado de Chinchilla (razão tokens/parâmetro) e por que ele mudou a forma de treinar LLMs.
- Listar as mudanças arquiteturais modernas (RMSNorm, SwiGLU, RoPE, GQA) e conectá-las ao que já foi visto no mod. 07.
- Explicar por que modelos de raciocínio (o1, DeepSeek-R1) usam mais compute em inferência, não só em treino.
- Avaliar um modelo aberto por critérios de engenharia (licença, tamanho, idioma, benchmark), não por hype.

---

## Por que isso importa

Saber que "GPT é decoder-only" não é suficiente. Você precisa entender por que **LLaMA 3, Mistral, Qwen 2.5, Gemma 2, Phi-3** fazem escolhas diferentes — e o que muda em performance, memória, custo de inferência. Isso é o que separa quem escolhe modelo por hype de quem escolhe por engenharia.

---

## 8.1 Famílias arquiteturais

### Encoder-only
**Para quê**: classificação, extração, embeddings, NER.

- `Paper` **BERT** — Devlin et al. (2018). https://arxiv.org/abs/1810.04805
- `Paper` **RoBERTa** (BERT melhor treinado) — Liu et al. (2019). https://arxiv.org/abs/1907.11692
- `Paper` **DeBERTa** (disentangled attention, ainda muito competitivo) — He et al. (2020). https://arxiv.org/abs/2006.03654
- `Paper` **ELECTRA** (replaced token detection, mais eficiente que MLM) — Clark et al. (2020). https://arxiv.org/abs/2003.10555
- `Paper` **ModernBERT** (2024) — atualiza BERT com avanços recentes. https://arxiv.org/abs/2412.13663

### Decoder-only (autoregressivos)
**Para quê**: geração, chat, completion, raciocínio.

- `Paper` **GPT-2** — Radford et al. (2019). https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- `Paper` **GPT-3 (Language Models are Few-Shot Learners)** — Brown et al. (2020). https://arxiv.org/abs/2005.14165
- `Paper` **LLaMA** — Touvron et al. (2023). https://arxiv.org/abs/2302.13971
- `Paper` **LLaMA 2** — Touvron et al. (2023). https://arxiv.org/abs/2307.09288
- `Paper` **The Llama 3 Herd of Models** (LLaMA 3) — Meta AI (2024). https://arxiv.org/abs/2407.21783
- `Paper` **Mistral 7B** — Jiang et al. (2023). https://arxiv.org/abs/2310.06825
- `Paper` **Mixtral of Experts** — Jiang et al. (2024). https://arxiv.org/abs/2401.04088
- `Paper` **Qwen2 Technical Report** — Yang et al. (2024). https://arxiv.org/abs/2407.10671
- `Paper` **Qwen2.5 Technical Report** (2024). https://arxiv.org/abs/2412.15115
- `Paper` **Gemma: Open Models** — Google DeepMind (2024). https://arxiv.org/abs/2403.08295
- `Paper` **Gemma 2** — Google DeepMind (2024). https://arxiv.org/abs/2408.00118
- `Paper` **Phi-3 Technical Report** (Microsoft, foco em modelos pequenos potentes). https://arxiv.org/abs/2404.14219
- `Paper` **DeepSeek-V3 Technical Report** — DeepSeek-AI (2024). https://arxiv.org/abs/2412.19437
- `Paper` **DeepSeek-R1** (raciocínio via RL puro). https://arxiv.org/abs/2501.12948

### Encoder-Decoder
**Para quê**: text-to-text unificado (tradução, sumarização, QA).

- `Paper` **T5** — Raffel et al. (2019). https://arxiv.org/abs/1910.10683
- `Paper` **BART** — Lewis et al. (2019). https://arxiv.org/abs/1910.13461
- `Paper` **mT5** (multilíngue). https://arxiv.org/abs/2010.11934
- `Paper` **Flan-T5** (instruction-tuned T5). https://arxiv.org/abs/2210.11416
- `Paper` **NLLB-200** (No Language Left Behind, tradução em 200 idiomas). https://arxiv.org/abs/2207.04672

> Se a intuição por trás de cada arquitetura (revisor vs contador de histórias vs tradutor) não estiver fresca, vale revisitar o mod. [07](07_transformers.mdx#74-arquiteturas-de-transformer) antes de seguir — aqui entramos nas famílias de modelos reais que implementam cada padrão, não na arquitetura em si.

---

## 8.2 Scaling Laws

### Conceito
Como performance varia com **parâmetros (N)**, **dados (D)**, e **compute (C)**? Existe um equilíbrio ótimo?

### Papers fundamentais
- `Paper` **Scaling Laws for Neural Language Models** — Kaplan et al. (2020). https://arxiv.org/abs/2001.08361 (Conclusão: aumentar N rapidamente; D mais lento.)
- `Paper` **Training Compute-Optimal Large Language Models (Chinchilla)** — Hoffmann et al. (2022). https://arxiv.org/abs/2203.15556 (Corrige Kaplan: N e D devem escalar juntos. Resultado: tokens por parâmetro ≈ 20 é o ótimo de compute.)
- `Paper` **Beyond Chinchilla-Optimal: Training Smaller Models Longer** (LLaMA 3 e outros pós-Chinchilla mostram que treinar **muito mais** que Chinchilla-ótimo melhora qualidade na inferência, mesmo desperdiçando compute de treino).

> **Intuição**: para um orçamento fixo de compute, você pode gastar em mais *parâmetros* (modelo maior) ou mais *dados* (mais tokens de treino) — Kaplan sugeriu priorizar parâmetros; Chinchilla mostrou empiricamente que a maioria dos modelos anteriores estava **subtreinada em dados** relativo ao tamanho — o ótimo real fica perto de 20 tokens de treino por parâmetro do modelo. Um exemplo concreto: um modelo de 10B parâmetros, Chinchilla-ótimo, treinaria em ~200B tokens, não menos.
>
> **Por que "além de Chinchilla-ótimo" faz sentido na prática**: Chinchilla otimiza o custo de *treino*, mas ignora o custo de *inferência* — um modelo menor, treinado bem além do ponto Chinchilla-ótimo (mais tokens do que o "ideal" para aquele tamanho), pode alcançar qualidade comparável a um modelo maior Chinchilla-ótimo, só que depois é mais barato de rodar em produção milhões de vezes. É por isso que LLaMA 3 8B foi treinado em ~15 trilhões de tokens — muito além do que Chinchilla recomendaria para esse tamanho — porque o custo de inferência ao longo da vida do modelo supera de longe o custo extra de treino.
>
> **Checkpoint**: sem olhar o texto, explique em uma frase o que Chinchilla corrigiu em relação ao trabalho anterior de Kaplan. Depois, explique por que treinar "além do Chinchilla-ótimo" pode ser uma boa decisão de engenharia, mesmo sendo "sub-ótimo" em compute de treino.

### Implicação prática
- **Modelos pequenos bem treinados** (ex: Phi-3 mini, Llama 3.2 3B) podem competir com modelos grandes.
- **Tokens consumidos** é métrica tão importante quanto parâmetros.

---

## 8.3 Pré-treinamento: o que faz uma LLM "saber"

### Componentes
- **Dataset massivo** (trilhões de tokens): Common Crawl, Wikipedia, livros, código, papers, fóruns.
- **Curadoria e limpeza**: deduplicação, filtros de qualidade, balanceamento de domínios.
- **Tokenização** (mod. [06](06_nlp_classico.md)).
- **Objetivo**: next-token prediction (decoder) ou MLM (encoder).
- **Treinamento distribuído**: data parallel, tensor parallel, pipeline parallel, FSDP.

### Datasets abertos relevantes
- **The Pile** (EleutherAI). https://arxiv.org/abs/2101.00027
- **C4** (Colossal Clean Crawled Corpus). Usado por T5.
- **RedPajama** (reprodução do dataset do LLaMA original). https://github.com/togethercomputer/RedPajama-Data
- **FineWeb / FineWeb-Edu** (HuggingFace, 2024). https://huggingface.co/datasets/HuggingFaceFW/fineweb
- **Dolma** (Allen AI). https://allenai.org/dolma
- **Common Corpus** (especial atenção a licenças abertas).

> O ciclo completo de pré-treinamento (curadoria → tokenização → treino distribuído → checkpoints) é aprofundado com pipeline detalhado no mod. [09](09_treinamento_e_alinhamento.mdx) — aqui o foco é nos ingredientes, lá é no processo.

### Referências
- `Paper` **Scaling Language Models: Methods, Analysis & Insights from Training Gopher** — DeepMind (2021). https://arxiv.org/abs/2112.11446
- `Paper` **OPT: Open Pre-trained Transformer** — Meta (2022, com logbook do treinamento, leitura excelente). https://arxiv.org/abs/2205.01068
- `Paper` **BLOOM** (consórcio aberto BigScience, 176B). https://arxiv.org/abs/2211.05100

---

## 8.4 Detalhes arquiteturais modernos (2024–2025)

### O que mudou desde o Transformer original
- **RMSNorm** em vez de LayerNorm (mais simples, igual ou melhor).
- **SwiGLU** em vez de ReLU/GELU no FFN.
- **RoPE** em vez de positional encoding sinusoidal.
- **Grouped-Query Attention (GQA)** para inferência mais eficiente.
- **Sliding Window Attention** (Mistral) para contexto longo.
- **Mixture of Experts (MoE)** para escalar parâmetros mantendo compute por token.
- **Multi-token prediction** (Meta, DeepSeek) — prever vários tokens à frente como auxiliar.
- **Long context**: 128k → 1M+ tokens (LongRoPE, YaRN, NTK-aware scaling).

> Cada um destes já foi explicado com intuição no mod. [07](07_transformers.mdx#75-anatomia-de-um-bloco-transformer) (RMSNorm, SwiGLU, RoPE) e [07.7](07_transformers.mdx#77-eficiência-de-attention) (GQA) — esta lista é o "onde cada peça é usada na prática": praticamente todo LLM decoder-only moderno (LLaMA, Mistral, Qwen) combina as quatro primeiras como padrão de fato em 2024-2025, não são mais escolhas exóticas.
>
> **Intuição — MoE**: em vez de uma única rede feed-forward densa processando cada token, MoE tem várias redes "especialistas" (experts) e um roteador aprendido que decide, por token, quais 1-2 especialistas ativar. O modelo tem muito mais parâmetros *totais* (todos os especialistas somados), mas cada token só passa por uma fração pequena deles — é assim que Mixtral 8×7B tem ~47B parâmetros totais mas custo de inferência por token comparável a um modelo denso de ~13B. É a resposta arquitetural direta pro trade-off "mais parâmetros = mais capacidade, mas mais custo por token".
>
> **Checkpoint**: sem olhar o texto, explique por que MoE permite mais parâmetros sem aumentar proporcionalmente o custo de inferência por token.

### Referências
- `Paper` **Mixture of Experts (Outrageously Large NN)** — Shazeer et al. (2017). https://arxiv.org/abs/1701.06538
- `Paper` **Switch Transformers** — Fedus et al. (2021). https://arxiv.org/abs/2101.03961
- `Paper` **YaRN: Efficient Context Window Extension** — Peng et al. (2023). https://arxiv.org/abs/2309.00071
- `Paper` **LongRoPE** — Microsoft (2024). https://arxiv.org/abs/2402.13753

---

## 8.5 Modelos de raciocínio (reasoning)

A onda de 2024–2025 — modelos otimizados para "pensar" antes de responder.

- `Paper` **Let's Verify Step by Step** (Process Reward Models). https://arxiv.org/abs/2305.20050
- `Paper` **DeepSeek-R1** — RL puro para raciocínio, com modelo aberto. https://arxiv.org/abs/2501.12948
- `Paper` **Self-Taught Reasoner (STaR)** — Zelikman et al. (2022). https://arxiv.org/abs/2203.14465

### Conceito
Modelos geram "chain of thought" longo *antes* da resposta final. Treinamento via RL com recompensa em corretude. Aumenta drasticamente performance em matemática, código, lógica.

> **Intuição**: até aqui, escalar um LLM significava gastar mais compute em *treino* (mais parâmetros, mais dados, seção 8.2). Modelos de raciocínio introduzem um segundo eixo: gastar mais compute em *inferência*, deixando o modelo "pensar em voz alta" (gerar uma cadeia de raciocínio longa) antes de comprometer-se com a resposta final — mais tokens gerados internamente, mais chance de corrigir um erro de raciocínio no caminho. O treinamento via RL (mod. [17](17_aprendizado_reforco.md)) recompensa cadeias de raciocínio que chegam à resposta correta, sem necessariamente um humano ter anotado "o passo certo" — o modelo explora diferentes cadeias e aprende quais padrões de raciocínio tendem a funcionar, de forma parecida com como um humano aprende resolvendo muitos problemas e vendo quais abordagens dão certo.
>
> **Aplicação real**: é por isso que modelos de raciocínio custam mais por resposta (mais tokens gerados, mesmo que invisíveis ao usuário final) mas performam muito melhor em tarefas que exigem múltiplos passos lógicos corretos em sequência (matemática, código, quebra-cabeças lógicos) — errar um passo no meio geralmente invalida a resposta final nessas tarefas, então "pensar mais antes de responder" compensa o custo extra.
>
> **Checkpoint**: sem olhar o texto, explique a diferença entre escalar compute de *treino* e escalar compute de *inferência* — qual delas os modelos de raciocínio exploram?

---

## 8.6 Embeddings e modelos especializados

### Embedding models
- **Sentence-BERT** — Reimers & Gurevych (2019). https://arxiv.org/abs/1908.10084
- **E5** (Microsoft). https://arxiv.org/abs/2212.03533
- **BGE** (BAAI General Embedding). https://github.com/FlagOpen/FlagEmbedding
- **Nomic Embed**. https://arxiv.org/abs/2402.01613
- **Jina Embeddings**.
- **MTEB** — benchmark canônico. https://huggingface.co/spaces/mteb/leaderboard

### Reranker models
- **Cross-encoders** (Sentence-BERT cross variants).
- **bge-reranker**.

### Code models
- **CodeLLaMA**, **DeepSeek-Coder**, **Qwen2.5-Coder**, **StarCoder 2**.

### Math/Science models
- **DeepSeek-Math**, **Qwen-Math**, **MathStral**.

---

## 8.7 Onde encontrar modelos abertos

- **Hugging Face Hub** — repositório central. https://huggingface.co/models
- **Open LLM Leaderboard** (HF). https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard
- **LMSys Chatbot Arena** (avaliação humana via comparação A/B). https://arena.lmsys.org/
- **Artificial Analysis** (benchmarks comparativos com custos). https://artificialanalysis.ai/

### Famílias open de 2024–2025 que valem conhecer
| Família | Tamanhos | Origem | Característica |
|---|---|---|---|
| **LLaMA 3.x** | 1B–405B | Meta | Padrão da indústria open |
| **Mistral / Mixtral** | 7B, MoE 8×7B, 8×22B | Mistral AI | Eficiência |
| **Qwen 2.5** | 0.5B–72B | Alibaba | Multilíngue, code, math |
| **Gemma 2** | 2B, 9B, 27B | Google | Distilação de Gemini |
| **Phi-3 / Phi-4** | 3.8B, 7B, 14B | Microsoft | "Small but capable" |
| **DeepSeek V3 / R1** | até ~671B (MoE) | DeepSeek | Raciocínio e custo |
| **Command R+** | 104B | Cohere | RAG-otimizado |
| **OLMo 2** | 7B, 13B | Allen AI | Totalmente aberto (dados + pesos) |

---

## Projetos práticos

### Projeto 8.1 — Comparativo de modelos abertos
- Escolha 4 modelos: LLaMA 3 8B, Mistral 7B, Gemma 2 9B, Qwen 2.5 7B.
- Rode localmente (via Ollama ou llama.cpp — preview do mod. [10](10_eficiencia_e_inferencia_local.md)).
- Avalie em 20 prompts próprios (geração, raciocínio, código, PT-BR).
- Documente: latência, qualidade subjetiva, comportamento em PT-BR.

> **Variante guiada**: use exatamente os mesmos 20 prompts para os 4 modelos (não improvise por modelo) — só uma comparação controlada, com a mesma entrada, produz uma conclusão que você pode defender.

### Projeto 8.2 — Ler e fichar 3 papers de LLMs
Escolha 3 dos relacionados acima (sugestão: LLaMA 3, Mixtral, DeepSeek-V3).
- Para cada: arquitetura, dados, treino, avaliação.
- Faça resumo de 1–2 páginas em Markdown.

### Projeto 8.3 — Implementar uma "mini-LLaMA"
- Pegue o nanoGPT do mod. [07](07_transformers.mdx).
- Modifique para usar: RMSNorm, SwiGLU, RoPE, GQA.
- Treine no Tiny Shakespeare ou subconjunto do FineWeb.
- Compare com a versão original.
- Repo de referência: https://github.com/karpathy/nanoGPT (GPT-2-like) e https://github.com/karpathy/llama2.c (LLaMA-like).

> **Variante guiada**: troque um componente de cada vez (primeiro só RMSNorm, confirme que ainda treina; depois SwiGLU; depois RoPE; depois GQA) em vez de trocar tudo de uma vez — isso isola qual mudança, se alguma, introduz um bug, em vez de depurar as quatro simultaneamente.

### Projeto 8.4 — Análise de scaling laws
- Treine 3 nano-LLMs com tamanhos diferentes (1M, 5M, 25M params) por compute equivalente.
- Plote loss × parâmetros, loss × tokens.
- Verifique empiricamente o trade-off Chinchilla.

### Projeto 8.5 — Embeddings e MTEB
- Use 3 modelos de embedding (E5, BGE, Nomic).
- Reproduza 1 ou 2 tarefas do MTEB.
- Compare com seus próprios documentos.

---

## Erros comuns

- **"Modelo X é melhor"** sem qualificar tarefa, idioma, contexto.
- **Ignorar licenças** — nem todo modelo "aberto" tem uso comercial permitido (LLaMA tem cláusulas; Gemma tem termos próprios).
- **Avaliar apenas em inglês** — muitos modelos colapsam em PT-BR.
- **Confiar em leaderboards de benchmark** sem entender saturação e contaminação.
- **Achar que "MoE = mais caro"** — em compute por token ativo, MoE pode ser mais barato.

---

## Conexão com módulos seguintes

| Conceito daqui | Aparece em |
|---|---|
| Pre-training | Treinamento (mod. [09](09_treinamento_e_alinhamento.mdx)) |
| Embedding models | RAG (mod. [12](12_rag.mdx)) |
| Tokenização específica do modelo | Tudo o que envolve usar o modelo |
| Scaling laws | Decisões de fine-tuning (mod. [09](09_treinamento_e_alinhamento.mdx)) |
| Reasoning models | Agentes (mod. [13](13_agentes_tools_protocolos.md)), avaliação (mod. [14](14_avaliacao_e_seguranca.md)) |

---

## Checklist de saída

- [ ] Sei explicar a diferença entre BERT, GPT e T5 sem hesitação.
- [ ] Conheço pelo menos 5 famílias open de 2024–2025 e suas diferenças.
- [ ] Entendo Chinchilla scaling law.
- [ ] Implementei mini-LLaMA com RMSNorm + RoPE + SwiGLU.
- [ ] Sei consultar e interpretar leaderboards (MTEB, Open LLM, Chatbot Arena).
