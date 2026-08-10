---
id: 10_eficiencia_e_inferencia_local
title: "Módulo 10 — Eficiência e Inferência Local"
sidebar_position: 10
---

# Módulo 10 — Eficiência e Inferência Local

> **Objetivo**: rodar e otimizar LLMs em hardware modesto (laptop, GPU consumer, CPU, edge). Quantização, distilação, KV-cache, servidores de inferência, edge inference.
>
> **Pré-requisitos**: Módulos [07](07_transformers.mdx)–[09](09_treinamento_e_alinhamento.mdx).
>
> **Tempo de referência**: 3–5 semanas.

## Objetivos de aprendizagem

Ao final deste módulo você deve ser capaz de:

- Calcular quanta VRAM um modelo precisa em qualquer quantização, e explicar por que Q4 não destrói a qualidade.
- Explicar por que KV-cache é indispensável em geração autoregressiva — o que aconteceria sem ele.
- Explicar como speculative decoding acelera geração sem mudar a distribuição de saída do modelo grande.
- Explicar por que um modelo destilado pode aprender mais do "soft label" do teacher do que de rótulos binários.
- Escolher os parâmetros de sampling (temperature, top-k, top-p) certos para uma tarefa dada, com justificativa.

---

## Por que isso importa

Sem este módulo, qualquer projeto sério vira refém de API paga. **Engenharia real exige**:
- Rodar modelos em hardware comum.
- Decidir custo × latência × qualidade conscientemente.
- Garantir privacidade quando dado é sensível (médico, jurídico, interno).
- Operar offline (edge, mobile).

Bigtechs têm GPUs ilimitadas. Você provavelmente não tem. Esta é a sua vantagem competitiva técnica.

---

## 10.1 Quantização

### Conceito
Reduzir precisão dos pesos (e/ou ativações): FP32 → FP16 → BF16 → INT8 → INT4. Menos memória, mais velocidade, perda controlada de qualidade.

### Tipos
- **Post-Training Quantization (PTQ)** — sem retreino, mais rápido.
- **Quantization-Aware Training (QAT)** — com fine-tuning, melhor qualidade.

### Formatos importantes
- **GGUF** — formato do `llama.cpp`. Suporta Q2_K, Q3_K_S/M/L, Q4_K_S/M, Q5_K_S/M, Q6_K, Q8_0, etc.
- **GPTQ** — quantização inteligente baseada em segunda ordem. Bom para GPU. `Paper` https://arxiv.org/abs/2210.17323
- **AWQ (Activation-aware Weight Quantization)** — usa estatísticas de ativação. `Paper` https://arxiv.org/abs/2306.00978
- **bitsandbytes** — quantização 8-bit/4-bit em PyTorch.
- **FP8** (training-time, em H100/MI300).

> **Intuição**: por que reduzir de 32 bits para 4 bits por peso não destrói o modelo? Pesos de uma rede treinada não usam a precisão total do float32 de forma uniformemente importante — a maior parte da informação relevante está na *magnitude relativa* dos pesos (este peso é bem maior que aquele), não nos últimos bits de precisão decimal. Quantização mapeia a faixa de valores de cada peso para um número menor de "níveis" discretos (16 níveis em INT4) escolhidos para minimizar o erro — e a rede, sendo robusta a ruído por natureza (ela já foi treinada com dropout, batch noise etc.), tolera bem essa perda controlada. AWQ e GPTQ melhoram sobre quantização ingênua ao proteger especificamente os pesos que mais impactam a saída (identificados por estatística de ativação ou informação de segunda ordem), em vez de quantizar tudo com o mesmo cuidado.

### Regra de bolso
Para um modelo de **N** bilhões de parâmetros, VRAM aproximada para inferência:
- FP16 / BF16: 2N GB
- INT8: N GB
- Q4 (GGUF Q4_K_M): ~0.6N GB
- Q3: ~0.45N GB
(Adicione ~1–2 GB para KV-cache e overhead.)

Exemplo: LLaMA-3 8B em Q4_K_M cabe folgado em GPU de 8 GB.

> **Exemplo resolvido**: retomando o cálculo do mod. [02](02_programacao_ferramentas.md#24-hardware-e-aceleração) para um modelo de 13B — em FP16, `13 × 2 = 26 GB`. Em Q4_K_M, usando a regra de bolso acima: `13 × 0.6 ≈ 7.8 GB` — cabe numa GPU de consumo de 8-12GB. É essa conta, feita antes de qualquer outra decisão, que determina se um modelo roda na sua máquina local ou precisa de cloud.

### Trade-offs
- **Q4_K_M** é o "sweet spot" comum: ~99% da qualidade do FP16 em muitos casos.
- **Q3** já degrada perceptivelmente em modelos pequenos.
- **Q8** é praticamente lossless mas dobra o footprint vs Q4.

> **Checkpoint**: sem olhar o texto, explique por que um modelo tolera bem a perda de precisão de FP16 para Q4, mas degrada visivelmente em Q3 ou Q2. Depois, calcule a VRAM aproximada (Q4_K_M) de um modelo de 30B parâmetros.

### Referências
- `Paper` **LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale** — Dettmers et al. (2022). https://arxiv.org/abs/2208.07339
- `Paper` **GPTQ** — Frantar et al. (2022). https://arxiv.org/abs/2210.17323
- `Paper` **AWQ** — Lin et al. (2023). https://arxiv.org/abs/2306.00978
- `Paper` **A Survey of Quantization Methods for Efficient Neural Network Inference**. https://arxiv.org/abs/2103.13630
- `Livro` **Hugging Face — Quantization documentation**. https://huggingface.co/docs/transformers/main/en/quantization/overview

---

## 10.2 KV-Cache e otimizações de inferência

### KV-Cache
Em geração autoregressiva, K e V de tokens passados não mudam. Cachear evita recomputar.

### Otimizações modernas
- **PagedAttention** (vLLM) — gerência de KV-cache estilo memória virtual. `Paper` https://arxiv.org/abs/2309.06180
- **Continuous Batching** — agrupa requisições em diferentes estágios. Padrão em vLLM, TGI.
- **Speculative Decoding** — modelo pequeno propõe tokens, modelo grande valida. `Paper` https://arxiv.org/abs/2211.17192
- **Flash Decoding** — variante de Flash Attention para inferência.
- **Prompt Caching** — cachear prefixos comuns entre chamadas.

> **Intuição — KV-Cache**: gerar cada novo token exige recalcular attention (mod. [07](07_transformers.mdx#72-self-attention--o-coração)) contra *todos* os tokens anteriores. Sem cache, gerar o token N exigiria refazer o cálculo de K e V para os N-1 tokens anteriores *do zero*, toda vez — desperdício enorme, já que esses valores não mudam token a token (só dependem dos tokens já processados, que são fixos). O KV-cache guarda os K e V já calculados e só computa os do token novo a cada passo — é a diferença entre custo linear e custo quadrático repetido por token gerado.
>
> **Intuição — Speculative Decoding**: um modelo pequeno e rápido "chuta" vários tokens seguintes de uma vez; o modelo grande então verifica *em paralelo* (não sequencialmente) se aceitaria cada um desses tokens — aceitar é barato (um forward pass grande, verificando vários tokens de uma vez), rejeitar custa só voltar ao ponto de divergência. Quando o modelo pequeno acerta a maioria dos tokens (comum em texto previsível), o resultado final tem a *mesma distribuição* que rodar o modelo grande sozinho — a técnica não aproxima nem degrada qualidade, só explora que verificar é mais barato que gerar sequencialmente token a token.
>
> **Checkpoint**: sem olhar o texto, explique por que gerar sem KV-cache seria muito mais lento — o que teria que ser recalculado a cada novo token? Depois, explique por que speculative decoding não muda a qualidade da saída, só a velocidade.

### Referências
- `Paper` **Efficient Memory Management for LLM Serving with PagedAttention (vLLM)** — Kwon et al. (2023). https://arxiv.org/abs/2309.06180
- `Paper` **Speculative Decoding** — Leviathan et al. (2022). https://arxiv.org/abs/2211.17192

---

## 10.3 Servidores de inferência

### Locais (uma máquina)
- `Ferramenta` **Ollama** — UX simples, GGUF, Mac/Linux/Windows. https://ollama.com/
- `Ferramenta` **llama.cpp** — backend C++ rápido, base do Ollama, controla detalhes. https://github.com/ggml-org/llama.cpp
- `Ferramenta` **LM Studio** — GUI desktop. https://lmstudio.ai/
- `Ferramenta` **GPT4All**. https://www.nomic.ai/gpt4all
- `Ferramenta` **Jan**. https://jan.ai/

### Servidor (alta performance, multi-usuário)
- `Ferramenta` **vLLM** — padrão para servir LLMs em GPU. https://docs.vllm.ai/
- `Ferramenta` **TGI (Text Generation Inference)** (Hugging Face). https://github.com/huggingface/text-generation-inference
- `Ferramenta` **TensorRT-LLM** (NVIDIA). Performance máxima em hardware NVIDIA.
- `Ferramenta` **SGLang** — alternativa moderna com runtime próprio. https://github.com/sgl-project/sglang
- `Ferramenta` **MLC-LLM** — compila modelos para múltiplos backends (CUDA, Metal, Vulkan, WebGPU). https://github.com/mlc-ai/mlc-llm

### API compatível com OpenAI
A maioria dos servidores acima expõe API compatível OpenAI (`/v1/chat/completions`). Isso permite trocar provedor sem mudar código.

---

## 10.4 Inferência em CPU

Sim, é possível. Útil quando:
- Não há GPU.
- Custo de cloud > tempo de inferência.
- Requisições raras.

### Ferramentas
- **llama.cpp** com OpenBLAS / Accelerate / OpenMP.
- **ONNX Runtime** com providers CPU.
- **Intel OpenVINO**.

### Realismo
- Modelo 7B em Q4 num CPU moderno: 5–20 tokens/s.
- Suficiente para muitos casos (chatbot interno, batch processing).

---

## 10.5 Inferência em edge

### Mobile / Browser
- `Ferramenta` **transformers.js** — ONNX runtime no browser/Node. https://huggingface.co/docs/transformers.js
- `Ferramenta` **MLC-LLM Web** — modelos rodando via WebGPU. https://github.com/mlc-ai/web-llm
- `Ferramenta` **Apple MLX** — framework Apple para Apple Silicon. https://github.com/ml-explore/mlx
- `Ferramenta` **MediaPipe LLM** (Google). https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference

### Apple Silicon
- M1/M2/M3 com unified memory são surpreendentemente bons para LLMs.
- llama.cpp tem backend Metal otimizado.
- Modelos 70B Q4 rodam em Mac Studio com 64+ GB.

### Considerações TS
Para inferência client-side em web app:
- `transformers.js` é a opção mais madura.
- ONNX é o formato comum.
- Cuidado com tamanho: modelos > 1 GB pesam o bundle.

---

## 10.6 Distillation (destilação)

### Conceito
Treinar modelo pequeno (student) para imitar saídas de modelo grande (teacher). Resultado: 5–10× menor com 80–95% da qualidade.

### Tipos
- **Logit distillation** (clássica): KL entre distribuições teacher/student.
- **Sequence distillation**: student aprende as saídas geradas pelo teacher.
- **Feature distillation**: alinhar representações intermediárias.

### Modelos famosos por destilação
- **DistilBERT** (40% menor que BERT, 97% performance). `Paper` https://arxiv.org/abs/1910.01108
- **TinyLlama** (1.1B, treinado long).
- **Phi-3** (filosofia: dataset sintético de alta qualidade do GPT-4).
- **Gemma 2** (destilação de Gemini).

> **Intuição — "dark knowledge"**: um rótulo binário ("isto é um gato") carrega 1 bit de informação. A distribuição de probabilidade completa que o teacher produz ("87% gato, 10% raposa, 2% cachorro, ...") carrega muito mais — ela expressa *o quanto* o teacher "acha" que a imagem se parece com cada classe, não só a resposta final. Treinar o student para imitar essa distribuição inteira (logit distillation, via KL-divergence — mod. [01](01_matematica.md#13-probabilidade-e-estatística)) transfere esse conhecimento mais rico, chamado informalmente de "dark knowledge", em vez de só o rótulo final que um dataset rotulado convencional daria.
>
> **Checkpoint**: sem olhar o texto, explique por que a distribuição de probabilidade completa do teacher carrega mais informação que só o rótulo mais provável.

### Quando vale a pena
- Você tem teacher (próprio ou via API barata).
- Tarefa específica (classificação, extração, conversão de formato).
- Precisa de inferência rápida em produção.

### Referências
- `Paper` **Distilling the Knowledge in a Neural Network** — Hinton, Vinyals, Dean (2015). https://arxiv.org/abs/1503.02531
- `Paper` **MiniLLM: Knowledge Distillation of Large Language Models** — Gu et al. (2023). https://arxiv.org/abs/2306.08543
- `Paper` **Phi-3 Technical Report**. https://arxiv.org/abs/2404.14219

---

## 10.7 Pruning, Sparsity e Mixture of Experts

### Pruning
Remover pesos pouco importantes. Pode ser:
- **Estruturado** (cabeças/camadas inteiras) — ganho real em hardware.
- **Não-estruturado** (pesos individuais) — ganho real exige hardware sparse-aware.

### Modelos esparsos
- **NVIDIA 2:4 sparsity** — em A100/H100, pesa menos com pouca perda.

### MoE em inferência
Inferência de Mixtral 8×7B usa apenas ~13B "ativos" por token, mesmo tendo 47B totais. Vantagem para inferência; desvantagem em VRAM total necessária.

> A intuição de MoE (por que mais parâmetros totais não significa proporcionalmente mais custo por token) já foi construída no mod. [08](08_llms_arquiteturas.md#84-detalhes-arquiteturais-modernos-20242025) — aqui a nuance extra é que, mesmo com poucos parâmetros *ativos* por token, o modelo inteiro (todos os experts) ainda precisa estar carregado na VRAM, já que o roteador pode escolher qualquer expert a cada token. MoE economiza *compute*, não necessariamente *memória*.

### Referências
- `Paper` **The Lottery Ticket Hypothesis** — Frankle & Carbin (2018). https://arxiv.org/abs/1803.03635
- `Paper` **SparseGPT** — Frantar & Alistarh (2023). https://arxiv.org/abs/2301.00774
- `Paper` **Wanda: A Simple and Effective Pruning Approach for LLMs**. https://arxiv.org/abs/2306.11695

---

## 10.8 Sampling e parâmetros de geração

### Métodos
- **Greedy** — sempre o token mais provável. Determinístico. Repetitivo.
- **Beam Search** — mantém top-k hipóteses. Bom para tradução, ruim para geração aberta.
- **Temperature sampling** — softmax(logits / T). T < 1 = mais focado, T > 1 = mais aleatório.
- **Top-k** — amostra só dos k tokens mais prováveis.
- **Top-p (nucleus)** — amostra do menor conjunto cuja prob acumulada ≥ p.
- **Min-p** — variante moderna, mais robusta. https://arxiv.org/abs/2407.01082
- **Repetition penalty**, **frequency penalty**, **presence penalty**.

> **Intuição — temperature**: dividir os logits por `T` antes do softmax (mod. [01](01_matematica.md#13-probabilidade-e-estatística)) achata ou afia a distribuição de probabilidade. `T < 1` amplifica a diferença entre o token mais provável e os demais (a distribuição fica mais "pontiaguda", saída mais determinística e repetitiva); `T > 1` achata a distribuição (mais tokens ficam com chance razoável, saída mais criativa/aleatória, mas também mais propensa a erros). `T = 0` equivale a greedy (sempre o mais provável). Top-k e top-p resolvem um problema diferente de temperature: mesmo com temperature moderada, a "cauda" da distribuição (milhares de tokens improváveis, mas não zero) pode ocasionalmente ser amostrada, gerando texto bizarro — top-k corta essa cauda mantendo só os k tokens mais prováveis; top-p corta dinamicamente (mantém tokens até a probabilidade acumulada atingir p), o que se adapta melhor a distribuições ora muito concentradas, ora muito espalhadas.
>
> **Aplicação real**: geração de código ou extração estruturada tipicamente usa temperature baixa (0-0.3, precisão importa mais que criatividade); brainstorming ou escrita criativa usa temperature mais alta (0.7-1.0). É por isso que a maioria das APIs de LLM expõe `temperature` como o parâmetro mais visível ao usuário — é o dial mais direto entre "confiável e previsível" e "criativo e variado".
>
> **Checkpoint**: sem olhar o texto, explique o que acontece com a distribuição de probabilidade quando `T > 1` vs `T < 1`. Depois, explique a diferença entre top-k e top-p.

### Constrained generation
- **Grammars** (GBNF em llama.cpp) — força saída em formato específico.
- **Outlines**, **Guidance**, **JSON Schema** mode.
- **Logit bias** — manipular probabilidades de tokens específicos.

### Referências
- `Paper` **The Curious Case of Neural Text Degeneration (top-p)** — Holtzman et al. (2019). https://arxiv.org/abs/1904.09751
- `Ferramenta` **Outlines** — structured generation. https://github.com/dottxt-ai/outlines

---

## Projetos práticos

### Projeto 10.1 — Quantizar e comparar
- Pegue LLaMA 3 8B (ou Qwen 2.5 7B).
- Quantize em Q8, Q5_K_M, Q4_K_M, Q3_K_M via llama.cpp.
- Compare em 30 prompts: latência, uso de memória, qualidade subjetiva.
- Documente o "ponto de quebra" da qualidade.

> **Variante guiada**: use os mesmos 30 prompts em todas as quantizações, na mesma ordem, com a mesma seed — a única variável mudando deve ser o nível de quantização, senão a comparação mistura ruído de amostragem com o efeito real da quantização.

### Projeto 10.2 — Servidor local com Ollama + cliente em TS
- Rode Ollama com 2 modelos.
- Cliente em Node/Bun via `ai` SDK (Vercel) apontando para `http://localhost:11434`.
- Implemente chat com streaming.

### Projeto 10.3 — vLLM em GPU + benchmark
- Suba modelo via vLLM (em servidor cloud com GPU, ou local).
- Carregue e meça throughput com 1, 4, 16, 64 requests concorrentes.
- Compare com llama.cpp / Ollama.

### Projeto 10.4 — Distilar para tarefa específica
- Tarefa: extração estruturada de dados (ex: JSON de descrições de produto).
- Teacher: modelo grande (via API ou local 70B+).
- Gere ~5k pares (input, output).
- Treine modelo pequeno (Phi-3 mini ou Qwen 0.5B) com SFT/QLoRA.
- Compare qualidade e latência.

### Projeto 10.5 — LLM no browser
- Use `transformers.js` ou `web-llm`.
- Faça uma demo: classificação de texto ou chat com modelo pequeno.
- Meça tempo de carregamento, tempo até primeiro token.

### Projeto 10.6 — Speculative decoding manual
- Modelo grande: LLaMA 3 70B (cloud).
- Modelo pequeno: LLaMA 3 8B.
- Implemente loop: pequeno gera, grande valida.
- Compare velocidade vs grande sozinho.

> **Variante guiada**: confirme primeiro que a saída final (com speculative decoding) é *idêntica* à saída do modelo grande sozinho, na mesma seed — só depois meça o ganho de velocidade. Speculative decoding que muda a saída indica bug de implementação, não uma otimização válida.

### Projeto 10.7 — Constrained generation
- Use `outlines` (Python) ou `llama.cpp` GBNF.
- Force JSON com schema específico.
- Compare confiabilidade vs prompt de "responda em JSON".

---

## Erros comuns

- **Quantizar excessivamente** modelos pequenos — degradação grande. Q4 em 70B ≠ Q4 em 1B.
- **Esquecer KV-cache** ao implementar inferência manual — fica 100× mais lento.
- **Comparar latências** sem warmup — primeiros tokens incluem inicialização.
- **Não controlar `seed` e `temperature`** ao avaliar qualidade — comparações são ruído.
- **Achar que CPU = lento sempre** — modelos pequenos quantizados em CPU moderna são úteis.
- **Misturar contextos** em servidor sem prompt cache adequado.

---

## Conexão com módulos seguintes

| Conceito daqui | Aparece em |
|---|---|
| Servidor local | RAG (mod. [12](12_rag.mdx)), Agentes (mod. [13](13_agentes_tools_protocolos.md)), Produção (mod. [15](15_engenharia_producao.mdx)) |
| Quantização | Deploy mobile (mod. [18](18_multimodal.mdx)), edge |
| Sampling | Prompt engineering (mod. [11](11_prompt_engineering.md)), avaliação (mod. [14](14_avaliacao_e_seguranca.md)) |
| Constrained generation | Tools/structured output (mod. [13](13_agentes_tools_protocolos.md)) |

---

## Checklist de saída

- [ ] Tenho um modelo open rodando localmente, sem API externa.
- [ ] Sei quantizar um modelo em GGUF/AWQ/GPTQ.
- [ ] Implementei um servidor local que serve API compatível OpenAI.
- [ ] Sei usar vLLM (ou TGI/SGLang) para servir LLM com throughput alto.
- [ ] Domino parâmetros de sampling e sei quando usar cada um.
- [ ] Posso fazer inferência client-side (browser ou Node) via transformers.js ou web-llm.
- [ ] Entendo trade-offs de quantização, distillation e MoE conceitualmente.
