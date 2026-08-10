# Módulo 10 — Eficiência e Inferência Local

> **Objetivo**: rodar e otimizar LLMs em hardware modesto (laptop, GPU consumer, CPU, edge). Quantização, distilação, KV-cache, servidores de inferência, edge inference.
>
> **Pré-requisitos**: Módulos 07–09.
>
> **Tempo de referência**: 3–5 semanas.

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
- **GPTQ** — quantização inteligente baseada em segunda ordem. Bom para GPU. 📄 https://arxiv.org/abs/2210.17323
- **AWQ (Activation-aware Weight Quantization)** — usa estatísticas de ativação. 📄 https://arxiv.org/abs/2306.00978
- **bitsandbytes** — quantização 8-bit/4-bit em PyTorch.
- **FP8** (training-time, em H100/MI300).

### Regra de bolso
Para um modelo de **N** bilhões de parâmetros, VRAM aproximada para inferência:
- FP16 / BF16: 2N GB
- INT8: N GB
- Q4 (GGUF Q4_K_M): ~0.6N GB
- Q3: ~0.45N GB
(Adicione ~1–2 GB para KV-cache e overhead.)

Exemplo: LLaMA-3 8B em Q4_K_M cabe folgado em GPU de 8 GB.

### Trade-offs
- **Q4_K_M** é o "sweet spot" comum: ~99% da qualidade do FP16 em muitos casos.
- **Q3** já degrada perceptivelmente em modelos pequenos.
- **Q8** é praticamente lossless mas dobra o footprint vs Q4.

### Referências
- 📄 **LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale** — Dettmers et al. (2022). https://arxiv.org/abs/2208.07339
- 📄 **GPTQ** — Frantar et al. (2022). https://arxiv.org/abs/2210.17323
- 📄 **AWQ** — Lin et al. (2023). https://arxiv.org/abs/2306.00978
- 📄 **A Survey of Quantization Methods for Efficient Neural Network Inference**. https://arxiv.org/abs/2103.13630
- 📚 **Hugging Face — Quantization documentation**. https://huggingface.co/docs/transformers/main/en/quantization/overview

---

## 10.2 KV-Cache e otimizações de inferência

### KV-Cache
Em geração autoregressiva, K e V de tokens passados não mudam. Cachear evita recomputar.

### Otimizações modernas
- **PagedAttention** (vLLM) — gerência de KV-cache estilo memória virtual. 📄 https://arxiv.org/abs/2309.06180
- **Continuous Batching** — agrupa requisições em diferentes estágios. Padrão em vLLM, TGI.
- **Speculative Decoding** — modelo pequeno propõe tokens, modelo grande valida. 📄 https://arxiv.org/abs/2211.17192
- **Flash Decoding** — variante de Flash Attention para inferência.
- **Prompt Caching** — cachear prefixos comuns entre chamadas.

### Referências
- 📄 **Efficient Memory Management for LLM Serving with PagedAttention (vLLM)** — Kwon et al. (2023). https://arxiv.org/abs/2309.06180
- 📄 **Speculative Decoding** — Leviathan et al. (2022). https://arxiv.org/abs/2211.17192

---

## 10.3 Servidores de inferência

### Locais (uma máquina)
- 🛠 **Ollama** — UX simples, GGUF, Mac/Linux/Windows. https://ollama.com/
- 🛠 **llama.cpp** — backend C++ rápido, base do Ollama, controla detalhes. https://github.com/ggml-org/llama.cpp
- 🛠 **LM Studio** — GUI desktop. https://lmstudio.ai/
- 🛠 **GPT4All**. https://www.nomic.ai/gpt4all
- 🛠 **Jan**. https://jan.ai/

### Servidor (alta performance, multi-usuário)
- 🛠 **vLLM** — padrão para servir LLMs em GPU. https://docs.vllm.ai/
- 🛠 **TGI (Text Generation Inference)** (Hugging Face). https://github.com/huggingface/text-generation-inference
- 🛠 **TensorRT-LLM** (NVIDIA). Performance máxima em hardware NVIDIA.
- 🛠 **SGLang** — alternativa moderna com runtime próprio. https://github.com/sgl-project/sglang
- 🛠 **MLC-LLM** — compila modelos para múltiplos backends (CUDA, Metal, Vulkan, WebGPU). https://github.com/mlc-ai/mlc-llm

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
- 🛠 **transformers.js** — ONNX runtime no browser/Node. https://huggingface.co/docs/transformers.js
- 🛠 **MLC-LLM Web** — modelos rodando via WebGPU. https://github.com/mlc-ai/web-llm
- 🛠 **Apple MLX** — framework Apple para Apple Silicon. https://github.com/ml-explore/mlx
- 🛠 **MediaPipe LLM** (Google). https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference

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
- **DistilBERT** (40% menor que BERT, 97% performance). 📄 https://arxiv.org/abs/1910.01108
- **TinyLlama** (1.1B, treinado long).
- **Phi-3** (filosofia: dataset sintético de alta qualidade do GPT-4).
- **Gemma 2** (destilação de Gemini).

### Quando vale a pena
- Você tem teacher (próprio ou via API barata).
- Tarefa específica (classificação, extração, conversão de formato).
- Precisa de inferência rápida em produção.

### Referências
- 📄 **Distilling the Knowledge in a Neural Network** — Hinton, Vinyals, Dean (2015). https://arxiv.org/abs/1503.02531
- 📄 **MiniLLM: Knowledge Distillation of Large Language Models** — Gu et al. (2023). https://arxiv.org/abs/2306.08543
- 📄 **Phi-3 Technical Report**. https://arxiv.org/abs/2404.14219

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

### Referências
- 📄 **The Lottery Ticket Hypothesis** — Frankle & Carbin (2018). https://arxiv.org/abs/1803.03635
- 📄 **SparseGPT** — Frantar & Alistarh (2023). https://arxiv.org/abs/2301.00774
- 📄 **Wanda: A Simple and Effective Pruning Approach for LLMs**. https://arxiv.org/abs/2306.11695

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

### Constrained generation
- **Grammars** (GBNF em llama.cpp) — força saída em formato específico.
- **Outlines**, **Guidance**, **JSON Schema** mode.
- **Logit bias** — manipular probabilidades de tokens específicos.

### Referências
- 📄 **The Curious Case of Neural Text Degeneration (top-p)** — Holtzman et al. (2019). https://arxiv.org/abs/1904.09751
- 🛠 **Outlines** — structured generation. https://github.com/dottxt-ai/outlines

---

## 🧪 Projetos práticos

### Projeto 10.1 — Quantizar e comparar
- Pegue LLaMA 3 8B (ou Qwen 2.5 7B).
- Quantize em Q8, Q5_K_M, Q4_K_M, Q3_K_M via llama.cpp.
- Compare em 30 prompts: latência, uso de memória, qualidade subjetiva.
- Documente o "ponto de quebra" da qualidade.

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

### Projeto 10.7 — Constrained generation
- Use `outlines` (Python) ou `llama.cpp` GBNF.
- Force JSON com schema específico.
- Compare confiabilidade vs prompt de "responda em JSON".

---

## ⚠️ Erros comuns

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
| Servidor local | RAG (mod. 12), Agentes (mod. 13), Produção (mod. 15) |
| Quantização | Deploy mobile (mod. 18), edge |
| Sampling | Prompt engineering (mod. 11), avaliação (mod. 14) |
| Constrained generation | Tools/structured output (mod. 13) |

---

## Checklist de saída

- [ ] Tenho um modelo open rodando localmente, sem API externa.
- [ ] Sei quantizar um modelo em GGUF/AWQ/GPTQ.
- [ ] Implementei um servidor local que serve API compatível OpenAI.
- [ ] Sei usar vLLM (ou TGI/SGLang) para servir LLM com throughput alto.
- [ ] Domino parâmetros de sampling e sei quando usar cada um.
- [ ] Posso fazer inferência client-side (browser ou Node) via transformers.js ou web-llm.
- [ ] Entendo trade-offs de quantização, distillation e MoE conceitualmente.
