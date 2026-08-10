---
id: 09_treinamento_e_alinhamento
title: "Módulo 09 — Treinamento e Alinhamento de LLMs"
sidebar_position: 9
---

# Módulo 09 — Treinamento e Alinhamento de LLMs

> **Objetivo**: entender o ciclo completo de criação de uma LLM moderna — do pré-treinamento bruto ao modelo "instruct/chat" alinhado, incluindo SFT, DPO, RLHF e variações.
>
> **Pré-requisitos**: Módulos 07–08.
>
> **Tempo de referência**: 4–6 semanas.

---

## Por que isso importa

Modelos pré-treinados puros são *completion engines*, não assistentes. O comportamento útil ("siga instrução", "recuse pedidos perigosos", "use formato X") é construído em **etapas pós-treinamento**. Entender essas etapas é essencial para:

- Escolher o modelo certo (base vs instruct).
- Fazer fine-tuning sem destruir o modelo (catastrophic forgetting).
- Avaliar trade-offs de safety, performance, custo.

---

## 9.1 Pipeline completo de uma LLM moderna

```
Dados brutos
    ↓
Pré-processamento (filtros, dedup, tokenização)
    ↓
PRÉ-TREINAMENTO (next-token prediction em larga escala)
    → Modelo BASE (ex: LLaMA-3-8B-base)
    ↓
SUPERVISED FINE-TUNING (SFT) com instruction data
    → Modelo SFT
    ↓
PREFERENCE OPTIMIZATION (RLHF / DPO / GRPO / outras)
    → Modelo INSTRUCT/CHAT (ex: LLaMA-3-8B-Instruct)
    ↓
(opcional) Continued pretraining em domínio
    → Modelo especializado
    ↓
(opcional) Reasoning RL (R1-style)
    → Modelo de raciocínio
```

---

## 9.2 Pré-treinamento em escala

### Hardware
- **Centenas a milhares de GPUs** (H100, MI300X) para modelos da escala atual.
- **Interconexão** (NVLink, InfiniBand) é gargalo crítico.
- **Custos**: dezenas de milhões de USD para um modelo grande.

### Paralelismo
- **Data Parallel (DP)**: cada GPU vê batch diferente.
- **Distributed Data Parallel (DDP)** — padrão.
- **Fully Sharded Data Parallel (FSDP)** — particiona pesos, gradientes e estados de otimizador.
- **Tensor Parallel (TP)** — particiona matrizes de atenção/FFN entre GPUs.
- **Pipeline Parallel (PP)** — divide camadas em estágios.
- **Sequence Parallel** — para contextos muito longos.
- **3D parallelism** — combinação de DP + TP + PP.

### Frameworks
- **PyTorch FSDP**.
- **DeepSpeed** (Microsoft) — ZeRO 1/2/3.
- **Megatron-LM** (NVIDIA) — TP + PP otimizados.
- **JAX + TPU** — usado pelo Google (T5, Gemma).
- **Hugging Face Accelerate** — abstração mais alta sobre FSDP/DeepSpeed.

### Referências
- 📄 **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models** — Rajbhandari et al. (2019). https://arxiv.org/abs/1910.02054
- 📄 **Megatron-LM** — Shoeybi et al. (2019). https://arxiv.org/abs/1909.08053
- 📄 **PaLM Technical Report** — DeepMind/Google (2022). https://arxiv.org/abs/2204.02311
- 📚 **The Ultra-Scale Playbook** (Hugging Face, 2024–2025) — guia profundo de paralelismo. https://huggingface.co/spaces/nanotron/ultrascale-playbook

### Cursos
- 🎓 **Stanford CS336 — Language Modeling from Scratch** (literalmente sobre construir LLMs do zero). https://stanford-cs336.github.io/

---

## 9.3 Supervised Fine-Tuning (SFT)

### Objetivo
Pegar um modelo BASE e ensinar a seguir o formato **instruction → response**.

### Dados
- **Instruction datasets**: pares (prompt, resposta) — sintéticos ou humanos.
- **Diálogos multi-turno**: conversas completas.

### Datasets abertos
- **Alpaca** (Stanford, sintético com GPT-3.5). https://crfm.stanford.edu/2023/03/13/alpaca.html
- **Dolly 15k** (Databricks, humano). https://huggingface.co/datasets/databricks/databricks-dolly-15k
- **OpenAssistant Conversations (OASST)**. https://arxiv.org/abs/2304.07327
- **UltraChat / UltraFeedback** (Tsinghua + UCL).
- **Tulu V2/V3** (AI2).
- **Open-Orca**, **SlimOrca**.
- **No Robots** (HuggingFaceH4).

### Técnicas
- **Loss masking**: calcular loss apenas nos tokens de resposta, não de prompt.
- **Packing**: empacotar múltiplas amostras em uma sequência para usar GPU.
- **Mixing**: balancear domínios (chat, código, math).

### Referências
- 📄 **Alpaca: A Strong, Replicable Instruction-Following Model**. https://crfm.stanford.edu/2023/03/13/alpaca.html
- 📄 **Self-Instruct: Aligning Language Models with Self-Generated Instructions** — Wang et al. (2022). https://arxiv.org/abs/2212.10560
- 📄 **The Tulu 3 Recipe** (AI2, 2024) — receita aberta completa de pós-treinamento. https://arxiv.org/abs/2411.15124

---

## 9.4 Parameter-Efficient Fine-Tuning (PEFT)

### Por que
Fine-tuning completo de um modelo de 70B requer ~1 TB de GPU. PEFT adiciona/treina apenas uma fração pequena dos parâmetros.

### Métodos
- **LoRA (Low-Rank Adaptation)** — adiciona matrizes de baixa-rank A·B aos pesos. 📄 https://arxiv.org/abs/2106.09685
- **QLoRA** — LoRA + quantização do modelo base em 4-bit. Permite fine-tunar 65B em uma GPU de 48GB. 📄 https://arxiv.org/abs/2305.14314
- **DoRA** — Weight-Decomposed LoRA (mais expressivo). 📄 https://arxiv.org/abs/2402.09353
- **Prefix Tuning**, **P-Tuning v2**, **Prompt Tuning** (treinam prefixos no input).
- **(IA)³**, **AdaLoRA** — variantes.

### Ferramentas
- 🛠 **Hugging Face PEFT**. https://huggingface.co/docs/peft
- 🛠 **Unsloth** — fine-tuning otimizado, 2x mais rápido, 70% menos VRAM. https://github.com/unslothai/unsloth
- 🛠 **Axolotl** — config-driven fine-tuning. https://github.com/axolotl-ai-cloud/axolotl
- 🛠 **TRL** (Hugging Face) — training de reinforcement learning. https://huggingface.co/docs/trl

---

## 9.5 Alinhamento via preferências

### O problema
SFT ensina formato. Mas como ensinar **bom** vs **ruim** quando "qualidade" é subjetiva?
**Solução**: aprender com pares de preferência (resposta_A preferida sobre resposta_B).

### RLHF (Reinforcement Learning from Human Feedback)
1. Coletar preferências humanas (pares ranqueados).
2. Treinar **Reward Model (RM)**: dado (prompt, resposta), prediz score de preferência.
3. Treinar política via RL (geralmente **PPO**) usando o RM como recompensa.
4. KL-penalty contra modelo SFT inicial para evitar "reward hacking".

### Papers fundamentais
- 📄 **Deep Reinforcement Learning from Human Preferences** — Christiano et al. (2017). https://arxiv.org/abs/1706.03741
- 📄 **InstructGPT (Training language models to follow instructions with human feedback)** — Ouyang et al. (2022). https://arxiv.org/abs/2203.02155
- 📄 **Constitutional AI: Harmlessness from AI Feedback** — Anthropic (Bai et al., 2022). https://arxiv.org/abs/2212.08073
- 📄 **Llama 2 paper** — descreve RLHF em detalhe. https://arxiv.org/abs/2307.09288

### Direct Preference Optimization (DPO) e variantes
RLHF é caro e instável. DPO transforma o problema em uma loss supervisionada equivalente (matematicamente derivada do PPO + KL).

- 📄 **Direct Preference Optimization (DPO)** — Rafailov et al. (2023). https://arxiv.org/abs/2305.18290
- 📄 **IPO** — Identity Preference Optimization. https://arxiv.org/abs/2310.12036
- 📄 **KTO (Kahneman-Tversky Optimization)** — não exige pares, só "good/bad". https://arxiv.org/abs/2402.01306
- 📄 **ORPO** — Monolithic Preference Optimization sem reference model. https://arxiv.org/abs/2403.07691
- 📄 **GRPO (Group Relative Policy Optimization)** — usado no DeepSeek-R1, sem value model. https://arxiv.org/abs/2402.03300

### Referências práticas
- 🛠 **TRL (Hugging Face)** — implementações de PPO, DPO, KTO, ORPO. https://huggingface.co/docs/trl
- 📚 **Hugging Face — Alignment Handbook**. https://github.com/huggingface/alignment-handbook
- 📚 **Tulu 3** — receita pública e detalhada de SFT + DPO + GRPO. https://arxiv.org/abs/2411.15124

---

## 9.6 Reasoning RL (estilo R1)

### Conceito
Treinar puramente com RL sobre tarefas verificáveis (matemática, código), sem reward model. Recompensa = "resposta correta".

### Característica
- O modelo "descobre" chain-of-thought longo sozinho.
- Funciona com modelos médios e gera ganhos enormes em raciocínio.

### Papers
- 📄 **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning**. https://arxiv.org/abs/2501.12948
- 📄 **DeepSeekMath** (introduz GRPO). https://arxiv.org/abs/2402.03300

---

## 9.7 Continued Pretraining (CPT)

### Quando usar
- Adaptar modelo a domínio específico (médico, jurídico, código de empresa) com **muito** texto não-instruct.
- Adaptar a idioma sub-representado.

### Cuidados
- **Catastrophic forgetting**: rodar CPT extenso pode destruir o conhecimento geral.
- Mitigar com **mistura de dados**: 80% domínio novo + 20% genérico.
- Learning rate baixo, warmup curto.

### Referências
- 📄 **Continued Pre-Training of Large Language Models: How to (re)warm your model?** https://arxiv.org/abs/2308.04014
- 📄 **Don't Stop Pretraining** — Gururangan et al. (2020). https://arxiv.org/abs/2004.10964

---

## 9.8 Avaliação durante o treinamento

- **Loss de validação** (mas não basta).
- **Perplexity** em conjuntos held-out por domínio.
- **Benchmarks** (MMLU, GSM8K, HumanEval, etc. — ver mod. 14).
- **Eval qualitativa** (rodadas com prompts próprios).
- **LM-Eval-Harness** (EleutherAI). https://github.com/EleutherAI/lm-evaluation-harness

---

## 🧪 Projetos práticos

### Projeto 9.1 — SFT em modelo pequeno
- Modelo: TinyLlama 1.1B ou Phi-3 mini.
- Dataset: Alpaca (subset) ou seu próprio dataset de instruções.
- Use Unsloth + QLoRA em GPU única (Colab ou local com 8–16 GB).
- Compare antes/depois em prompts próprios.

### Projeto 9.2 — DPO em cima do projeto 9.1
- Crie ~500 pares de preferência (manual ou com modelo "juiz" mais forte).
- Aplique DPO via TRL.
- Compare DPO vs SFT puro.

### Projeto 9.3 — CPT em PT-BR
- Pegue um modelo base (Gemma 2B, Qwen 2.5 1.5B).
- Faça continued pretraining em corpus PT-BR (Wikipedia em português, OSCAR-PT).
- Avalie qualidade em PT-BR antes/depois.
- Documente catastrophic forgetting (testando inglês/código).

### Projeto 9.4 — Reproduzir pipeline pequeno do Tulu
- Siga o **Alignment Handbook** ou a receita do Tulu 3.
- Modelo: pequeno (≤7B), GPU única.
- Faça SFT + DPO end-to-end.
- Avalie em LM-Eval-Harness.

### Projeto 9.5 — Reasoning RL minimalista
- Use TRL com GRPO.
- Tarefa: aritmética simples ou problemas verificáveis pequenos.
- Recompensa: parser de resposta correta.
- Observe se "chain of thought" emerge.

---

## ⚠️ Erros comuns

- **Treinar SFT em formato errado** — cada modelo tem seu chat template (ChatML, Llama-3, Gemma, etc.). Não respeitar isso destrói performance.
- **Misturar BOS/EOS errados** — bug silencioso comum.
- **Loss em todos os tokens** em vez de só na resposta — fine-tuning ruim.
- **DPO sem reference model bem ajustado** — colapso de comportamento.
- **Achar que catastrophic forgetting não é real** — sempre teste capacidades antigas após CPT.
- **Avaliar só em "vibe check"** — sem benchmarks objetivos, é folclore.

---

## Conexão com módulos seguintes

| Conceito daqui | Aparece em |
|---|---|
| LoRA, QLoRA | Eficiência (mod. 10), customização (mod. 13) |
| Chat templates | Prompt engineering (mod. 11), Agentes (mod. 13) |
| RLHF/DPO mindset | Avaliação (mod. 14), safety |
| Reasoning RL | Agentes complexos (mod. 13) |

---

## Checklist de saída

- [ ] Sei distinguir base, SFT e instruct/chat de um mesmo modelo.
- [ ] Apliquei LoRA/QLoRA com sucesso em modelo ≥3B.
- [ ] Apliquei DPO em cenário real (mesmo que pequeno).
- [ ] Sei o que é GRPO e quando faz sentido.
- [ ] Entendo paralelismo (DDP, FSDP, TP, PP) suficientemente para escolher estratégia.
- [ ] Sei o que muda entre SFT, RLHF, DPO, KTO conceitualmente.
