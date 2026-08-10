# Módulo 11 — Prompt Engineering

> **Objetivo**: dominar a arte e ciência de instruir LLMs. Não é "truque mágico"; é interface técnica entre intenção humana e modelo. Inclui CoT, ToT, ReAct, structured output, prompt injection.
>
> **Pré-requisitos**: Módulos 07–10.
>
> **Tempo de referência**: 2–3 semanas.

---

## Por que isso importa

Um LLM bem prompted é, em muitos casos, melhor que um LLM mal fine-tuned. E **toda** aplicação real de LLM passa por engenharia de prompt — mesmo após fine-tuning. Mais importante: prompt é um **vetor de ataque** (prompt injection); ignorar isso em produção é negligência.

---

## 11.1 Anatomia de um prompt moderno

### Componentes
- **System prompt**: instruções persistentes (papel, regras, formato).
- **User message**: pedido específico.
- **Few-shot examples**: pares (input, output) demonstrativos.
- **Context** (RAG): documentos relevantes injetados.
- **Output schema** ou format hint.

### Chat templates
Cada modelo tem seu **chat template** específico (ChatML, Llama-3, Gemma, Mistral, Qwen, ...). Usar template errado = qualidade despenca silenciosamente.

```
# ChatML (OpenAI, Qwen, ModernBERT-Chat):
<|im_start|>system
Você é um assistente útil.<|im_end|>
<|im_start|>user
Olá<|im_end|>
<|im_start|>assistant
```

```
# Llama 3:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
...
<|eot_id|><|start_header_id|>user<|end_header_id|>
```

Ferramentas como `tokenizer.apply_chat_template()` (Hugging Face) abstraem isso.

---

## 11.2 Técnicas fundamentais

### Zero-shot
"Resolva X" sem exemplos. Sufficiente para tarefas comuns em modelos grandes.

### Few-shot prompting
Mostrar 1–10 exemplos. Útil quando:
- Formato de saída específico.
- Tarefa rara/idiossincrática.
- Modelo menor.

### Instruções claras
- **Verbos imperativos** ("Liste", "Resuma em 3 frases", "Responda apenas em JSON").
- **Restrições explícitas**: "Não invente fontes", "Use apenas as informações entre <doc>".
- **Critérios de qualidade**: "Seja conciso", "Use linguagem técnica".

### Persona / Role
"Você é um revisor técnico sênior..." — melhora consistência, mas evite usar como muleta para prompts mal estruturados.

### Referências
- 📄 **Language Models are Few-Shot Learners (GPT-3)** — Brown et al. (2020). https://arxiv.org/abs/2005.14165
- 📚 **Prompt Engineering Guide** (DAIR.AI). https://www.promptingguide.ai/
- 📄 **OpenAI — Prompt engineering best practices** (parte da documentação oficial).

---

## 11.3 Chain-of-Thought (CoT) e variações

### Chain-of-Thought
Pedir para o modelo "pensar passo a passo" antes da resposta. Melhora drasticamente raciocínio em modelos grandes.

- 📄 **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models** — Wei et al. (2022). https://arxiv.org/abs/2201.11903
- 📄 **Large Language Models are Zero-Shot Reasoners** ("Let's think step by step") — Kojima et al. (2022). https://arxiv.org/abs/2205.11916

### Self-Consistency
Gerar várias CoTs com `temperature > 0`, votar na resposta mais comum.
- 📄 **Self-Consistency Improves CoT Reasoning** — Wang et al. (2022). https://arxiv.org/abs/2203.11171

### Tree-of-Thought (ToT)
Explorar múltiplos caminhos como árvore, com avaliação intermediária.
- 📄 **Tree of Thoughts: Deliberate Problem Solving with LLMs** — Yao et al. (2023). https://arxiv.org/abs/2305.10601

### Graph-of-Thought, Skeleton-of-Thought
Variantes para casos específicos.

### Quando *não* usar CoT
- Modelos com reasoning interno (DeepSeek-R1, o1-style) já fazem CoT internamente; CoT explícito pode atrapalhar.
- Tarefas simples — overhead de tokens sem benefício.

---

## 11.4 ReAct: raciocínio + ação

📄 **ReAct: Synergizing Reasoning and Acting in Language Models** — Yao et al. (2022). https://arxiv.org/abs/2210.03629

Padrão:
```
Thought: preciso verificar X
Action: search("X")
Observation: ...
Thought: agora sei Y, próximo passo...
```

Foundation conceitual de **agentes** (mod. 13).

---

## 11.5 Structured Output

### Por que
Aplicações reais consomem JSON, não prosa. "Parsing prosa" é frágil.

### Técnicas
- **Prompt-only**: pedir JSON com schema in-prompt + few-shot. Frágil sem grammar.
- **Function calling / Tool use** (OpenAI, Anthropic, Mistral, etc.): API expõe schema, modelo retorna JSON estruturado.
- **JSON Mode** (em alguns provedores): garante saída JSON-válido.
- **Constrained decoding**: força a saída a respeitar grammar/schema **token a token**. Ex: `outlines`, `llama.cpp` GBNF, `lm-format-enforcer`.
- **Pydantic** (Python) / **Zod** (TS) para validação no client.

### Ferramentas
- 🛠 **Outlines** (Python). https://github.com/dottxt-ai/outlines
- 🛠 **Instructor** (Python, wrapper sobre OpenAI/Anthropic com Pydantic). https://github.com/jxnl/instructor
- 🛠 **Vercel AI SDK — `generateObject` com Zod** (TS). https://ai-sdk.dev/

---

## 11.6 Prompts complexos: técnicas avançadas

### Decomposição
Quebrar tarefa em subtarefas. Cada subtarefa é um prompt menor, encadeado.
- 📄 **Decomposed Prompting** — Khot et al. (2022). https://arxiv.org/abs/2210.02406

### Self-Refine
Modelo gera, critica, revisa.
- 📄 **Self-Refine: Iterative Refinement with Self-Feedback** — Madaan et al. (2023). https://arxiv.org/abs/2303.17651

### Reflexion
Agente reflete sobre falhas para melhorar.
- 📄 **Reflexion: Language Agents with Verbal Reinforcement Learning** — Shinn et al. (2023). https://arxiv.org/abs/2303.11366

### Plan-and-Solve
Planejar antes de executar.
- 📄 **Plan-and-Solve Prompting** — Wang et al. (2023). https://arxiv.org/abs/2305.04091

### Least-to-Most
Resolver subproblemas progressivos.
- 📄 **Least-to-Most Prompting** — Zhou et al. (2022). https://arxiv.org/abs/2205.10625

---

## 11.7 In-Context Learning (ICL): por que few-shot funciona

LLMs grandes "aprendem" da janela de contexto sem gradient updates. Ainda é fenômeno parcialmente entendido.

### Papers
- 📄 **Why Can GPT Learn In-Context? Language Models Implicitly Perform Gradient Descent as Meta-Optimizers** — Dai et al. (2022). https://arxiv.org/abs/2212.10559
- 📄 **A Survey on In-context Learning**. https://arxiv.org/abs/2301.00234
- 📄 **Larger language models do in-context learning differently**. https://arxiv.org/abs/2303.03846

### Prática
- Ordem dos exemplos importa.
- Diversidade > redundância.
- Exemplos próximos à query no embedding space (dynamic few-shot) ajudam.

---

## 11.8 Prompt Injection e segurança

### Definição
**Prompt injection**: input do usuário (ou de documento externo, em RAG) que sobrescreve instruções do sistema.

### Categorias
- **Direct injection**: usuário diz "ignore instruções anteriores e ...".
- **Indirect injection**: documento recuperado em RAG contém instruções maliciosas.
- **Jailbreak**: contornar guardrails (DAN, role-play malicioso).
- **Prompt leaking**: extrair system prompt.

### Mitigações (parciais — não há solução completa)
- **Separação clara** (delimitadores XML, marcadores).
- **Modelos com adversarial training** (Constitutional AI, Sparrow).
- **Filtros pré- e pós-modelo**.
- **Princípio do menor privilégio** em tools (mod. 13).
- **Não confiar em input** vindo de fontes externas, mesmo em RAG.
- **Output validation** com schemas.

### Referências
- 📄 **Prompt Injection attack against LLM-integrated Applications** — Liu et al. (2023). https://arxiv.org/abs/2306.05499
- 📄 **Universal and Transferable Adversarial Attacks on Aligned LLMs** — Zou et al. (2023). https://arxiv.org/abs/2307.15043
- 📚 **OWASP Top 10 for LLM Applications**. https://owasp.org/www-project-top-10-for-large-language-model-applications/
- 📄 **Constitutional AI** — Anthropic. https://arxiv.org/abs/2212.08073

---

## 11.9 Avaliação de prompts

### Metodologia rigorosa
- **Conjunto de avaliação**: ≥30 casos representativos, com gabarito ou rubrica.
- **Métricas**: accuracy, exact match, qualitative rating, LLM-as-judge.
- **A/B test** entre versões de prompt.
- **Versionamento de prompts** (eles mudam tanto quanto código).

### Ferramentas
- 🛠 **Promptfoo** — testing CLI/CI para prompts. https://www.promptfoo.dev/
- 🛠 **LangSmith / Langfuse** — observability + eval. (Mais em mod. 15.)
- 🛠 **OpenAI Evals**. https://github.com/openai/evals
- 🛠 **DSPy** (Stanford) — abordagem programática para prompts otimizáveis. https://github.com/stanfordnlp/dspy

### Referências
- 📄 **DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines** — Khattab et al. (2023). https://arxiv.org/abs/2310.03714

---

## 11.10 Diferenças entre modelos

Cada família responde melhor a estilos diferentes:
- **Claude**: prefere XML para estruturação ("envolva em `<example>...</example>`").
- **GPT-4/o-series**: bom com Markdown e instruções diretas.
- **LLaMA-3 / Mistral**: chat template estrito; sensível a temperatura.
- **Gemini**: bom com prompts longos e multimodais.

Não exista prompt "ótimo universal". Avalie por modelo.

---

## 🧪 Projetos práticos

### Projeto 11.1 — Suite de testes de prompts
- Tarefa: extração estruturada de informações de receitas (ingredientes, modo de preparo, tempo).
- Crie 30 receitas reais como conjunto de teste.
- Compare 4 prompts: zero-shot, few-shot (3 ex), CoT, structured output via Outlines/Instructor.
- Use Promptfoo ou suite própria para benchmarking.

### Projeto 11.2 — CoT vs reasoning model
- Mesma bateria (GSM8K reduzido, ou problemas próprios).
- Compare: LLaMA-3-8B sem CoT, LLaMA-3-8B com CoT, DeepSeek-R1-Distill-7B.
- Documente casos onde CoT explícito **piora** desempenho do reasoning model.

### Projeto 11.3 — ReAct manual
- Implemente loop ReAct em Python ou TS, sem framework de agente.
- Forneça 2 "tools" simples (calculadora, busca).
- Tarefa: questões de QA que exigem cálculo + lookup.
- Loop: parsear `Thought:`/`Action:` da saída do modelo, executar, devolver `Observation:`.

### Projeto 11.4 — Defesa contra prompt injection
- Implemente um pipeline RAG simples vulnerável.
- Crie 10 documentos maliciosos.
- Teste 3 mitigações: delimitadores XML, separação de instruções, post-validation.
- Documente o que funciona, o que não.

### Projeto 11.5 — Structured output rigoroso
- Tarefa: parse de currículos PDF para JSON com schema fixo.
- Compare 3 abordagens:
  - Prompt + few-shot, parse manual.
  - Function calling / tool use.
  - Constrained decoding com Outlines.
- Meça % de saídas válidas e qualidade.

### Projeto 11.6 (avançado) — DSPy
- Reproduza um exemplo da documentação do DSPy (RAG ou multi-hop).
- Compare prompt manual vs prompt otimizado por DSPy.

---

## ⚠️ Erros comuns

- **Não usar chat template** correto do modelo. Bug silencioso, qualidade despenca.
- **Confiar em "responda em JSON"** sem grammar — modelo vaza prosa nas bordas.
- **Avaliar prompt em 5 casos** e declarar vitória.
- **Não versionar prompts** — debugar regressões fica impossível.
- **Subestimar prompt injection** em RAG e em assistentes que recebem texto externo.
- **Confundir CoT com pensamento real** — modelo pode gerar reasoning errado mas chegar à resposta certa, e vice-versa.

---

## Conexão com módulos seguintes

| Conceito daqui | Aparece em |
|---|---|
| ReAct | Agentes (mod. 13) |
| Structured output | Tools/MCP (mod. 13) |
| LLM-as-judge | Avaliação (mod. 14) |
| Prompt injection | Segurança (mod. 14), Produção (mod. 15) |
| DSPy / promptfoo | Engenharia de produção (mod. 15) |

---

## Checklist de saída

- [ ] Sei usar chat templates corretos por família de modelo.
- [ ] Construí pipeline com structured output confiável (Outlines/Instructor/Zod).
- [ ] Implementei ReAct manual sem framework.
- [ ] Tenho intuição sobre quando CoT ajuda e quando não.
- [ ] Testei e documentei vulnerabilidades de prompt injection no meu projeto.
- [ ] Versiono prompts e tenho conjunto de eval rodando em CI.
