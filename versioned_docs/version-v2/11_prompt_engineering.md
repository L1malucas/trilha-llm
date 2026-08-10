---
id: 11_prompt_engineering
title: "Módulo 11 — Prompt Engineering"
sidebar_position: 11
---

# Módulo 11 — Prompt Engineering

> **Objetivo**: dominar a arte e ciência de instruir LLMs. Não é "truque mágico"; é interface técnica entre intenção humana e modelo. Inclui CoT, ToT, ReAct, structured output, prompt injection.
>
> **Pré-requisitos**: Módulos [07](07_transformers.mdx)–[10](10_eficiencia_e_inferencia_local.md).
>
> **Tempo de referência**: 2–3 semanas.

## Objetivos de aprendizagem

Ao final deste módulo você deve ser capaz de:

- Montar um prompt com os componentes certos (system, few-shot, contexto, schema) para uma tarefa dada.
- Explicar por que Chain-of-Thought melhora raciocínio num modelo autoregressivo — o mecanismo, não só o efeito.
- Explicar por que in-context learning funciona sem gradient updates.
- Implementar structured output confiável, sabendo quando "pedir JSON" no prompt não é suficiente.
- Reconhecer e mitigar prompt injection, direto e indireto.

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

> Chat templates existem porque o modelo foi *treinado* (SFT, mod. [09](09_treinamento_e_alinhamento.mdx#93-supervised-fine-tuning-sft)) para reconhecer esses marcadores especiais como delimitadores de papel (system/user/assistant) — usar o template errado não é "só cosmético", é literalmente alimentar o modelo com uma estrutura que ele nunca viu no fine-tuning, degradando a qualidade de forma silenciosa e difícil de diagnosticar sem saber que esse é o problema.

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

> **Intuição**: um LLM decoder-only (mod. [07](07_transformers.mdx)) gera o próximo token condicionado em *tudo* que está no contexto até ali — few-shot examples funcionam porque cada exemplo (input, output) no prompt se torna parte desse contexto condicionante, deslocando a distribuição de probabilidade do próximo token na direção do padrão demonstrado. É por isso que formato importa mais que quantidade: 2 exemplos bem formatados, representativos e no formato exato desejado costumam superar 10 exemplos ruidosos ou inconsistentes entre si.
>
> **Checkpoint**: sem olhar o texto, explique por que few-shot prompting ajuda mais em modelos menores do que em modelos muito grandes.

### Referências
- `Paper` **Language Models are Few-Shot Learners (GPT-3)** — Brown et al. (2020). https://arxiv.org/abs/2005.14165
- `Livro` **Prompt Engineering Guide** (DAIR.AI). https://www.promptingguide.ai/
- `Paper` **OpenAI — Prompt engineering best practices** (parte da documentação oficial).

---

## 11.3 Chain-of-Thought (CoT) e variações

### Chain-of-Thought
Pedir para o modelo "pensar passo a passo" antes da resposta. Melhora drasticamente raciocínio em modelos grandes.

- `Paper` **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models** — Wei et al. (2022). https://arxiv.org/abs/2201.11903
- `Paper` **Large Language Models are Zero-Shot Reasoners** ("Let's think step by step") — Kojima et al. (2022). https://arxiv.org/abs/2205.11916

> **Intuição**: um modelo autoregressivo produz um token por vez, e cada token gerado passa a fazer parte do contexto para o próximo (mod. [07](07_transformers.mdx#72-self-attention--o-coração)). Isso significa que a *quantidade de computação* disponível para "pensar" antes de emitir a resposta final é literalmente proporcional a quantos tokens intermediários o modelo gera. Pedir resposta direta força o modelo a acertar um problema complexo "de cabeça", numa única passada; CoT dá ao modelo um espaço de rascunho — cada passo intermediário gerado vira contexto adicional que informa o próximo, permitindo o equivalente a "quebrar o problema em partes" dentro do próprio processo de geração. É por isso que CoT ajuda mais em problemas que exigem múltiplos passos lógicos encadeados, e ajuda pouco em perguntas de fato único.
>
> **Aplicação real**: essa mesma intuição — mais tokens de "pensamento" antes da resposta final = mais capacidade de raciocínio — é o que os modelos de raciocínio (DeepSeek-R1, mod. [08](08_llms_arquiteturas.md#85-modelos-de-raciocínio-reasoning)) internalizam via RL, gerando CoT muito mais longo e refinado do que um simples "pense passo a passo" no prompt conseguiria induzir.
>
> **Checkpoint**: sem olhar o texto, explique por que gerar passos intermediários pode melhorar a resposta final de um modelo autoregressivo — o que muda tecnicamente entre responder direto e responder após CoT?

### Self-Consistency
Gerar várias CoTs com `temperature > 0`, votar na resposta mais comum.
- `Paper` **Self-Consistency Improves CoT Reasoning** — Wang et al. (2022). https://arxiv.org/abs/2203.11171

### Tree-of-Thought (ToT)
Explorar múltiplos caminhos como árvore, com avaliação intermediária.
- `Paper` **Tree of Thoughts: Deliberate Problem Solving with LLMs** — Yao et al. (2023). https://arxiv.org/abs/2305.10601

### Graph-of-Thought, Skeleton-of-Thought
Variantes para casos específicos.

### Quando *não* usar CoT
- Modelos com reasoning interno (DeepSeek-R1, o1-style) já fazem CoT internamente; CoT explícito pode atrapalhar.
- Tarefas simples — overhead de tokens sem benefício.

---

## 11.4 ReAct: raciocínio + ação

`Paper` **ReAct: Synergizing Reasoning and Acting in Language Models** — Yao et al. (2022). https://arxiv.org/abs/2210.03629

Padrão:
```
Thought: preciso verificar X
Action: search("X")
Observation: ...
Thought: agora sei Y, próximo passo...
```

Foundation conceitual de **agentes** (mod. [13](13_agentes_tools_protocolos.md)).

> ReAct é CoT (seção 11.3) com um passo extra: em vez de só "pensar" internamente, o modelo intercala pensamento com **ações externas verificáveis** (chamar uma tool, buscar informação) — o "Observation" que volta de cada ação vira novo contexto para o próximo "Thought", ancorando o raciocínio em informação real em vez de só no que o modelo já sabia. É essa intercalação pensar→agir→observar→pensar que fundamenta como agentes (mod. 13) usam tools de forma confiável.

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

> **Intuição — por que "prompt-only" é frágil**: pedir "responda em JSON" no prompt é uma instrução que o modelo pode seguir ou não, token a token — nada tecnicamente impede o modelo de gerar um token que quebra o schema. Constrained decoding ataca o problema na raiz: a cada passo de geração, a distribuição de probabilidade sobre o próximo token é *mascarada* para permitir só os tokens que manteriam a saída sintaticamente válida conforme a grammar/schema — o modelo literalmente não consegue gerar um JSON malformado, porque a opção nem está disponível a cada passo. É a diferença entre "pedir educadamente" e "restringir estruturalmente".

### Ferramentas
- `Ferramenta` **Outlines** (Python). https://github.com/dottxt-ai/outlines
- `Ferramenta` **Instructor** (Python, wrapper sobre OpenAI/Anthropic com Pydantic). https://github.com/jxnl/instructor
- `Ferramenta` **Vercel AI SDK — `generateObject` com Zod** (TS). https://ai-sdk.dev/

---

## 11.6 Prompts complexos: técnicas avançadas

### Decomposição
Quebrar tarefa em subtarefas. Cada subtarefa é um prompt menor, encadeado.
- `Paper` **Decomposed Prompting** — Khot et al. (2022). https://arxiv.org/abs/2210.02406

### Self-Refine
Modelo gera, critica, revisa.
- `Paper` **Self-Refine: Iterative Refinement with Self-Feedback** — Madaan et al. (2023). https://arxiv.org/abs/2303.17651

### Reflexion
Agente reflete sobre falhas para melhorar.
- `Paper` **Reflexion: Language Agents with Verbal Reinforcement Learning** — Shinn et al. (2023). https://arxiv.org/abs/2303.11366

### Plan-and-Solve
Planejar antes de executar.
- `Paper` **Plan-and-Solve Prompting** — Wang et al. (2023). https://arxiv.org/abs/2305.04091

### Least-to-Most
Resolver subproblemas progressivos.
- `Paper` **Least-to-Most Prompting** — Zhou et al. (2022). https://arxiv.org/abs/2205.10625

---

## 11.7 In-Context Learning (ICL): por que few-shot funciona

LLMs grandes "aprendem" da janela de contexto sem gradient updates. Ainda é fenômeno parcialmente entendido.

### Papers
- `Paper` **Why Can GPT Learn In-Context? Language Models Implicitly Perform Gradient Descent as Meta-Optimizers** — Dai et al. (2022). https://arxiv.org/abs/2212.10559
- `Paper` **A Survey on In-context Learning**. https://arxiv.org/abs/2301.00234
- `Paper` **Larger language models do in-context learning differently**. https://arxiv.org/abs/2303.03846

> Esse é o mesmo fenômeno introduzido no mod. [04](04_ml_moderno.md#43-few-shot-e-meta-learning): nenhum peso do modelo muda durante ICL — é inferência pura, o "aprendizado" acontece só dentro da computação de attention sobre os exemplos no contexto. Uma das hipóteses mais discutidas (Dai et al., acima) é que o mecanismo de attention, processando os exemplos do prompt, executa algo estruturalmente parecido com um passo implícito de gradient descent — mas isso continua sendo uma área ativa de pesquisa, não um consenso fechado.

### Prática
- Ordem dos exemplos importa.
- Diversidade > redundância.
- Exemplos próximos à query no embedding space (dynamic few-shot) ajudam.

> **Checkpoint**: sem olhar o texto, explique a diferença entre few-shot learning que ajusta pesos (mod. 04) e in-context learning num LLM.

---

## 11.8 Prompt Injection e segurança

### Definição
**Prompt injection**: input do usuário (ou de documento externo, em RAG) que sobrescreve instruções do sistema.

### Categorias
- **Direct injection**: usuário diz "ignore instruções anteriores e ...".
- **Indirect injection**: documento recuperado em RAG contém instruções maliciosas.
- **Jailbreak**: contornar guardrails (DAN, role-play malicioso).
- **Prompt leaking**: extrair system prompt.

> **Intuição**: prompt injection é conceitualmente parecido com SQL injection — a raiz do problema é a mesma em ambos: **dados e instruções compartilham o mesmo canal**. Numa query SQL vulnerável, input do usuário concatenado sem escape pode virar comando; num prompt de LLM, todo o texto (instrução do sistema, contexto RAG, input do usuário) entra na mesma sequência de tokens, e o modelo não tem uma forma estruturalmente garantida de saber "isto é instrução confiável" vs "isto é dado a ser processado". Diferente de SQL injection, que tem soluções robustas (prepared statements separam dados de comando de forma estrutural), prompt injection **não tem solução estrutural equivalente ainda** — todas as mitigações listadas abaixo são camadas de defesa parciais, não uma correção definitiva.

### Mitigações (parciais — não há solução completa)
- **Separação clara** (delimitadores XML, marcadores).
- **Modelos com adversarial training** (Constitutional AI, Sparrow).
- **Filtros pré- e pós-modelo**.
- **Princípio do menor privilégio** em tools (mod. [13](13_agentes_tools_protocolos.md)).
- **Não confiar em input** vindo de fontes externas, mesmo em RAG.
- **Output validation** com schemas.

> **Checkpoint**: sem olhar o texto, explique o paralelo entre prompt injection e SQL injection — qual é a causa raiz compartilhada? Depois, explique por que "não há solução completa" para prompt injection, diferente de SQL injection.

### Referências
- `Paper` **Prompt Injection attack against LLM-integrated Applications** — Liu et al. (2023). https://arxiv.org/abs/2306.05499
- `Paper` **Universal and Transferable Adversarial Attacks on Aligned LLMs** — Zou et al. (2023). https://arxiv.org/abs/2307.15043
- `Livro` **OWASP Top 10 for LLM Applications**. https://owasp.org/www-project-top-10-for-large-language-model-applications/
- `Paper` **Constitutional AI** — Anthropic. https://arxiv.org/abs/2212.08073

---

## 11.9 Avaliação de prompts

### Metodologia rigorosa
- **Conjunto de avaliação**: ≥30 casos representativos, com gabarito ou rubrica.
- **Métricas**: accuracy, exact match, qualitative rating, LLM-as-judge.
- **A/B test** entre versões de prompt.
- **Versionamento de prompts** (eles mudam tanto quanto código).

### Ferramentas
- `Ferramenta` **Promptfoo** — testing CLI/CI para prompts. https://www.promptfoo.dev/
- `Ferramenta` **LangSmith / Langfuse** — observability + eval. (Mais em mod. [15](15_engenharia_producao.mdx).)
- `Ferramenta` **OpenAI Evals**. https://github.com/openai/evals
- `Ferramenta` **DSPy** (Stanford) — abordagem programática para prompts otimizáveis. https://github.com/stanfordnlp/dspy

### Referências
- `Paper` **DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines** — Khattab et al. (2023). https://arxiv.org/abs/2310.03714

---

## 11.10 Diferenças entre modelos

Cada família responde melhor a estilos diferentes:
- **Claude**: prefere XML para estruturação ("envolva em `<example>...</example>`").
- **GPT-4/o-series**: bom com Markdown e instruções diretas.
- **LLaMA-3 / Mistral**: chat template estrito; sensível a temperatura.
- **Gemini**: bom com prompts longos e multimodais.

Não exista prompt "ótimo universal". Avalie por modelo.

---

## Projetos práticos

### Projeto 11.1 — Suite de testes de prompts
- Tarefa: extração estruturada de informações de receitas (ingredientes, modo de preparo, tempo).
- Crie 30 receitas reais como conjunto de teste.
- Compare 4 prompts: zero-shot, few-shot (3 ex), CoT, structured output via Outlines/Instructor.
- Use Promptfoo ou suite própria para benchmarking.

> **Variante guiada**: rode os 4 prompts nas mesmas 30 receitas, na mesma ordem, e salve todas as saídas antes de julgar qualquer uma — julgar prompt por prompt sequencialmente introduz viés de confirmação (você espera que few-shot vá melhor, e tende a julgar mais generosamente).

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

> **Variante guiada**: primeiro confirme que o pipeline é *de fato* vulnerável (rode um documento malicioso e veja o ataque funcionar) antes de aplicar qualquer mitigação — sem essa linha de base, você não consegue saber se uma mitigação realmente funcionou ou se o ataque nunca teria funcionado ali.

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

## Erros comuns

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
| ReAct | Agentes (mod. [13](13_agentes_tools_protocolos.md)) |
| Structured output | Tools/MCP (mod. [13](13_agentes_tools_protocolos.md)) |
| LLM-as-judge | Avaliação (mod. [14](14_avaliacao_e_seguranca.md)) |
| Prompt injection | Segurança (mod. [14](14_avaliacao_e_seguranca.md)), Produção (mod. [15](15_engenharia_producao.mdx)) |
| DSPy / promptfoo | Engenharia de produção (mod. [15](15_engenharia_producao.mdx)) |

---

## Checklist de saída

- [ ] Sei usar chat templates corretos por família de modelo.
- [ ] Construí pipeline com structured output confiável (Outlines/Instructor/Zod).
- [ ] Implementei ReAct manual sem framework.
- [ ] Tenho intuição sobre quando CoT ajuda e quando não.
- [ ] Testei e documentei vulnerabilidades de prompt injection no meu projeto.
- [ ] Versiono prompts e tenho conjunto de eval rodando em CI.
