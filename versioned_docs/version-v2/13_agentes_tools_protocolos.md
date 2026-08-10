---
id: 13_agentes_tools_protocolos
title: "Módulo 13 — Agentes, Tools e Protocolos (MCP, A2A)"
sidebar_position: 13
---

# Módulo 13 — Agentes, Tools e Protocolos (MCP, A2A)

> **Objetivo**: dominar o paradigma de agentes — LLMs que decidem, planejam, usam ferramentas, mantêm estado, colaboram. Function calling, ReAct, MCP, frameworks (LangGraph, CrewAI, Mastra), padrões multi-agente.
>
> **Pré-requisitos**: Módulos [11](11_prompt_engineering.md), [12](12_rag.mdx).
>
> **Tempo de referência**: 4–6 semanas.

## Objetivos de aprendizagem

Ao final deste módulo você deve ser capaz de:

- Distinguir workflow, router e agente de verdade — e explicar por que a diferença importa para custo e previsibilidade.
- Explicar o ciclo completo de function calling, do lado do modelo e do lado do orquestrador.
- Explicar o que MCP padroniza e por que isso importa para quem constrói integrações.
- Escolher a topologia multi-agente certa (ou nenhuma) para um problema dado.
- Explicar por que code execution sem sandbox é um risco de segurança, não só uma prática ruim.

---

## Por que isso importa

A próxima fronteira de aplicações de IA não é "chatbot" — é **agente**: sistema autônomo que executa tarefas multi-passo, integra com APIs, manipula arquivos, lê e escreve em sistemas externos. Mas agentes mal-projetados são caros, lentos e perigosos. Você precisa entender o paradigma a fundo.

---

## 13.1 O que é (e o que não é) um agente

### Definição operacional
Sistema baseado em LLM que:
1. Recebe um **objetivo** (não apenas pergunta).
2. Tem acesso a **ferramentas** (funções, APIs, arquivos).
3. **Decide** quais ferramentas chamar e quando.
4. Mantém **estado** entre passos.
5. **Itera** até concluir ou falhar.

### Espectro de agência
- **Workflow** (estático): pipeline fixo de chamadas LLM. Não é agente.
- **Router**: LLM escolhe entre N caminhos pré-definidos. Agente fraco.
- **Tool-calling agent**: ReAct, decide ações dinamicamente. Agente padrão.
- **Multi-agente**: múltiplos LLMs colaborando.
- **Autonomous agent**: longo horizonte, auto-correção, planejamento meta.

> **Intuição**: a diferença entre workflow e agente não é sobre "usar LLM" — é sobre **onde mora o controle de fluxo**. Num workflow, o código decide a sequência de passos e o LLM só preenche conteúdo dentro de cada passo (previsível, testável, barato). Num agente de verdade, o *LLM* decide a sequência — quantos passos, em que ordem, quando parar — o que ganha flexibilidade pra tarefas abertas, mas perde previsibilidade e controle de custo (um agente pode, em teoria, rodar por muito mais passos que o esperado). O espectro acima é uma régua de "quanto controle você está cedendo ao modelo", não uma hierarquia de "melhor para pior" — a recomendação da própria Anthropic (referência abaixo) é usar o mínimo de agência necessário pra tarefa, não o máximo disponível.
>
> **Checkpoint**: sem olhar o texto, explique a diferença entre um "router" e um "tool-calling agent" nesse espectro. Depois, explique por que "mais agência" não é sempre a escolha certa.

### Referências
- `Paper` **Building Effective Agents** (Anthropic, 2024) — leitura essencial. https://www.anthropic.com/engineering/building-effective-agents
- `Paper` **The Rise and Potential of Large Language Model Based Agents: A Survey** — Xi et al. (2023). https://arxiv.org/abs/2309.07864
- `Paper` **Cognitive Architectures for Language Agents** — Sumers et al. (2023). https://arxiv.org/abs/2309.02427

---

## 13.2 Function calling / Tool use

### Conceito
Modelo é treinado para retornar **JSON estruturado** indicando função a chamar e argumentos. O orquestrador (seu código) executa, devolve o resultado, modelo continua.

### Formatos
- **OpenAI tools** (function calling).
- **Anthropic tool use**.
- **Mistral, Cohere, Gemini** — todos com tool use.
- **Modelos open**: LLaMA-3, Qwen 2.5, Mistral — tool use nativo no chat template.

### Anatomia
```
User: Que horas são em Tóquio?

Tool definition: get_time(timezone: str) → str

Modelo retorna: { "tool": "get_time", "args": {"timezone": "Asia/Tokyo"} }

Sistema executa: "14:32 JST"

Modelo gera resposta final.
```

> **Intuição**: o modelo *nunca executa nada* — ele só gera texto (JSON estruturado, mod. [11](11_prompt_engineering.md#115-structured-output)) descrevendo a intenção de chamar uma função com certos argumentos. Todo o poder (e todo o risco) está no seu código, que decide se e como executar essa intenção. É por isso que "validar argumentos do modelo" não é paranoia — o modelo é, estruturalmente, só uma fonte de sugestões de ação; seu orquestrador é o único ponto que decide se essas sugestões viram execução real. Um function-calling que executa direto o que o modelo retorna, sem validação, está tratando texto gerado por um modelo probabilístico como se fosse comando confiável.

### Ferramentas e bibliotecas
- `Ferramenta` **OpenAI Python/TS SDKs**.
- `Ferramenta` **Anthropic SDKs**.
- `Ferramenta` **Vercel AI SDK** — abstração agnóstica de provedor para tools (TS). https://ai-sdk.dev/
- `Ferramenta` **Instructor** (Python) — tools com Pydantic.

### Validação
**Sempre** valide argumentos retornados pelo modelo (Pydantic, Zod). Modelo pode produzir JSON inválido ou argumentos perigosos.

> **Checkpoint**: sem olhar o texto, explique por que "o modelo decidiu chamar essa função" não é o mesmo que "essa função deve ser executada sem verificação".

### Referências
- `Paper` **Toolformer: Language Models Can Teach Themselves to Use Tools** — Schick et al. (2023). https://arxiv.org/abs/2302.04761
- `Paper` **Gorilla: Large Language Model Connected with Massive APIs** — Patil et al. (2023). https://arxiv.org/abs/2305.15334
- `Paper` **ToolLLM** — Qin et al. (2023). https://arxiv.org/abs/2307.16789

---

## 13.3 ReAct revisitado e padrões de loop

### Loop ReAct canônico
```
while not done:
    response = llm(history + tools_description)
    if response.is_final_answer:
        return response.answer
    tool_call = response.tool_call
    observation = execute_tool(tool_call)
    history.append((response, observation))
```

### Variações
- **Plan-and-Execute**: gerar plano primeiro, executar, replanejar se falhar.
- `Paper` **Plan-and-Solve Prompting**. https://arxiv.org/abs/2305.04091
- **Reflexion**: agente reflete sobre tentativas anteriores. https://arxiv.org/abs/2303.11366
- **Voyager**: agente que adiciona habilidades a uma "skill library". https://arxiv.org/abs/2305.16291

---

## 13.4 Memória de agentes

### Tipos
- **Short-term**: histórico da conversa (limitado pelo contexto).
- **Working memory**: estado intermediário (variáveis, scratchpad).
- **Long-term**: persistente entre sessões. Geralmente RAG sobre logs/notas.
- **Episodic**: experiências passadas indexadas para recall.
- **Semantic**: conhecimento abstraído.

> **Intuição**: cada tipo de memória resolve uma limitação diferente do modelo. Short-term é literalmente o que cabe na janela de contexto do LLM — finita e cara (mais tokens = mais custo e mais risco de "lost in the middle", mod. [12](12_rag.mdx#127-geração-com-contexto)). Long-term contorna esse limite armazenando informação *fora* do contexto (num banco vetorial, por exemplo) e trazendo de volta só o relevante quando necessário — é literalmente RAG (mod. 12) aplicado à memória do próprio agente, em vez de a uma base de conhecimento externa. Episodic e semantic são refinamentos: episodic guarda "o que aconteceu" (experiências específicas, recuperáveis por similaridade), semantic guarda "o que foi aprendido/abstraído" daquelas experiências — a diferença entre lembrar de um evento específico e lembrar de uma regra geral extraída de vários eventos.

### Ferramentas
- `Ferramenta` **Mem0** — camada de memória para LLMs. https://mem0.ai/
- `Ferramenta` **LangGraph** tem checkpointing nativo.
- `Ferramenta` **Letta (ex-MemGPT)**. https://github.com/letta-ai/letta

### Referências
- `Paper` **MemGPT: Towards LLMs as Operating Systems** — Packer et al. (2023). https://arxiv.org/abs/2310.08560
- `Paper` **A Survey on Memory Mechanism of Large Language Model based Agents** (2024). https://arxiv.org/abs/2404.13501

---

## 13.5 MCP — Model Context Protocol

`Paper` **Especificação oficial**: https://modelcontextprotocol.io/
`Ferramenta` **Repositório oficial**: https://github.com/modelcontextprotocol

### O que é
Protocolo aberto da Anthropic (proposto fim de 2024) para padronizar como LLMs e aplicações cliente acessam **ferramentas, recursos, dados e prompts** de servidores externos. Análogo a "USB para IA".

### Conceitos do protocolo
- **Server**: expõe capabilities (tools, resources, prompts).
- **Client**: consumido por uma aplicação host (Claude Desktop, IDEs, agentes).
- **Tools**: funções invocáveis (similar a function calling, padronizado).
- **Resources**: dados arbitrários (arquivos, queries de DB) referenciáveis por URI.
- **Prompts**: templates parametrizáveis.
- **Transports**: stdio (local), HTTP/SSE (remoto), Streamable HTTP.

> **Intuição — "USB para IA"**: antes do MCP, cada integração (Slack, GitHub, um banco de dados interno) precisava de um adaptador específico escrito para cada framework de agente diferente — N ferramentas × M frameworks = N×M integrações para manter. MCP padroniza a interface entre "coisa que expõe capacidades" (server) e "coisa que consome capacidades" (client/host), do mesmo jeito que USB padronizou a interface entre periférico e computador — um servidor MCP escrito uma vez funciona com qualquer client MCP (Claude Desktop, um IDE, um agente customizado), sem reescrever nada. O ganho é de manutenção e composição: quem constrói uma integração a constrói uma vez; quem constrói um agente ganha acesso a todo o ecossistema de servidores MCP existentes sem escrever adaptador nenhum.

### Por que importa
- Desacopla **provedor** de **capacidade**: troque LLM, não troque integrações.
- Padroniza segurança, descoberta de tools, autenticação (OAuth para servidores remotos).
- Ecossistema crescente: GitHub, Slack, Notion, Postgres, Filesystem, etc.

> **Checkpoint**: sem olhar o texto, explique o problema de "N ferramentas × M frameworks" que MCP resolve, com suas próprias palavras.

### SDKs
- `Ferramenta` **MCP Python SDK**. https://github.com/modelcontextprotocol/python-sdk
- `Ferramenta` **MCP TypeScript SDK**. https://github.com/modelcontextprotocol/typescript-sdk
- `Ferramenta` **MCP servers** (catálogo oficial). https://github.com/modelcontextprotocol/servers

### Hands-on
- Servidor MCP em Python que expõe ferramentas para um banco SQLite local.
- Mesmo servidor em TS.
- Cliente que consome via Claude Desktop ou via SDK programático.

---

## 13.6 Outros protocolos / padrões emergentes

### A2A (Agent-to-Agent)
Protocolo do Google para comunicação entre agentes de fornecedores diferentes. https://github.com/a2aproject/A2A

### OpenAI Agents SDK
Framework oficial OpenAI para agentes. https://github.com/openai/openai-agents-python

### AutoGen (Microsoft)
Framework de agentes multi-LLM com mensagens entre eles. https://github.com/microsoft/autogen

### Histórico relevante
- **AutoGPT** (2023) — primeiro agente autônomo viral. https://github.com/Significant-Gravitas/AutoGPT
- **BabyAGI** — minimalista, didático.
- **GPT Engineer** — agente para gerar projetos.

> Vale notar a diferença de escopo: MCP (seção 13.5) padroniza como um agente acessa *ferramentas e dados*; A2A padroniza como *agentes diferentes conversam entre si* — são complementares, não concorrentes, resolvendo camadas diferentes do mesmo problema de interoperabilidade.

---

## 13.7 Frameworks de agentes

### Python
- **LangGraph** (LangChain) — máquinas de estado, checkpoints, streaming, popular. https://www.langchain.com/langgraph
- **CrewAI** — multi-agente focado em personas. https://www.crewai.com/
- **AutoGen** (Microsoft) — diálogo multi-agente. https://github.com/microsoft/autogen
- **smolagents** (Hugging Face) — agentes simples com code execution. https://github.com/huggingface/smolagents
- **OpenAI Agents SDK**.
- **Pydantic AI** — type-safe agents com Pydantic. https://ai.pydantic.dev/

### TypeScript
- **Mastra** — framework opinativo, observability nativa. https://mastra.ai/
- **LangGraph.js** (porte de LangGraph). https://langchain-ai.github.io/langgraphjs/
- **Vercel AI SDK** com `useChat` + tools.
- **VoltAgent**, **AI SDK Tools** — opções emergentes.

### Quando *não* usar framework
- Tarefa simples (1–3 chamadas LLM): escreva direto.
- Aprendizado: implemente loop ReAct manual primeiro.
- Performance crítica: frameworks adicionam latência.

### Referência
- `Paper` **Building Effective Agents** (Anthropic) — recomenda **começar simples**, frameworks só quando necessário. https://www.anthropic.com/engineering/building-effective-agents

---

## 13.8 Padrões multi-agente

### Topologias
- **Pipeline / Sequential**: agente A → agente B → agente C.
- **Hierarchical (orchestrator + workers)**: gerente delega a especialistas.
- **Debate / Adversarial**: agentes argumentam, juiz decide.
- **Round-robin / Group chat**: agentes se revezam.
- **Swarm**: handoffs livres entre agentes.

### Quando faz sentido
- Tarefa complexa com sub-domínios distintos (pesquisa: planejador + crawler + analista + redator).
- Agente "juiz" + agente "executor" para validação.

### Cuidados
- **Custo explode** com mais agentes.
- **Cascata de erros**: erro em A vira input ruim para B.
- **Difícil debugar**.

> **Intuição**: multi-agente é essencialmente decomposição de tarefa (mod. [11](11_prompt_engineering.md#116-prompts-complexos-técnicas-avançadas)) aplicada em nível de sistema em vez de prompt único — cada agente tem um escopo mais estreito (e um prompt/persona mais focado), o que pode melhorar qualidade por sub-tarefa. Mas o custo não é só em tokens (cada agente é uma ou mais chamadas LLM completas): é também em superfície de erro — numa pipeline sequencial, um erro do agente A vira *entrada* do agente B, que não tem como saber que a entrada já está corrompida. É por isso que a recomendação padrão (mesmo dos criadores de frameworks multi-agente) é começar com um único agente bem prompted, e só fragmentar em múltiplos quando há evidência concreta de que um agente único está sobrecarregado ou confuso sobre seu papel.
>
> **Checkpoint**: sem olhar o texto, explique por que erros em pipelines multi-agente sequenciais são mais difíceis de debugar que erros num agente único.

### Referências
- `Paper` **AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation** — Wu et al. (2023). https://arxiv.org/abs/2308.08155
- `Paper` **MetaGPT: Meta Programming for Multi-Agent Collaborative Framework**. https://arxiv.org/abs/2308.00352

---

## 13.9 Code execution como ferramenta

Muitos problemas (matemática, análise de dados) são melhor resolvidos com **código gerado pelo LLM e executado em sandbox**.

### Padrões
- **Code Interpreter** (OpenAI), **Code Execution** (Anthropic).
- **REPL local sandboxed** (Docker, gVisor, Firecracker).
- **e2b** (sandbox como serviço). https://e2b.dev/
- **Modal**, **Daytona**, **CodeSandbox**.

### Frameworks com code execution
- **smolagents** — `CodeAgent` é code-first.
- **AutoGen** — agente com Python REPL.

### Riscos
- Execução de código gerado por LLM em ambiente sem sandbox = catástrofe esperando para acontecer.
- **Sempre** isole.

> **Intuição**: código gerado por um LLM tem a mesma confiabilidade que texto gerado por um LLM — pode estar certo, pode alucinar uma chamada perigosa, pode (via prompt injection, mod. [11](11_prompt_engineering.md#118-prompt-injection-e-segurança)) ter sido manipulado por um input malicioso a fazer algo destrutivo. Rodar esse código diretamente no mesmo ambiente da sua aplicação é conceder a um texto gerado probabilisticamente o mesmo nível de confiança que você daria a código revisado por humano. Sandboxing (Docker, gVisor, Firecracker, ou serviços dedicados como e2b) isola a execução — se o código tentar deletar arquivos, acessar rede indevidamente, ou consumir recursos indefinidamente, o dano fica contido ao sandbox descartável, não à sua infraestrutura real.

### Referências
- `Paper` **PAL: Program-Aided Language Models** — Gao et al. (2022). https://arxiv.org/abs/2211.10435
- `Paper` **Program of Thoughts** — Chen et al. (2022). https://arxiv.org/abs/2211.12588

---

## 13.10 Computer Use / Browser Use

Agentes que controlam interface gráfica (mouse, teclado) ou browser via screenshots + cliques.

### Implementações
- **Anthropic Computer Use** (Claude). https://docs.anthropic.com/en/docs/build-with-claude/computer-use
- **OpenAI Operator**.
- **browser-use** (open). https://github.com/browser-use/browser-use
- **Playwright + LLM** (DIY).

### Limitações
- Lento.
- Erros visuais.
- Custos altos (frames como imagens).

---

## 13.11 Observability de agentes

Não dá para entender o que agente fez sem **trace estruturado**: cada chamada LLM, cada tool call, latências, tokens, decisões.

### Ferramentas
- `Ferramenta` **LangSmith** (LangChain). https://www.langchain.com/langsmith
- `Ferramenta` **Langfuse** (open-source). https://langfuse.com/
- `Ferramenta` **Helicone** — proxy + analytics. https://www.helicone.ai/
- `Ferramenta` **Phoenix** (Arize). https://github.com/Arize-ai/phoenix
- `Ferramenta` **OpenTelemetry GenAI** — padrão emergente. https://opentelemetry.io/docs/specs/semconv/gen-ai/

Mais detalhes no módulo [15](15_engenharia_producao.mdx).

---

## 13.12 Avaliação de agentes

### Diferença vs eval de LLM puro
- Trajetórias longas, não-determinísticas.
- Sucesso é multidimensional: completou tarefa? quantos passos? custo? segurança?

### Benchmarks
- **AgentBench**. https://arxiv.org/abs/2308.03688
- **WebArena** — agentes em ambiente web realista. https://webarena.dev/
- **SWE-bench** — agentes corrigindo bugs em repos GitHub reais. https://www.swebench.com/
- **GAIA**. https://arxiv.org/abs/2311.12983
- **τ-bench** (tau-bench) — agentes em domínios de atendimento. https://github.com/sierra-research/tau-bench

### Métricas
- **Task success rate**.
- **Steps to success**.
- **Cost per task**.
- **Error recovery rate**.
- **Tool use accuracy**.

---

## Projetos práticos

### Projeto 13.1 — Agente ReAct from scratch
- Sem framework. Loop manual em Python E em TS.
- 3 ferramentas: cálculo (eval seguro), busca local (em corpus do mod. [12](12_rag.mdx)), data/hora.
- Logging estruturado de cada passo.

> **Variante guiada**: implemente e teste cada tool isoladamente (fora do loop do agente) antes de conectá-las — confirme que `calc("2+2")` retorna 4 e que a busca retorna resultados sensatos, antes de depurar por que o *agente* não está usando a ferramenta certa. Separa bug de tool de bug de orquestração.

### Projeto 13.2 — Servidor MCP
- Crie um servidor MCP em Python que expõe:
  - Tool: `query_db(sql: str)` (sandboxed SQLite).
  - Resource: arquivos de uma pasta.
  - Prompt: template "summarize_table".
- Conecte via Claude Desktop **e** programaticamente.
- Refazer em TS para entender ambos SDKs.

### Projeto 13.3 — Agente de pesquisa
- Frameworks: LangGraph (Python) ou Mastra (TS).
- Tools: web search (DuckDuckGo, Brave, Tavily), fetch URL, write file.
- Tarefa: "Pesquise sobre X, escreva relatório em Markdown citando fontes".
- Implemente checkpointing — se falhar, retoma.

### Projeto 13.4 — Multi-agente: pipeline editorial
- Agentes: Pesquisador → Redator → Revisor → Editor.
- Cada um tem prompt e tools específicas.
- Orquestre em CrewAI ou em LangGraph.

> **Variante guiada**: antes de encadear os 4 agentes, rode cada um isoladamente com uma entrada de exemplo e confira a saída — um pipeline de 4 agentes onde um deles produz saída ruim silenciosamente é muito mais difícil de diagnosticar depois de tudo encadeado.

### Projeto 13.5 — Agente com code execution
- Use smolagents `CodeAgent` ou AutoGen.
- Tarefa: análise de CSV (estatísticas, plots).
- Sandbox via Docker ou e2b.

### Projeto 13.6 — Avaliação rigorosa
- Construa eval set de 30 tarefas multi-passo com gabarito.
- Rode 2 implementações (com framework vs sem).
- Compare success rate, steps, custo, latência.
- Use Langfuse para tracing.

### Projeto 13.7 — Agente browser
- Use `browser-use` ou Playwright + LLM.
- Tarefa simples: navegar até site, extrair dados, preencher formulário.
- Documente quando falha e por quê.

---

## Erros comuns

- **Loop infinito**: agente repete a mesma ação. Sempre tenha **max_steps** e detecção de loop.
- **Tools sem validação**: argumentos do LLM tratados como confiáveis = falha de segurança.
- **Sem observability**: debugar agente sem trace é impossível.
- **Frameworks como prematuros**: começar com CrewAI antes de entender ReAct manual.
- **Multi-agente para tudo**: simplicidade > performance teórica em quase todos os casos.
- **Code execution sem sandbox**: catástrofe esperando.
- **Memória ilimitada**: contexto cresce até estourar; precisa de truncamento/compressão.
- **Confiar em saída de tool externa** sem validar.

---

## Conexão com módulos seguintes

| Conceito daqui | Aparece em |
|---|---|
| Tool use | Avaliação (mod. [14](14_avaliacao_e_seguranca.md)), Produção (mod. [15](15_engenharia_producao.mdx)) |
| MCP | Integrações em produção (mod. [15](15_engenharia_producao.mdx)) |
| Agent observability | Engenharia de produção (mod. [15](15_engenharia_producao.mdx)) |
| Agentic RAG | RAG (mod. [12](12_rag.mdx) — bidirecional) |
| Multimodal agents | Mod. [18](18_multimodal.mdx) |

---

## Checklist de saída

- [ ] Implementei loop ReAct manual em Python e em TS.
- [ ] Construí servidor MCP funcional e conectei a um cliente real.
- [ ] Construí agente multi-passo com framework (LangGraph ou Mastra).
- [ ] Tenho observability (Langfuse/LangSmith) configurado.
- [ ] Sei distinguir workflow de agente, e quando preferir cada um.
- [ ] Avaliei agente em conjunto de teste com métricas objetivas.
- [ ] Sei o risco de prompt injection em agentes com tools potentes.
