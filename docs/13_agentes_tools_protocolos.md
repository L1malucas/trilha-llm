---
id: 13_agentes_tools_protocolos
title: "Módulo 13 — Agentes, Tools e Protocolos (MCP, A2A)"
sidebar_position: 6
---

# Módulo 13 — Agentes, Tools e Protocolos (MCP, A2A)

> **Objetivo**: dominar o paradigma de agentes — LLMs que decidem, planejam, usam ferramentas, mantêm estado, colaboram. Function calling, ReAct, MCP, frameworks (LangGraph, CrewAI, Mastra), padrões multi-agente.
>
> **Pré-requisitos**: Módulos [08](08_llms_arquiteturas.md)–[12](12_rag.mdx) — em particular, o loop ReAct manual (Projeto 11.3), function calling (Projeto 11.5), o cliente TypeScript com streaming (Projeto 10.2) e o mini-RAG (Projeto 12.1), que este módulo estende diretamente.
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

A próxima fronteira de aplicações de IA não é "chatbot" — é agente: sistema autônomo que executa tarefas multi-passo, integra com APIs, manipula arquivos, lê e escreve em sistemas externos. Você já construiu a peça central disso (o loop ReAct) no Projeto 11.3; este módulo formaliza o paradigma, adiciona protocolos padronizados (MCP) e ferramentas de produção, e é honesto sobre os riscos: agentes mal-projetados são caros, lentos e, quando têm acesso a ferramentas poderosas sem validação, perigosos.

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
- **Tool-calling agent**: ReAct, decide ações dinamicamente. Agente padrão — é o que você já implementou no Projeto 11.3, e o que o Projeto 13.1 formaliza melhor.
- **Multi-agente**: múltiplos LLMs colaborando (seção 13.8, Projeto 13.4).
- **Autonomous agent**: longo horizonte, auto-correção, planejamento meta.

> **Intuição**: a diferença entre workflow e agente não é sobre "usar LLM" — é sobre **onde mora o controle de fluxo**. Num workflow, o código decide a sequência de passos e o LLM só preenche conteúdo dentro de cada passo (previsível, testável, barato). Num agente de verdade, o *LLM* decide a sequência — quantos passos, em que ordem, quando parar — o que ganha flexibilidade pra tarefas abertas, mas perde previsibilidade e controle de custo (um agente pode, em teoria, rodar por muito mais passos que o esperado, como você já viu precisar de um `max_passos` no Projeto 11.3). O espectro acima é uma régua de "quanto controle você está cedendo ao modelo", não uma hierarquia de "melhor para pior" — a recomendação mais citada na indústria (o guia "Building Effective Agents" da Anthropic) é usar o mínimo de agência necessário pra tarefa, não o máximo disponível.
>
> **Checkpoint**: sem olhar o texto, explique a diferença entre um "router" e um "tool-calling agent" nesse espectro. Depois, explique por que "mais agência" não é sempre a escolha certa.

---

## 13.2 Function calling / Tool use

### Conceito
Modelo é treinado para retornar **JSON estruturado** indicando função a chamar e argumentos. O orquestrador (seu código) executa, devolve o resultado, modelo continua. É o mesmo mecanismo que você já usou no Projeto 11.5, com `tools` e `tool_choice` na API compatível OpenAI.

### Formatos
OpenAI, Anthropic, Mistral, Cohere e Gemini implementam tool use nativamente em suas APIs; entre os modelos abertos, LLaMA-3, Qwen 2.5 e Mistral têm suporte a tool use direto no chat template (o mesmo `apply_chat_template` do Projeto 9.1 aceita uma lista de ferramentas, além de mensagens).

### Anatomia
```
User: Que horas são em Tóquio?

Tool definition: get_time(timezone: str) → str

Modelo retorna: { "tool": "get_time", "args": {"timezone": "Asia/Tokyo"} }

Sistema executa: "14:32 JST"

Modelo gera resposta final.
```

> **Intuição**: o modelo *nunca executa nada* — ele só gera texto (JSON estruturado, mod. [11](11_prompt_engineering.md#115-structured-output)) descrevendo a intenção de chamar uma função com certos argumentos. Todo o poder (e todo o risco) está no seu código, que decide se e como executar essa intenção. É por isso que "validar argumentos do modelo" não é paranoia — o modelo é, estruturalmente, só uma fonte de sugestões de ação; seu orquestrador é o único ponto que decide se essas sugestões viram execução real. Um function-calling que executa direto o que o modelo retorna, sem validação, está tratando texto gerado por um modelo probabilístico como se fosse comando confiável — a mesma lição do parser de `Action:` do Projeto 11.3, agora com consequências maiores porque as ferramentas de um agente costumam ser mais poderosas que uma calculadora de brinquedo.

### Validação
**Sempre** valide argumentos retornados pelo modelo (Pydantic, Zod). Modelo pode produzir JSON inválido ou argumentos perigosos.

> **Checkpoint**: sem olhar o texto, explique por que "o modelo decidiu chamar essa função" não é o mesmo que "essa função deve ser executada sem verificação".

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

Este é exatamente o `react_loop` que você escreveu no Projeto 11.3, generalizado — no lugar de um parser de regex específico, um agente de produção usa function calling estruturado (seção 13.2) para extrair `tool_call` de forma confiável.

### Variações
- **Plan-and-Execute**: em vez de decidir um passo de cada vez, o agente gera um plano completo primeiro (uma lista de passos), executa cada um, e só replaneja se algum passo falhar — reduz o número de chamadas ao LLM em tarefas cujo caminho é previsível desde o início, ao custo de ser menos adaptável a descobertas no meio do caminho.
- **Reflexion** (já introduzido no mod. [11](11_prompt_engineering.md#116-prompts-complexos-técnicas-avançadas)): depois de uma tentativa que falha, o agente gera uma reflexão verbal sobre o erro e a adiciona ao contexto da próxima tentativa.
- **Voyager**: um agente (originalmente desenhado para jogar Minecraft) que, ao resolver um problema novo, salva a solução como uma "habilidade" reutilizável numa biblioteca de código — cada problema resolvido expande o repertório de ações disponíveis para os próximos, em vez de recomeçar do zero toda vez.

---

## 13.4 Memória de agentes

### Tipos
- **Short-term**: histórico da conversa (limitado pelo contexto).
- **Working memory**: estado intermediário (variáveis, scratchpad).
- **Long-term**: persistente entre sessões. Geralmente RAG sobre logs/notas.
- **Episodic**: experiências passadas indexadas para recall.
- **Semantic**: conhecimento abstraído.

> **Intuição**: cada tipo de memória resolve uma limitação diferente do modelo. Short-term é literalmente o que cabe na janela de contexto do LLM — finita e cara (mais tokens = mais custo e mais risco de "lost in the middle", mod. [12](12_rag.mdx#127-geração-com-contexto)). Long-term contorna esse limite armazenando informação *fora* do contexto (num banco vetorial, por exemplo) e trazendo de volta só o relevante quando necessário — é literalmente RAG (mod. 12, incluindo o pipeline que você já construiu no Projeto 12.1) aplicado à memória do próprio agente, em vez de a uma base de conhecimento externa. Episodic e semantic são refinamentos: episodic guarda "o que aconteceu" (experiências específicas, recuperáveis por similaridade), semantic guarda "o que foi aprendido/abstraído" daquelas experiências — a diferença entre lembrar de um evento específico e lembrar de uma regra geral extraída de vários eventos.

Ferramentas dedicadas a isso incluem o Mem0 (uma camada de memória pronta para agentes), o checkpointing nativo do LangGraph (usado no Projeto 13.3), e o Letta (antigo MemGPT, que trata a memória do agente de forma explicitamente parecida com memória virtual de sistema operacional — parte "quente" no contexto, parte "fria" recuperável sob demanda).

---

## 13.5 MCP — Model Context Protocol

### O que é
Protocolo aberto da Anthropic (proposto no fim de 2024) para padronizar como LLMs e aplicações cliente acessam **ferramentas, recursos, dados e prompts** de servidores externos. Análogo a "USB para IA". Especificação e SDKs oficiais em https://modelcontextprotocol.io/.

### Conceitos do protocolo
- **Server**: expõe capabilities (tools, resources, prompts).
- **Client**: consumido por uma aplicação host (Claude Desktop, IDEs, agentes).
- **Tools**: funções invocáveis (similar a function calling, padronizado).
- **Resources**: dados arbitrários (arquivos, queries de DB) referenciáveis por URI.
- **Prompts**: templates parametrizáveis.
- **Transports**: stdio (local), HTTP/SSE (remoto), Streamable HTTP.

> **Intuição — "USB para IA"**: antes do MCP, cada integração (Slack, GitHub, um banco de dados interno) precisava de um adaptador específico escrito para cada framework de agente diferente — N ferramentas × M frameworks = N×M integrações para manter. MCP padroniza a interface entre "coisa que expõe capacidades" (server) e "coisa que consome capacidades" (client/host), do mesmo jeito que USB padronizou a interface entre periférico e computador — um servidor MCP escrito uma vez funciona com qualquer client MCP (Claude Desktop, um IDE, um agente customizado), sem reescrever nada. O ganho é de manutenção e composição: quem constrói uma integração a constrói uma vez; quem constrói um agente ganha acesso a todo o ecossistema de servidores MCP existentes sem escrever adaptador nenhum. Você constrói um servidor MCP real, do zero, no Projeto 13.2.

### Por que importa
- Desacopla **provedor** de **capacidade**: troque LLM, não troque integrações.
- Padroniza segurança, descoberta de tools, autenticação (OAuth para servidores remotos).
- Ecossistema crescente: GitHub, Slack, Notion, Postgres, Filesystem, etc.

> **Checkpoint**: sem olhar o texto, explique o problema de "N ferramentas × M frameworks" que MCP resolve, com suas próprias palavras.

---

## 13.6 Outros protocolos / padrões emergentes

**A2A (Agent-to-Agent)** é um protocolo do Google, com escopo diferente de MCP: em vez de padronizar como um agente acessa ferramentas e dados, padroniza como *agentes diferentes* (potencialmente de fornecedores diferentes) se comunicam entre si — complementar a MCP, não concorrente. O **OpenAI Agents SDK** e o **AutoGen** (Microsoft) são frameworks oficiais/patrocinados de fornecedores específicos para construir agentes e conversas multi-agente. Historicamente, **AutoGPT** (2023) foi o primeiro agente autônomo a viralizar, com **BabyAGI** como uma versão minimalista e didática do mesmo princípio, e **GPT Engineer** aplicando a ideia especificamente a gerar projetos de software inteiros a partir de uma descrição.

---

## 13.7 Frameworks de agentes

### Python
- **LangGraph** (LangChain) — modela o agente como uma máquina de estados explícita, com checkpointing e streaming nativos; usado no Projeto 13.3.
- **CrewAI** — multi-agente focado em personas nomeadas (um "Pesquisador", um "Redator") orquestradas num processo sequencial ou hierárquico; usado no Projeto 13.4.
- **AutoGen** (Microsoft) — diálogo multi-agente, agentes conversando entre si em vez de um orquestrador central decidindo tudo.
- **smolagents** (Hugging Face) — agentes onde a "ação" do modelo é literalmente escrever e executar código Python, não chamar uma função pré-definida; usado no Projeto 13.5.
- **Pydantic AI** — agentes com tipagem forte via Pydantic em toda a interface.

### TypeScript
- **Mastra** — framework opinativo, com observability nativa.
- **LangGraph.js** — porte do LangGraph.
- **Vercel AI SDK** com `useChat` + tools — a extensão natural do que você já usa desde o Projeto 10.2.

### Quando *não* usar framework
- Tarefa simples (1–3 chamadas LLM): escreva direto, como no Projeto 13.1.
- Aprendizado: implemente loop ReAct manual primeiro — você já fez isso.
- Performance crítica: frameworks adicionam latência e camadas de abstração.

O guia "Building Effective Agents" (Anthropic) resume essa recomendação: comece simples, adicione framework só quando a complexidade real do problema justificar.

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

> **Intuição**: multi-agente é essencialmente decomposição de tarefa (mod. [11](11_prompt_engineering.md#116-prompts-complexos-técnicas-avançadas)) aplicada em nível de sistema em vez de prompt único — cada agente tem um escopo mais estreito (e um prompt/persona mais focado), o que pode melhorar qualidade por sub-tarefa. Mas o custo não é só em tokens (cada agente é uma ou mais chamadas LLM completas): é também em superfície de erro — numa pipeline sequencial, um erro do agente A vira *entrada* do agente B, que não tem como saber que a entrada já está corrompida. É por isso que a recomendação padrão (mesmo dos criadores de frameworks multi-agente) é começar com um único agente bem prompted, e só fragmentar em múltiplos quando há evidência concreta de que um agente único está sobrecarregado ou confuso sobre seu papel. Você observa esse trade-off diretamente no Projeto 13.4.
>
> **Checkpoint**: sem olhar o texto, explique por que erros em pipelines multi-agente sequenciais são mais difíceis de debugar que erros num agente único.

---

## 13.9 Code execution como ferramenta

Muitos problemas (matemática, análise de dados) são melhor resolvidos com **código gerado pelo LLM e executado em sandbox**.

### Padrões
- **Code Interpreter** (OpenAI), **Code Execution** (Anthropic) — a versão hospedada, dentro da própria API do provedor.
- **REPL local sandboxed** (Docker, gVisor, Firecracker) — você controla o isolamento.
- **e2b** — sandbox como serviço, usado no Projeto 13.5.
- **Modal**, **Daytona**, **CodeSandbox** — alternativas com propósito semelhante.

### Riscos
Execução de código gerado por LLM em ambiente sem sandbox é uma catástrofe esperando para acontecer — sempre isole.

> **Intuição**: código gerado por um LLM tem a mesma confiabilidade que texto gerado por um LLM — pode estar certo, pode alucinar uma chamada perigosa, pode (via prompt injection, mod. [11](11_prompt_engineering.md#118-prompt-injection-e-segurança)) ter sido manipulado por um input malicioso a fazer algo destrutivo. Rodar esse código diretamente no mesmo ambiente da sua aplicação é conceder a um texto gerado probabilisticamente o mesmo nível de confiança que você daria a código revisado por humano. Sandboxing (Docker, gVisor, Firecracker, ou serviços dedicados como e2b) isola a execução — se o código tentar deletar arquivos, acessar rede indevidamente, ou consumir recursos indefinidamente, o dano fica contido ao sandbox descartável, não à sua infraestrutura real. É a mesma lógica do `eval` restrito da calculadora do Projeto 11.3, levada a sério: ali, restringir `__builtins__` era uma proteção mínima o bastante para uma expressão aritmética; código Python arbitrário gerado por um modelo exige isolamento de processo/máquina virtual de verdade, não só remover builtins.

---

## 13.10 Computer Use / Browser Use

Agentes que controlam interface gráfica (mouse, teclado) ou browser via screenshots + cliques.

### Implementações
- **Anthropic Computer Use** (Claude) e **OpenAI Operator** — versões hospedadas pelos próprios provedores.
- **browser-use** — biblioteca open-source especializada em automação de browser via LLM.
- **Playwright + LLM (DIY)** — a abordagem usada no Projeto 13.7, para entender o mecanismo sem depender de uma biblioteca pronta.

### Limitações
- Lento.
- Erros visuais.
- Custos altos (frames como imagens custam mais tokens que texto).

---

## 13.11 Observability de agentes

Não dá para entender o que um agente fez sem **trace estruturado**: cada chamada LLM, cada tool call, latências, tokens, decisões. Ferramentas como LangSmith, Langfuse (open-source, usado no Projeto 13.6), Helicone (proxy + analytics) e Phoenix (Arize) capturam esse trace automaticamente, com o OpenTelemetry GenAI emergindo como um padrão comum entre elas. O mod. [15](15_engenharia_producao.mdx) aprofunda observability para produção de forma mais geral.

---

## 13.12 Avaliação de agentes

### Diferença vs eval de LLM puro
- Trajetórias longas, não-determinísticas.
- Sucesso é multidimensional: completou tarefa? quantos passos? custo? segurança?

### Benchmarks
AgentBench, WebArena (agentes em ambiente web realista), SWE-bench (agentes corrigindo bugs em repositórios GitHub reais), GAIA e τ-bench (agentes em domínios de atendimento) são os benchmarks públicos mais citados — cada um mede um domínio de tarefa diferente, então "meu agente é bom" só faz sentido qualificado por qual desses (ou de um conjunto próprio, como o do Projeto 13.6) foi usado para medir.

### Métricas
- **Task success rate**.
- **Steps to success**.
- **Cost per task**.
- **Error recovery rate**.
- **Tool use accuracy**.

---

## Projetos práticos

### Projeto 13.1 — Agente ReAct from scratch, em Python e em TypeScript

Você vai estender o loop ReAct do Projeto 11.3 com uma terceira ferramenta e logging estruturado, e reimplementar o mesmo loop em TypeScript.

**Pré-requisitos**: Ollama; para a versão TS, o setup do Projeto 10.2.

**1. Versão Python — adicione uma ferramenta de data/hora e logging estruturado ao loop do Projeto 11.3**:

```python
import re
import json
import time
from datetime import datetime

def calculadora(expressao):
    try:
        return str(eval(expressao, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Erro: {e}"

def busca(query):
    # reaproveita a base de conhecimento (ou o mini-RAG do Projeto 12.1) do Projeto 11.3
    return BASE_CONHECIMENTO.get(query.lower().strip(), "Não encontrado.")

def data_hora(_):
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

TOOLS = {"calculadora": calculadora, "busca": busca, "data_hora": data_hora}

def react_loop_com_log(pergunta, max_passos=5):
    historico = f"{SYSTEM_PROMPT}\n\nPergunta: {pergunta}\n"  # SYSTEM_PROMPT do Projeto 11.3, com data_hora adicionada à lista de ferramentas
    log = []

    for passo in range(max_passos):
        t0 = time.time()
        resposta = query_ollama(historico)
        latencia = time.time() - t0
        historico += resposta

        if "Final Answer:" in resposta:
            resposta_final = resposta.split("Final Answer:")[-1].strip()
            log.append({"passo": passo, "tipo": "final_answer", "conteudo": resposta_final, "latencia_s": round(latencia, 2)})
            with open("agent_trace.jsonl", "a") as f:
                for entrada in log:
                    f.write(json.dumps(entrada, ensure_ascii=False) + "\n")
            return resposta_final

        match = re.search(r"Action:\s*(\w+)\((.*?)\)", resposta)
        if not match:
            log.append({"passo": passo, "tipo": "erro_parse", "conteudo": resposta})
            break

        tool_nome, tool_arg = match.group(1), match.group(2).strip("\"'")
        observation = TOOLS.get(tool_nome, lambda x: f"Ferramenta '{tool_nome}' não existe.")(tool_arg)
        log.append({"passo": passo, "tipo": "tool_call", "tool": tool_nome, "arg": tool_arg, "observation": observation, "latencia_s": round(latencia, 2)})
        historico += f"\nObservation: {observation}\nThought:"

    return "Limite de passos atingido sem resposta final."
```

A diferença em relação ao Projeto 11.3 é o `log`: cada passo do agente (chamada de ferramenta ou resposta final) é registrado com o que foi decidido, o resultado, e quanto tempo levou, e gravado em `agent_trace.jsonl` — o mesmo princípio da seção 13.11 (observability), numa versão manual e simples. Sem esse trace, debugar por que um agente tomou uma decisão específica exigiria reler o histórico de texto inteiro a cada vez.

**2. Teste cada ferramenta isoladamente antes de rodar o agente completo**: `calculadora("2+2")` deve retornar `"4"`, `data_hora(None)` deve retornar a hora atual, `busca("capital da frança")` deve retornar `"Paris"` — confirme os três antes de rodar `react_loop_com_log`, para separar bug de ferramenta de bug de orquestração.

**3. Versão TypeScript** (`agent.ts`), reaproveitando `streamText`/`generateText` do Projeto 10.2:

```typescript
import { ollama } from "ollama-ai-provider";
import { generateText } from "ai";

type Tool = (arg: string) => string;

function calculadora(expressao: string): string {
  try {
    // Function() em vez de eval() direto, restrito a uma expressão aritmética simples
    return String(Function(`"use strict"; return (${expressao})`)());
  } catch (e) {
    return `Erro: ${e}`;
  }
}

function dataHora(): string {
  return new Date().toISOString();
}

const TOOLS: Record<string, Tool> = { calculadora, data_hora: dataHora };

const SYSTEM_PROMPT = `Você resolve perguntas usando Thought/Action/Observation.
Ferramentas: calculadora(expressao), data_hora().
Formato: Thought: ...\\nAction: nome(argumento)\\n(pare e espere Observation)
Quando souber a resposta: Thought: ...\\nFinal Answer: ...`;

async function reactLoop(pergunta: string, maxPassos = 5): Promise<string> {
  let historico = `${SYSTEM_PROMPT}\n\nPergunta: ${pergunta}\n`;

  for (let passo = 0; passo < maxPassos; passo++) {
    const { text } = await generateText({ model: ollama("qwen2.5:7b"), prompt: historico });
    historico += text;

    if (text.includes("Final Answer:")) {
      return text.split("Final Answer:")[1].trim();
    }

    const match = text.match(/Action:\s*(\w+)\((.*?)\)/);
    if (!match) return "Não consegui interpretar a ação.";

    const [, toolNome, toolArg] = match;
    const observation = TOOLS[toolNome] ? TOOLS[toolNome](toolArg.replace(/['"]/g, "")) : `Ferramenta '${toolNome}' não existe.`;
    historico += `\nObservation: ${observation}\nThought:`;
  }
  return "Limite de passos atingido.";
}

reactLoop("Que horas são, e quanto é 15 * 8?").then(console.log);
```

A estrutura é idêntica à versão Python — a diferença mecânica é `Function(...)` no lugar de `eval` restrito (o equivalente mais próximo em JavaScript para avaliar uma expressão dinâmica com algum controle sobre o escopo, embora, assim como em Python, não seja uma sandbox real) e `match`/regex do JavaScript no lugar de `re.search`.

---

### Projeto 13.2 — Servidor MCP em Python e em TypeScript

Você vai construir um servidor MCP real, expondo uma tool, um resource e um prompt, e conectar a ele tanto programaticamente quanto (opcionalmente) via Claude Desktop.

**Pré-requisitos**: `pip install "mcp[cli]"`, um arquivo SQLite de exemplo (`sqlite3 exemplo.db "CREATE TABLE produtos (nome TEXT, preco REAL); INSERT INTO produtos VALUES ('Notebook', 3499.00), ('Mouse', 89.90);"`).

**1. Servidor MCP em Python**, usando o `FastMCP` (a API de alto nível do SDK oficial):

```python
from mcp.server.fastmcp import FastMCP
import sqlite3

mcp = FastMCP("servidor-produtos")

@mcp.tool()
def query_db(sql: str) -> str:
    """Executa uma query SQL somente-leitura no banco de produtos."""
    if not sql.strip().upper().startswith("SELECT"):
        return "Erro: apenas queries SELECT são permitidas."
    conn = sqlite3.connect("exemplo.db")
    try:
        cursor = conn.execute(sql)
        return str(cursor.fetchall())
    finally:
        conn.close()

@mcp.resource("file://produtos-doc")
def documentacao_produtos() -> str:
    """Documentação do schema da tabela produtos."""
    return "Tabela produtos: colunas 'nome' (TEXT) e 'preco' (REAL)."

@mcp.prompt()
def summarize_table(nome_tabela: str) -> str:
    return f"Resuma o conteúdo da tabela {nome_tabela}, destacando os 3 itens mais caros."

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

`@mcp.tool()` expõe a função `query_db` como uma tool MCP — o SDK gera automaticamente o schema JSON a partir da assinatura e das anotações de tipo da função, o mesmo tipo de schema que você escreveu manualmente em `ferramenta_extracao` no Projeto 11.5. A checagem `sql.strip().upper().startswith("SELECT")` é a mesma disciplina de "nunca confie no que o modelo/cliente manda executar" da seção 13.2, aplicada aqui: mesmo um servidor MCP não deveria executar SQL arbitrário sem alguma validação. `@mcp.resource` expõe dados referenciáveis por URI (aqui, uma string fixa; em um servidor real, poderia ler de um arquivo ou banco). `@mcp.prompt()` expõe um template parametrizável. `transport="stdio"` faz o servidor se comunicar via entrada/saída padrão — o transporte usado para servidores MCP locais, consumidos por um processo cliente na mesma máquina.

**2. Cliente programático**, para testar o servidor sem depender do Claude Desktop:

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    params = StdioServerParameters(command="python", args=["servidor_mcp.py"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            resultado = await session.call_tool("query_db", {"sql": "SELECT * FROM produtos"})
            print(resultado)

asyncio.run(main())
```

`stdio_client` inicia o servidor como um subprocesso e se comunica com ele via stdin/stdout — o cliente e o servidor são processos separados, exatamente como aconteceria com um cliente MCP real (Claude Desktop, um IDE) conectando ao seu servidor.

**3. Versão TypeScript**, usando o SDK oficial (`npm install @modelcontextprotocol/sdk zod`):

```typescript
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import Database from "better-sqlite3";

const server = new McpServer({ name: "servidor-produtos", version: "1.0.0" });

server.tool(
  "query_db",
  { sql: z.string() },
  async ({ sql }) => {
    if (!sql.trim().toUpperCase().startsWith("SELECT")) {
      return { content: [{ type: "text", text: "Erro: apenas SELECT é permitido." }] };
    }
    const db = new Database("exemplo.db");
    const resultado = db.prepare(sql).all();
    return { content: [{ type: "text", text: JSON.stringify(resultado) }] };
  }
);

const transport = new StdioServerTransport();
await server.connect(transport);
```

A estrutura é a mesma da versão Python — `z.string()` (Zod, já usado no Projeto 11.5 na versão TS de structured output) define o schema do argumento no lugar da anotação de tipo Python.

---

### Projeto 13.3 — Agente de pesquisa com LangGraph

Você vai construir um agente que pesquisa um tópico na web, busca o conteúdo das páginas relevantes, e escreve um relatório em Markdown citando fontes — com checkpointing, para poder retomar se falhar no meio.

**Pré-requisitos**: `pip install langgraph langchain-community duckduckgo-search`.

```python
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_ollama import ChatOllama
import requests

busca_web = DuckDuckGoSearchRun()

def fetch_url(url: str) -> str:
    """Baixa o texto de uma URL (sem parsing de HTML sofisticado — só o texto bruto, truncado)."""
    resposta = requests.get(url, timeout=10)
    return resposta.text[:3000]

def write_file(nome_arquivo: str, conteudo: str) -> str:
    """Salva conteúdo em um arquivo local."""
    with open(nome_arquivo, "w") as f:
        f.write(conteudo)
    return f"Arquivo {nome_arquivo} salvo."

modelo = ChatOllama(model="qwen2.5:7b")
checkpointer = MemorySaver()

agente = create_react_agent(
    modelo,
    tools=[busca_web, fetch_url, write_file],
    checkpointer=checkpointer,
)

config = {"configurable": {"thread_id": "pesquisa-1"}}
resultado = agente.invoke(
    {"messages": [{"role": "user", "content": "Pesquise sobre RMSNorm, escreva um relatório em Markdown em 'relatorio.md' citando as fontes usadas."}]},
    config=config,
)
print(resultado["messages"][-1].content)
```

`create_react_agent` é a implementação pronta do LangGraph para o mesmo loop ReAct que você escreveu à mão no Projeto 11.3 e estendeu no Projeto 13.1 — a diferença é que aqui as tools são objetos com schema declarado (via decorators ou classes do LangChain), e o loop de parsing/execução/observação é gerenciado pelo framework. `MemorySaver` é o checkpointing: o estado da conversa (`thread_id="pesquisa-1"`) é salvo a cada passo, então se o processo cair no meio da pesquisa, invocar `agente.invoke` de novo com o mesmo `thread_id` continua de onde parou, em vez de recomeçar.

---

### Projeto 13.4 — Multi-agente: pipeline editorial com CrewAI

Você vai orquestrar 4 agentes com papéis distintos (Pesquisador, Redator, Revisor, Editor) numa pipeline sequencial, e testar cada um isoladamente antes de encadeá-los.

**Pré-requisitos**: `pip install crewai crewai-tools`.

**1. Teste cada agente isoladamente primeiro** — rode cada um com uma entrada de exemplo fixa e confira a saída manualmente, antes de montar a pipeline completa. Um pipeline de 4 agentes onde um deles produz saída ruim silenciosamente é muito mais difícil de diagnosticar depois de tudo encadeado.

**2. Defina os 4 agentes e suas tarefas**:

```python
from crewai import Agent, Task, Crew, Process, LLM

modelo = LLM(model="ollama/qwen2.5:7b", base_url="http://localhost:11434")

pesquisador = Agent(
    role="Pesquisador",
    goal="Levantar fatos precisos e verificáveis sobre o tópico dado",
    backstory="Você é meticuloso e sempre cita de onde tirou cada informação.",
    llm=modelo,
)
redator = Agent(
    role="Redator",
    goal="Transformar os fatos levantados em um texto claro e bem estruturado",
    backstory="Você escreve para leitores técnicos, mas iniciantes no assunto específico.",
    llm=modelo,
)
revisor = Agent(
    role="Revisor",
    goal="Encontrar erros factuais, ambiguidades e frases confusas no texto",
    backstory="Você é cético por natureza — sua função é achar problemas, não elogiar.",
    llm=modelo,
)
editor = Agent(
    role="Editor",
    goal="Produzir a versão final, incorporando as correções do revisor",
    backstory="Você tem a palavra final sobre o que é publicado.",
    llm=modelo,
)

tarefa_pesquisa = Task(description="Levante 5 fatos sobre {topico}.", agent=pesquisador, expected_output="Lista de 5 fatos com fonte.")
tarefa_redacao = Task(description="Escreva um parágrafo introdutório sobre {topico} usando os fatos levantados.", agent=redator, expected_output="Um parágrafo de texto.")
tarefa_revisao = Task(description="Revise o parágrafo, listando problemas encontrados.", agent=revisor, expected_output="Lista de problemas, ou 'sem problemas'.")
tarefa_edicao = Task(description="Produza a versão final do parágrafo, corrigindo os problemas apontados.", agent=editor, expected_output="Parágrafo final.")

pipeline = Crew(
    agents=[pesquisador, redator, revisor, editor],
    tasks=[tarefa_pesquisa, tarefa_redacao, tarefa_revisao, tarefa_edicao],
    process=Process.sequential,
)

resultado = pipeline.kickoff(inputs={"topico": "RMSNorm"})
print(resultado)
```

`Process.sequential` executa as tarefas na ordem declarada, passando o contexto acumulado de uma tarefa para a próxima — a saída de `tarefa_pesquisa` fica disponível para `tarefa_redacao`, e assim por diante, sem você precisar encadear manualmente (o CrewAI gerencia essa passagem de contexto internamente). Cada `Agent` tem seu próprio `role`/`goal`/`backstory`, que vira parte do prompt daquele agente especificamente — é a persona (mod. 11) aplicada por agente, não ao sistema inteiro.

---

### Projeto 13.5 — Agente com code execution (análise de CSV)

Você vai usar um agente cuja "ação" é escrever e executar código Python diretamente, para analisar um CSV e gerar um gráfico — a abordagem `CodeAgent` do smolagents.

**Pré-requisitos**: `pip install smolagents pandas matplotlib`, um CSV de exemplo (pode ser os resultados do Projeto 8.4, `scaling_law.png` já tem os dados de origem, ou qualquer CSV com algumas colunas numéricas).

```python
from smolagents import CodeAgent, LiteLLMModel

modelo = LiteLLMModel(model_id="ollama/qwen2.5:7b", api_base="http://localhost:11434")
agente = CodeAgent(tools=[], model=modelo, additional_authorized_imports=["pandas", "matplotlib.pyplot"])

resultado = agente.run(
    "Carregue o arquivo 'dados.csv' com pandas, calcule a média e o desvio padrão de cada coluna numérica, "
    "e salve um histograma da primeira coluna numérica em 'histograma.png'."
)
print(resultado)
```

Diferente dos agentes anteriores (que escolhem entre um conjunto fixo de tools pré-definidas), o `CodeAgent` gera código Python livremente e o executa — mais flexível para tarefas de análise de dados (onde enumerar todas as operações possíveis como tools separadas seria impraticável), mas exatamente o tipo de caso que a seção 13.9 alerta: código gerado livremente por um LLM precisa rodar isolado. Por padrão, o `CodeAgent` do smolagents já roda num interpretador Python restrito (`additional_authorized_imports` lista explicitamente quais bibliotecas são permitidas, negando tudo o resto); para isolamento real de processo/sistema operacional (a mesma classe de proteção do sandboxing da seção 13.9), a biblioteca oferece integração com o e2b:

```python
from smolagents import CodeAgent, LiteLLMModel, E2BSandbox

agente_sandboxed = CodeAgent(
    tools=[], model=modelo, executor_type="e2b",
    additional_authorized_imports=["pandas", "matplotlib.pyplot"],
)
```

Com `executor_type="e2b"`, o código gerado roda numa máquina virtual descartável na nuvem (via conta e2b, que exige uma API key configurada como variável de ambiente), não no seu processo local — se o código gerado tentar algo destrutivo, o dano fica contido àquela VM temporária.

---

### Projeto 13.6 — Avaliação rigorosa: framework vs manual

Você vai construir um conjunto de 30 tarefas multi-passo com gabarito e comparar a implementação manual do Projeto 13.1 com a implementação em LangGraph do Projeto 13.3, medindo taxa de sucesso, número de passos, e custo.

**Pré-requisitos**: os Projetos 13.1 e 13.3 completos, `pip install langfuse`.

**1. Monte o eval set**: 30 tarefas que exigem 2-4 chamadas de ferramenta cada (ex.: "Qual é a população de São Paulo multiplicada pela hora atual em formato 24h?" — exige `busca` + `data_hora` + `calculadora`), com o resultado esperado.

**2. Instrumente as duas implementações com Langfuse**:

```python
from langfuse.decorators import observe

@observe()
def rodar_agente_manual(pergunta):
    return react_loop_com_log(pergunta)  # do Projeto 13.1

@observe()
def rodar_agente_langgraph(pergunta):
    resultado = agente.invoke({"messages": [{"role": "user", "content": pergunta}]}, config=config)
    return resultado["messages"][-1].content
```

O decorator `@observe()` captura automaticamente entradas, saídas, e (quando integrado com as chamadas ao modelo) tokens e latência de cada execução, enviando esse trace para o Langfuse — você pode inspecionar cada execução individualmente no painel do Langfuse depois, vendo exatamente quais ferramentas foram chamadas e em que ordem.

**3. Rode as 30 tarefas nas duas implementações e compare**:

```python
resultados = {"manual": {"sucessos": 0, "passos_totais": 0}, "langgraph": {"sucessos": 0, "passos_totais": 0}}

for tarefa, resposta_esperada in eval_set:
    for nome, fn in [("manual", rodar_agente_manual), ("langgraph", rodar_agente_langgraph)]:
        resposta = fn(tarefa)
        if resposta_esperada.lower() in resposta.lower():
            resultados[nome]["sucessos"] += 1

for nome, r in resultados.items():
    print(f"{nome}: {r['sucessos']}/30 sucessos")
```

Complemente essa contagem simples com o que o Langfuse capturou (custo estimado por tarefa, latência, número de chamadas ao modelo por tarefa) para uma comparação mais completa do que só taxa de sucesso — um framework pode ter taxa de sucesso igual à implementação manual, mas custar mais chamadas por tarefa (overhead do framework), o que só aparece olhando o trace.

---

### Projeto 13.7 — Agente de browser com Playwright

Você vai implementar, sem nenhuma biblioteca especializada em "browser use", um agente que navega até um site, captura o que está na tela, pede ao modelo a próxima ação, e executa — o mesmo padrão ReAct, agora com "a tela do navegador" como Observation.

**Pré-requisitos**: `pip install playwright && playwright install chromium`.

```python
from playwright.sync_api import sync_playwright
import requests

def descrever_pagina(page):
    """Extrai uma descrição textual simplificada dos elementos clicáveis da página."""
    elementos = page.eval_on_selector_all(
        "a, button, input",
        "els => els.map((e, i) => `[${i}] ${e.tagName} '${e.innerText || e.placeholder || ''}'`).join('\\n')"
    )
    return elementos

def agente_browser(objetivo, url_inicial, max_passos=8):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url_inicial)

        for passo in range(max_passos):
            descricao = descrever_pagina(page)
            prompt = (
                f"Objetivo: {objetivo}\nPágina atual, elementos disponíveis:\n{descricao}\n\n"
                "Responda com uma ação: CLICK <índice> ou TYPE <índice> <texto> ou DONE."
            )
            resposta = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "qwen2.5:7b", "prompt": prompt, "stream": False},
            ).json()["response"].strip()

            print(f"Passo {passo}: {resposta}")
            if resposta.startswith("DONE"):
                break
            elif resposta.startswith("CLICK"):
                indice = int(resposta.split()[1])
                page.eval_on_selector_all("a, button, input", f"els => els[{indice}].click()")
            elif resposta.startswith("TYPE"):
                partes = resposta.split(maxsplit=2)
                indice, texto = int(partes[1]), partes[2]
                page.eval_on_selector_all("a, button, input", f"els => els[{indice}].focus()")
                page.keyboard.type(texto)
            page.wait_for_timeout(1000)

        browser.close()

agente_browser("Encontrar o link 'Sobre' e clicar nele", "https://example.com")
```

`descrever_pagina` é uma alternativa simplificada (e bem mais barata) à abordagem de screenshot-como-imagem usada por Anthropic Computer Use e OpenAI Operator: em vez de enviar uma imagem da tela (cara em tokens, como mencionado na seção 13.10), extrai uma lista textual dos elementos interativos da página via JavaScript injetado (`eval_on_selector_all`), numerados, e pede ao modelo para referenciar esses números. É uma simplificação real — páginas com estruturas visuais complexas (um mapa, um canvas) não são bem representadas só por essa lista — mas é suficiente para navegação básica e muito mais barata e rápida que processar screenshots a cada passo.

**Documente quando falha**: rode o agente em 3-5 sites reais diferentes e anote os padrões de falha — páginas com muitos elementos (a lista fica longa demais para o modelo escolher bem), elementos sem texto identificável (`innerText` vazio), ou ações que exigem esperar um carregamento assíncrono que o `wait_for_timeout` fixo não cobre adequadamente.

---

## Erros comuns

- **Loop infinito**: agente repete a mesma ação. Sempre tenha **max_steps** (como em todos os projetos deste módulo) e detecção de loop.
- **Tools sem validação**: argumentos do LLM tratados como confiáveis = falha de segurança.
- **Sem observability**: debugar agente sem trace (seção 13.11, Projeto 13.6) é impossível.
- **Frameworks prematuros**: começar com CrewAI antes de entender ReAct manual — é por isso que este módulo pede o loop manual primeiro.
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

- [ ] Implementei loop ReAct manual em Python e em TS, com logging estruturado (se não, revise o Projeto 13.1).
- [ ] Construí servidor MCP funcional (tool + resource + prompt) e conectei via cliente programático (se não, revise o Projeto 13.2 e a seção 13.5).
- [ ] Construí agente multi-passo com framework (LangGraph), com checkpointing (se não, revise o Projeto 13.3).
- [ ] Testei cada agente de uma pipeline multi-agente isoladamente antes de encadear (se não, revise o Projeto 13.4 e a seção 13.8).
- [ ] Rodei um agente de code execution num ambiente restrito e entendo por que sandboxing real (e2b) é diferente de só restringir imports (se não, revise o Projeto 13.5 e a seção 13.9).
- [ ] Tenho observability (Langfuse) configurado e comparei framework vs implementação manual com números, não impressão (se não, revise o Projeto 13.6).
- [ ] Sei distinguir workflow de agente, e quando preferir cada um (se não, revise a seção 13.1).
- [ ] Entendo o risco de prompt injection em agentes com tools potentes (se não, revise a seção 13.2 e o mod. [11](11_prompt_engineering.md#118-prompt-injection-e-segurança)).
