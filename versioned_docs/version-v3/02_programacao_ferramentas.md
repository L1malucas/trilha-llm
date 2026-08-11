---
id: 02_programacao_ferramentas
title: "Módulo 02 — Programação e Ferramentas"
sidebar_position: 19
---

# Módulo 02 — Programação e Ferramentas

> **Objetivo**: ter um ambiente de trabalho profissional em Python e TypeScript, dominar as bibliotecas numéricas, e entender o hardware sob o qual ML roda — formalizando ferramentas que você já usa informalmente desde o início da trilha.
>
> **Pré-requisitos**: toda a trilha até aqui — você já escreveu Python extensivamente (desde o Projeto 8.1) e TypeScript desde o Projeto 10.2 (Node, `tsx`, Vercel AI SDK), já usou Docker (mod. 15), já usou GitHub Actions (Projeto 15.5), e já calculou VRAM de modelo em quantizações diferentes (seção 10.1). Este módulo formaliza e completa esse ferramental.
>
> **Tempo de referência**: 2–4 semanas.

## Objetivos de aprendizagem

Ao final deste módulo você deve ser capaz de:

- Montar um ambiente Python reprodutível (lockfile, container) que roda igual em qualquer máquina.
- Justificar, com critério, quando usar Python e quando usar TypeScript num projeto de IA — você já sentiu essa divisão na prática, aqui ela vira régua explícita.
- Calcular quanta VRAM um modelo precisa, em qualquer precisão, de cabeça.
- Explicar por que GPU acelera treinamento de rede neural especificamente (não é "GPU é mais rápido", é sobre o *tipo* de operação).
- Diagnosticar os erros de ambiente mais comuns (conflito CUDA/driver, mistura pip/conda) antes que eles te custem um dia inteiro.

---

## Por que isso importa

Ferramentas mal configuradas custam dias de debug. Você já sentiu, ao longo de toda a trilha, o custo de um ambiente mal montado (uma dependência de versão errada, um Docker que não sobe) — este módulo é sobre prevenir isso sistematicamente, não sobre aprender ferramentas do zero. Antes de treinar qualquer modelo, você precisa de um ambiente reprodutível, performático e portátil, e precisa entender a diferença entre rodar em CPU, GPU NVIDIA, GPU AMD, Apple Silicon e edge.

---

## 2.1 Python para ML/IA

### Núcleo da linguagem
- Tipagem (com `mypy`, `pyright`), `dataclasses`, `typing`, `Protocol`.
- Generators, `itertools`, comprehensions.
- Async (`asyncio`) — fundamental para servir LLMs, o mesmo princípio por trás do `asyncio.gather` que você usou no benchmark de concorrência do Projeto 10.3.
- Context managers, decorators — o mesmo `@observe()` do Langfuse (Projeto 13.6) é um decorator.

### Bibliotecas obrigatórias

| Biblioteca | Para que serve | Onde você já usou |
|---|---|---|
| **NumPy** | Computação numérica em arrays, base de todo o resto | Desde o Projeto 7.1 (attention em NumPy) |
| **SciPy** | Otimização, álgebra linear avançada, estatística | Projeto 15.6 (`ks_2samp`) |
| **Pandas** | Manipulação de dados tabulares | Projetos 3.x, 4.x |
| **Matplotlib / Seaborn / Plotly** | Visualização | Desde o Projeto 8.4 |
| **Jupyter / IPython** | Exploração e prototipagem | — |
| **scikit-learn** | ML clássico | Módulos 03, 04 |
| **PyTorch** | DL — padrão de fato em pesquisa | Desde o Projeto 8.3 |
| **JAX** | DL funcional, alternativa séria a PyTorch | — |

> **Por que importa especificamente aqui**: em código de ML, um bug de tipo costuma ser um bug de **shape** — uma matriz `(batch, 10)` sendo somada com uma `(10, batch)` por engano não quebra na hora, quebra 3 camadas depois com um erro de dimensão difícil de rastrear até a origem, ou pior, faz broadcasting silencioso e produz um resultado errado sem erro nenhum (o mesmo tipo de bug que os comentários de forma — `# (B, T, dim)` — no `MiniLlama` do Projeto 8.3 existem para prevenir). `mypy`/`pyright` não pegam erro de shape diretamente (Python não tem tipos de shape nativos), mas forçam disciplina de assinatura de função que reduz esse tipo de deslize. `asyncio` importa porque servir um LLM em produção significa lidar com múltiplas requisições de streaming simultâneas sem bloquear — exatamente o padrão do Projeto 10.3.
>
> **Checkpoint**: sem olhar o texto, explique por que um bug de shape em NumPy/PyTorch costuma ser mais perigoso que um erro de tipo comum (dica: pense em broadcasting).

### Gerenciamento de ambiente
- `venv` (padrão), `conda` (científico), **`uv`** (moderno, rapidíssimo, recomendado — usado no Projeto 2.1).
- `pyproject.toml` em vez de `requirements.txt` quando possível.
- `pip-tools` ou `uv` para lockfiles.

---

## 2.2 TypeScript para IA aplicada

### Núcleo
- Tipagem estrutural, generics, conditional types, mapped types.
- `zod` para validação de runtime (essencial para LLM outputs) — você já usou isso no Projeto 13.2 (schema de tools do servidor MCP).
- `tsx` (o runtime que você usa desde o Projeto 10.2), `bun`, `deno` como alternativas a Node puro.

### Bibliotecas relevantes para IA

| Biblioteca | Para que serve | Onde você já usou |
|---|---|---|
| **Vercel AI SDK** (`ai`) | Streaming, tool use, agnóstico de provedor | Desde o Projeto 10.2 |
| **LangChain.js** | Chains, agents, integrações | — |
| **LlamaIndex.TS** | RAG e ingestão de documentos | — |
| **transformers.js** | Inferência de modelos no browser/Node via ONNX | Desde o Projeto 10.5 |
| **ONNX Runtime Web** | Inferência otimizada client-side | (por baixo do transformers.js) |
| **Mastra** | Framework de agentes em TS | — |
| **MCP SDK (TypeScript)** | Model Context Protocol oficial | Projeto 13.2 |

### Quando TS é melhor que Python
- Aplicações web full-stack com streaming de LLM (Next.js, SvelteKit) — o padrão do Projeto 10.2.
- Edge computing (Vercel Edge, Cloudflare Workers) — o Projeto 15.4 comparou isso diretamente.
- Serverless functions.
- Agentes embutidos em SaaS existente.

### Quando Python é incontornável
- Treinamento de modelos.
- Fine-tuning (LoRA, QLoRA, full) — mod. 09.
- Pesquisa e reprodução de papers.
- Pipelines de dados em larga escala.
- Avaliação científica de modelos.

**Equivalência prática**: para qualquer pipeline de RAG ou agente em produção, existe uma versão TS razoável — você mesmo construiu isso no Projeto 12.1 (RAG em Python e TS) e no Projeto 13.1 (ReAct em Python e TS). Para qualquer coisa que envolva treinar pesos, vá de Python.

> **Intuição**: a divisão não é arbitrária — é sobre *onde* o ecossistema de cada linguagem é mais maduro. O ecossistema científico Python (NumPy, PyTorch, CUDA bindings) existe há mais de uma década e é onde toda pesquisa de ML acontece primeiro; frameworks novos (arquiteturas, técnicas de fine-tuning) chegam em Python meses ou anos antes de terem equivalente em TS, se algum dia tiverem. Já o ecossistema de aplicações web (streaming, edge, tooling de frontend) é onde TS historicamente é mais forte. A régua prática: se a tarefa é "mexer nos pesos do modelo" (treinar, ajustar), é território Python; se é "orquestrar chamadas a um modelo já treinado" (chamar API, montar agente, servir numa aplicação web), TS é uma escolha tão boa quanto, e às vezes melhor por causa do ecossistema de deploy/edge — exatamente a régua que você já aplicou intuitivamente ao longo de toda a trilha, sem tê-la ainda visto formalizada.
>
> **Checkpoint**: sem olhar o texto, explique a régua de decisão Python vs TS em uma frase — e dê um exemplo de tarefa de cada lado, idealmente um projeto que você já fez.

---

## 2.3 Engenharia de software para ML

### Controle de versão
- Git avançado: rebase, cherry-pick, bisect, hooks.
- **DVC** (Data Version Control) — versionar datasets e modelos, usado no Projeto 2.1.
- Git LFS para modelos pequenos.
- **Hugging Face Hub** como repositório de modelos/datasets (usa Git+LFS) — você já baixou modelos de lá desde o Projeto 9.1.

### Containers e orquestração
- Docker: imagens, multi-stage builds, otimização de cache — você já escreveu um `Dockerfile` no Projeto 15.4.
- Docker Compose para pilhas locais (LLM + vector DB + app) — o mesmo `docker-compose.yml` do Langfuse no Projeto 15.1.
- Conhecimento básico de Kubernetes (não precisa ser expert).

### Reprodutibilidade
- **Weights & Biases (W&B)** — você já usou no Projeto 5.6, **MLflow**, ou **Aim** — tracking de experimentos.
- Seeds determinísticos.
- Documentação de versões de CUDA, drivers, bibliotecas.

> **Intuição**: engenharia de software tradicional versiona código; ML precisa versionar **código + dados + modelo + configuração** simultaneamente, porque o mesmo código com um dataset diferente (ou uma seed diferente) produz um modelo diferente. "Funciona na minha máquina" em ML costuma significar "funciona com a versão do dataset que só existe na minha máquina" — daí ferramentas como DVC (que trata dados grandes como Git trata código, sem inflar o repositório) e trackers de experimento como o W&B que você já usou (que registram exatamente qual combinação de código+dados+hiperparâmetros gerou qual resultado, pra você conseguir voltar a qualquer ponto).
>
> **Checkpoint**: sem olhar o texto, explique por que "versionar só o código" não é suficiente para reproduzir um experimento de ML.

---

## 2.4 Hardware e aceleração

### CPU vs GPU vs TPU
- Por que GPU acelera ML: paralelismo massivo em operações matriciais.
- Quando CPU basta: inferência de modelos pequenos, dados tabulares.
- TPU (Google) — relevante se for usar JAX em larga escala.

### Plataformas
- **NVIDIA CUDA** (padrão de fato, melhor suporte).
- **AMD ROCm** (alternativa em Linux, suporte crescente).
- **Apple Silicon (MPS)** — `torch.backends.mps`, suporte parcial mas funcional, já mencionado desde o Projeto 8.3 (`.to("mps")`).
- **Intel oneAPI**, **Vulkan compute** (nichos).

> **Intuição — por que GPU**: uma CPU tem poucos núcleos (dezenas), cada um muito rápido e flexível, ótimo para tarefas sequenciais complexas. Uma GPU tem milhares de núcleos simples, cada um lento individualmente, mas todos fazendo a *mesma* operação em paralelo sobre dados diferentes — exatamente o padrão de um produto matricial, onde cada elemento da saída é um produto interno independente dos outros (o mesmo `Q @ K.T` que você já escreveu no Projeto 7.1). É por isso que GPU não acelera qualquer código, só código que se encaixa nesse padrão de paralelismo massivo e uniforme — um loop com lógica condicional complexa por iteração não ganha o mesmo benefício, e é exatamente por isso que o LSTM do Projeto 5.4 (sequencial, estado carregado passo a passo) ganha menos com GPU do que o Transformer do Projeto 8.3 (paralelizável).

### VRAM como limite
- Regra de bolso: parâmetros × bytes por parâmetro.
- FP32 = 4 bytes, FP16/BF16 = 2 bytes, INT8 = 1 byte, INT4 = 0.5 byte.
- Modelo 7B em FP16 = ~14 GB; em INT4 = ~3.5 GB. Isso define o que cabe na sua placa — a mesma conta da seção 10.1, revisitada aqui na origem.
- Treinamento exige ~3–4× a memória de inferência (gradientes + estados de otimizador).

> **Exemplo resolvido**: quanto de VRAM um modelo de 13B parâmetros precisa só para **inferência** em BF16? `13.000.000.000 parâmetros × 2 bytes = 26.000.000.000 bytes ≈ 26 GB` — não cabe numa GPU de consumo comum de 24GB, mas cabe (com folga apertada) numa A100 de 40GB. Em INT4, o mesmo modelo cai para `13B × 0.5 byte ≈ 6.5 GB` — cabe até em GPUs de laptop. Isso é a conta que decide, antes de qualquer outra consideração, se um modelo roda na sua máquina — a mesma que você já aplicou na prática desde o mod. 10.
>
> **Checkpoint**: sem olhar o texto, calcule quanta VRAM um modelo de 7B parâmetros precisa em FP32 — e explique por que treinamento precisa de vários múltiplos a mais de memória que inferência.

### Cloud para quem não tem GPU
Google Colab (free tier com T4), Kaggle Notebooks (free tier com P100/T4), Lambda Labs/RunPod/Vast.ai (alugar por hora), e Hugging Face Spaces (deploy gratuito de demos) — você já pode ter usado algum desses ao longo dos projetos mais pesados da trilha (fine-tuning de VLM no mod. 18, geração de vídeo no Projeto 19.7).

---

## Projetos práticos

### Projeto 2.1 — Setup completo reprodutível

Você vai formalizar, num único repositório modelo, toda a disciplina de reprodutibilidade que você já aplicou informalmente ao longo da trilha.

**Pré-requisitos**: `pip install uv` (ou `curl -LsSf https://astral.sh/uv/install.sh | sh`), Docker.

**1. `pyproject.toml`** (declarando dependências e metadados do projeto, no lugar de um `requirements.txt` solto):

```toml
[project]
name = "meu-projeto-llm"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.2",
    "transformers>=4.40",
    "requests>=2.31",
]

[project.optional-dependencies]
dev = ["pytest", "ruff", "mypy", "pre-commit"]
```

**2. Lockfile reprodutível com `uv`**:

```bash
uv venv
uv pip install -e ".[dev]"
uv lock   # gera uv.lock — fixa a versão exata de cada dependência transitiva, não só a declarada
```

O lockfile é o que garante que "funciona na minha máquina" vire "funciona em qualquer máquina que rode `uv sync`" — sem ele, `torch>=2.2` no `pyproject.toml` pode resolver para versões diferentes em máquinas diferentes, dependendo de quando cada uma instalou.

**3. `Dockerfile`** (multi-stage, reaproveitando o padrão do Projeto 15.4):

```dockerfile
FROM python:3.11-slim AS base
RUN pip install uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

FROM base AS final
COPY . .
CMD ["uv", "run", "python", "main.py"]
```

`--frozen` faz `uv sync` falhar (em vez de silenciosamente resolver versões novas) se o `uv.lock` não bater exatamente com o `pyproject.toml` — uma proteção extra contra builds não-reprodutíveis.

**4. Pre-commit hooks** (`.pre-commit-config.yaml`), rodando automaticamente antes de cada commit:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.0
    hooks:
      - id: ruff
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.9.0
    hooks:
      - id: mypy
```

`pip install pre-commit && pre-commit install` ativa isso — a partir daí, `ruff` (lint) e `mypy` (checagem de tipos) rodam automaticamente a cada `git commit`, recusando o commit se algo falhar.

**5. CI básico** (`.github/workflows/ci.yml`, o mesmo padrão do Projeto 15.5, agora para testes gerais em vez de eval de prompt):

```yaml
name: CI
on: [pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install uv && uv sync --frozen
      - run: uv run pytest
      - run: uv run ruff check .
```

**Valide o "clone em outra máquina" literalmente**: peça pra alguém (ou use uma VM limpa, ou um container Docker sem nada além do que o `Dockerfile` traz) clonar seu repositório e rodar só os comandos documentados no README (`uv sync`, `docker build`). Qualquer passo que você "esqueceu" de documentar porque já estava configurado na sua própria máquina vai aparecer exatamente aqui.

---

### Projeto 2.2 — Análise exploratória profissional

Você vai conduzir uma EDA (Exploratory Data Analysis) completa e documentada, no padrão que precede qualquer um dos projetos de ML tabular que você já fez nos módulos 03/04.

**Pré-requisitos**: `pip install pandas plotly jupyter`.

```python
import pandas as pd
import plotly.express as px

df = pd.read_csv("dataset.csv")

# 1. Visão geral: tipos, nulos, cardinalidade
print(df.info())
print(df.isnull().sum().sort_values(ascending=False))
print(df.nunique())

# 2. Distribuições das variáveis numéricas
for coluna in df.select_dtypes(include="number").columns:
    fig = px.histogram(df, x=coluna, title=f"Distribuição de {coluna}")
    fig.show()

# 3. Correlações
fig_corr = px.imshow(df.select_dtypes(include="number").corr(), text_auto=True, title="Matriz de correlação")
fig_corr.show()

# 4. Relação de cada feature com o target
for coluna in df.select_dtypes(include="number").columns:
    if coluna != "target":
        fig = px.box(df, x="target", y=coluna, title=f"{coluna} por classe do target")
        fig.show()
```

Cada bloco responde uma pergunta específica de diagnóstico: `df.info()`/`isnull().sum()` revela problemas estruturais (colunas com tipo errado, nulos concentrados); os histogramas revelam distribuições assimétricas ou outliers extremos (candidatos a normalização ou remoção); a matriz de correlação revela redundância entre features (duas colunas quase perfeitamente correlacionadas raramente precisam estar as duas no modelo); os box plots por classe revelam quais features parecem discriminativas antes mesmo de treinar qualquer modelo.

**Documente hipóteses e decisões em Markdown** conforme avança — não só os gráficos, mas o raciocínio: "a coluna X tem 40% de nulos, provavelmente porque [hipótese]; decisão: [imputar/remover], porque [razão]". Exporte o notebook e esse relatório junto — a EDA só tem valor para outra pessoa (ou para você, 3 meses depois) se o raciocínio estiver registrado, não só os gráficos finais.

---

### Projeto 2.3 — DX comparada: mesmo modelo em Python e TypeScript

Você já fez essa comparação várias vezes na prática (Projetos 12.1, 13.1, 16.7, 18.1) — aqui, o objetivo é refletir explicitamente sobre a experiência de desenvolvedor (DX) das duas pilhas, não implementar de novo.

**Pré-requisitos**: nenhum novo — reaproveite os projetos já feitos.

**Reflita e documente**, comparando pelo menos dois projetos que você já implementou nas duas linguagens:

1. **Setup**: quantos comandos até rodar o "hello world" de cada ecossistema? (`uv venv && uv pip install` vs `npm init && npm install`)
2. **Erros em tempo de desenvolvimento**: o type-checker de qual linguagem pegou mais bugs antes de rodar? (TS costuma ganhar aqui, especialmente com `zod` validando shape de dado em runtime também)
3. **Velocidade de iteração**: qual ecossistema tem feedback mais rápido (hot reload, tempo de start)?
4. **Maturidade de biblioteca**: para a tarefa específica que você implementou, qual das duas versões (Python ou TS) exigiu mais código para o mesmo resultado, e por quê?

**Se quiser um experimento novo além da reflexão**: carregue um modelo pequeno de classificação (`distilbert-base-uncased`, por exemplo) via `transformers` em Python e via `transformers.js` em TS, rode a mesma entrada nos dois, e confirme que os outputs batem (ou são consistentes) antes de comparar latência — sem essa validação, uma diferença de latência pode estar escondendo uma diferença real de comportamento entre as duas implementações, não só velocidade.

---

### Projeto 2.4 — Benchmark de hardware

Você vai medir, com números, o ganho de GPU sobre CPU (e MPS, se tiver Apple Silicon) no treino de uma rede pequena — confirmando empiricamente a Intuição da seção 2.4.

**Pré-requisitos**: o código do Projeto 5.1 (MLP) ou do Projeto 5.2 (CNN), adaptado para rodar em diferentes dispositivos.

```python
import torch
import time

def benchmark_treino(dispositivo, modelo_fn, X, y, n_passos=100):
    device = torch.device(dispositivo)
    modelo = modelo_fn().to(device)
    X, y = X.to(device), y.to(device)
    optimizer = torch.optim.AdamW(modelo.parameters(), lr=1e-3)

    if dispositivo != "cpu":
        torch.cuda.synchronize() if dispositivo == "cuda" else None  # garante que a GPU terminou operações pendentes antes de cronometrar

    inicio = time.time()
    for _ in range(n_passos):
        logits = modelo(X)
        loss = torch.nn.functional.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if dispositivo != "cpu":
        torch.cuda.synchronize() if dispositivo == "cuda" else None
    tempo_total = time.time() - inicio

    return tempo_total, tempo_total / n_passos

dispositivos_disponiveis = ["cpu"]
if torch.cuda.is_available():
    dispositivos_disponiveis.append("cuda")
if torch.backends.mps.is_available():
    dispositivos_disponiveis.append("mps")

for dispositivo in dispositivos_disponiveis:
    tempo_total, tempo_por_passo = benchmark_treino(dispositivo, lambda: CNNSimples(), X_treino_batch, y_treino_batch)
    print(f"{dispositivo}: {tempo_total:.2f}s total, {tempo_por_passo*1000:.2f}ms/passo")
```

`torch.cuda.synchronize()` é necessário porque operações em GPU são assíncronas por padrão — o código Python "termina" a chamada `loss.backward()` antes da GPU necessariamente ter concluído o trabalho, então cronometrar sem sincronizar mediria só o tempo de *disparar* as operações, não de completá-las (um erro de benchmark comum e enganoso).

**Antes de rodar, formule uma hipótese**: quantas vezes mais rápido você espera que a GPU seja, baseado na Intuição de paralelismo da seção 2.4? Depois compare com o número real — para redes muito pequenas (como o MLP do Projeto 5.1 em MNIST), o ganho de GPU pode ser menor do que se espera, ou até negativo, porque o overhead de transferir dados entre CPU e GPU a cada passo supera o ganho de paralelismo quando o cálculo em si é pequeno demais; para modelos maiores (uma CNN do Projeto 5.2, ou o `MiniLlama` do Projeto 8.3), o ganho deve ficar mais evidente. Documentar onde sua previsão errou é mais instrutivo do que só confirmar o óbvio.

---

## Erros comuns

- **Misturar `pip` e `conda` no mesmo ambiente.** Caos garantido.
- **Ignorar versões de CUDA.** PyTorch + CUDA + driver têm matriz de compatibilidade restrita.
- **Não fixar seeds.** Resultados não reproduzíveis = não-ciência.
- **Subestimar Docker.** Sem container, "funciona na minha máquina" é a regra.
- **Tentar treinar tudo localmente.** Saiba quando subir para cloud.

---

## Checklist de saída

- [ ] Tenho um setup Python e TS reprodutíveis em qualquer máquina, validado por clone limpo (se não, revise o Projeto 2.1).
- [ ] Sei diferenciar quando usar Python vs TS para uma tarefa de IA, com exemplos próprios já implementados (se não, revise a seção 2.2 e o Projeto 2.3).
- [ ] Entendo o que é VRAM e sei calcular quanta um modelo precisa, de cabeça (se não, revise a seção 2.4).
- [ ] Consigo rodar um modelo Hugging Face em CPU, GPU local, e medir a diferença real, não só teórica (se não, revise o Projeto 2.4).
- [ ] Tenho um workflow de versionamento de código + dados + modelos (se não, revise a seção 2.3 e o Projeto 2.1).
