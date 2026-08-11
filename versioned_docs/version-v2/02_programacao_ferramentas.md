---
id: 02_programacao_ferramentas
title: "Módulo 02 — Programação e Ferramentas"
sidebar_position: 2
---

# Módulo 02 — Programação e Ferramentas

> **Objetivo**: ter um ambiente de trabalho profissional em Python e TypeScript, dominar as bibliotecas numéricas, e entender o hardware sob o qual ML roda.
>
> **Pré-requisitos**: Módulo [01](01_matematica.md) (matemática), conhecimento prévio de programação.
>
> **Tempo de referência**: 2–4 semanas.

## Objetivos de aprendizagem

Ao final deste módulo você deve ser capaz de:

- Montar um ambiente Python reprodutível (lockfile, container) que roda igual em qualquer máquina.
- Justificar, com critério, quando usar Python e quando usar TypeScript num projeto de IA — não só citar a lista.
- Calcular quanta VRAM (Video RAM, a memória da GPU) um modelo precisa, em qualquer precisão, de cabeça.
- Explicar por que GPU acelera treinamento de rede neural especificamente (não é "GPU é mais rápido", é sobre o *tipo* de operação).
- Diagnosticar os erros de ambiente mais comuns (conflito CUDA/driver, mistura pip/conda) antes que eles te custem um dia inteiro.

---

## Por que isso importa

Ferramentas mal configuradas custam **dias** de debug. Antes de treinar qualquer modelo, você precisa de um ambiente reprodutível, performático e portátil. E precisa entender a diferença entre rodar em CPU, GPU NVIDIA, GPU AMD, Apple Silicon e edge.

---

## 2.1 Python para ML/IA

### Núcleo da linguagem
- Tipagem (com `mypy`, `pyright`), `dataclasses`, `typing`, `Protocol`.
- Generators, `itertools`, comprehensions.
- Async (`asyncio`) — fundamental para servir LLMs.
- Context managers, decorators.

### Bibliotecas obrigatórias

| Biblioteca | Para que serve |
|---|---|
| **NumPy** | Computação numérica em arrays, base de todo o resto |
| **SciPy** | Otimização, álgebra linear avançada, estatística |
| **Pandas** | Manipulação de dados tabulares |
| **Matplotlib / Seaborn / Plotly** | Visualização |
| **Jupyter / IPython** | Exploração e prototipagem |
| **scikit-learn** | ML clássico |
| **PyTorch** | DL — padrão de fato em pesquisa |
| **JAX** | DL funcional, alternativa séria a PyTorch |

> **Intuição — generators e context managers**: um `generator` (função com `yield`) produz valores um de cada vez, sob demanda, sem materializar a sequência inteira em memória — essencial ao iterar sobre um dataset maior que a RAM disponível. Um `context manager` (`with ... as ...`) garante que um recurso (arquivo, conexão, sessão de GPU) seja liberado mesmo se o código dentro do bloco lançar uma exceção — o mesmo padrão por trás de `torch.no_grad()`, que desliga o cálculo de gradiente só durante o bloco, sem exigir lembrar de reativá-lo depois. Um `decorator` (`@algo`) envolve uma função com comportamento extra sem reescrever a função em si — o `@observe()` de rastreamento de experimentos (seção 2.3) e o `@app.route` de um servidor web são o mesmo mecanismo.
>
> **Por que importa especificamente aqui**: em código de ML, um bug de tipo costuma ser um bug de **shape** — uma matriz `(batch, 10)` sendo somada com uma `(10, batch)` por engano não quebra na hora, quebra 3 camadas depois com um erro de dimensão difícil de rastrear até a origem, ou pior, faz broadcasting silencioso e produz um resultado errado sem erro nenhum. `mypy`/`pyright` não pegam erro de shape diretamente (Python não tem tipos de shape nativos), mas forçam disciplina de assinatura de função que reduz esse tipo de deslize. `asyncio` importa porque servir um LLM em produção significa lidar com múltiplas requisições de streaming simultâneas sem bloquear — é o mesmo padrão usado em qualquer servidor de I/O intensivo, aplicado a tokens em vez de bytes de rede.
>
> **Checkpoint**: sem olhar o texto, explique por que um bug de shape em NumPy/PyTorch costuma ser mais perigoso que um erro de tipo comum (dica: pense em broadcasting).

### Gerenciamento de ambiente
- `venv` (padrão), `conda` (científico), **`uv`** (moderno, rapidíssimo, recomendado).
- `pyproject.toml` em vez de `requirements.txt` quando possível.
- `pip-tools` ou `uv` para lockfiles.

> Quer ver esse ambiente montado de ponta a ponta — `pyproject.toml`, lockfile, `Dockerfile` multi-stage, pre-commit e CI (Continuous Integration) rodando, com cada arquivo completo? A versão Acelerado tem tudo isso no [Projeto 2.1 — Setup completo reprodutível](/trilha-llm/02_programacao_ferramentas#projeto-21--setup-completo-reprodutível).

### Referências
- `Livro` **Fluent Python (2nd ed.)** — Luciano Ramalho.
- `Curso` **NumPy Documentation — User Guide** (oficial). https://numpy.org/doc/stable/user/
- `Curso` **PyTorch Tutorials** (oficial). https://pytorch.org/tutorials/
- `Ferramenta` **uv** (gerenciador de pacotes/ambientes). https://docs.astral.sh/uv/

---

## 2.2 TypeScript para IA aplicada

### Núcleo
- Tipagem estrutural, generics, conditional types, mapped types.
- `zod` para validação de runtime (essencial para LLM outputs).
- `tsx`, `bun`, `deno` como runtimes alternativos a Node.

### Bibliotecas relevantes para IA

| Biblioteca | Para que serve |
|---|---|
| **Vercel AI SDK** (`ai`) | Streaming, tool use, agnóstico de provedor |
| **LangChain.js** | Chains, agents, integrações |
| **LlamaIndex.TS** | RAG e ingestão de documentos |
| **transformers.js** | Inferência de modelos no browser/Node via ONNX (Open Neural Network Exchange) |
| **ONNX Runtime Web** | Inferência otimizada client-side |
| **Mastra** | Framework de agentes em TS |
| **MCP SDK (TypeScript)** | Model Context Protocol oficial |

### Quando TS é melhor que Python
- Aplicações web full-stack com streaming de LLM (Next.js, SvelteKit).
- Edge computing (Vercel Edge, Cloudflare Workers).
- Serverless functions.
- Agentes embutidos em SaaS existente.

### Quando Python é incontornável
- Treinamento de modelos.
- Fine-tuning (LoRA, QLoRA, full).
- Pesquisa e reprodução de papers.
- Pipelines de dados em larga escala.
- Avaliação científica de modelos.

**Equivalência prática**: para qualquer pipeline de RAG ou agente em produção, existe uma versão TS razoável. Para qualquer coisa que envolva treinar pesos, vá de Python.

> **Intuição**: a divisão não é arbitrária — é sobre *onde* o ecossistema de cada linguagem é mais maduro. O ecossistema científico Python (NumPy, PyTorch, CUDA bindings) existe há mais de uma década e é onde toda pesquisa de ML acontece primeiro; frameworks novos (arquiteturas, técnicas de fine-tuning) chegam em Python meses ou anos antes de terem equivalente em TS, se algum dia tiverem. Já o ecossistema de aplicações web (streaming, edge, tooling de frontend) é onde TS historicamente é mais forte. A régua prática: se a tarefa é "mexer nos pesos do modelo" (treinar, ajustar), é território Python; se é "orquestrar chamadas a um modelo já treinado" (chamar API, montar agente, servir numa aplicação web), TS é uma escolha tão boa quanto, e às vezes melhor por causa do ecossistema de deploy/edge.
>
> **Checkpoint**: sem olhar o texto, explique a régua de decisão Python vs TS em uma frase — e dê um exemplo de tarefa de cada lado.

> Quer ver essa régua aplicada de verdade — o mesmo modelo rodando em PyTorch e em `transformers.js`, com DX (Developer Experience) comparada em setup, erros de tipo e velocidade de iteração? A versão Acelerado cobre isso no [Projeto 2.3 — DX comparada](/trilha-llm/02_programacao_ferramentas#projeto-23--dx-comparada-mesmo-modelo-em-python-e-typescript).

### Referências
- `Livro` **Effective TypeScript** — Dan Vanderkam.
- `Ferramenta` **Vercel AI SDK Documentation**. https://ai-sdk.dev/
- `Ferramenta` **transformers.js**. https://huggingface.co/docs/transformers.js
- `Ferramenta` **MCP TypeScript SDK**. https://github.com/modelcontextprotocol/typescript-sdk

---

## 2.3 Engenharia de software para ML

### Controle de versão
- Git avançado: rebase, cherry-pick, bisect, hooks.
- **DVC** (Data Version Control) — versionar datasets e modelos. https://dvc.org/
- Git LFS (Large File Storage) para modelos pequenos.
- **Hugging Face Hub** como repositório de modelos/datasets (usa Git+LFS).

### Containers e orquestração
- Docker: imagens, multi-stage builds, otimização de cache.
- Docker Compose para pilhas locais (LLM + vector DB + app).
- Conhecimento básico de Kubernetes (não precisa ser expert).

### Reprodutibilidade
- **Weights & Biases (W&B)**, **MLflow**, ou **Aim** — tracking de experimentos.
- Seeds determinísticos.
- Documentação de versões de CUDA, drivers, bibliotecas.

> **Intuição**: engenharia de software tradicional versiona código; ML precisa versionar **código + dados + modelo + configuração** simultaneamente, porque o mesmo código com um dataset diferente (ou uma seed diferente) produz um modelo diferente. "Funciona na minha máquina" em ML costuma significar "funciona com a versão do dataset que só existe na minha máquina" — daí ferramentas como DVC (que trata dados grandes como Git trata código, sem inflar o repositório) e trackers de experimento (que registram exatamente qual combinação de código+dados+hiperparâmetros gerou qual resultado, pra você conseguir voltar a qualquer ponto).
>
> **Checkpoint**: sem olhar o texto, explique por que "versionar só o código" não é suficiente para reproduzir um experimento de ML.

### Referências
- `Livro` **Designing Machine Learning Systems** — Chip Huyen.
- `Curso` **Made With ML — MLOps Course**. https://madewithml.com/
- `Ferramenta` **MLflow Documentation**. https://mlflow.org/

---

## 2.4 Hardware e aceleração

### CPU vs GPU vs TPU
- Por que GPU acelera ML: paralelismo massivo em operações matriciais.
- Quando CPU basta: inferência de modelos pequenos, dados tabulares.
- TPU (Tensor Processing Unit, chip da Google feito sob medida para álgebra linear de ML) — relevante se for usar JAX em larga escala.

### Plataformas
- **NVIDIA CUDA** (Compute Unified Device Architecture — padrão de fato, melhor suporte).
- **AMD ROCm** (Radeon Open Compute — alternativa em Linux, suporte crescente).
- **Apple Silicon (MPS, Metal Performance Shaders)** — `torch.backends.mps`, suporte parcial mas funcional.
- **Intel oneAPI**, **Vulkan compute** (nichos).

> **Intuição — por que GPU**: uma CPU tem poucos núcleos (dezenas), cada um muito rápido e flexível, ótimo para tarefas sequenciais complexas. Uma GPU tem milhares de núcleos simples, cada um lento individualmente, mas todos fazendo a *mesma* operação em paralelo sobre dados diferentes — exatamente o padrão de um produto matricial (mod. [01](01_matematica.md#11-álgebra-linear)), onde cada elemento da saída é um produto interno independente dos outros. É por isso que GPU não acelera qualquer código, só código que se encaixa nesse padrão de paralelismo massivo e uniforme — um loop com lógica condicional complexa por iteração não ganha o mesmo benefício.

### VRAM como limite
- Regra de bolso: parâmetros × bytes por parâmetro.
- FP32 = 4 bytes, FP16/BF16 = 2 bytes, INT8 = 1 byte, INT4 = 0.5 byte.
- Modelo 7B em FP16 = ~14 GB; em INT4 = ~3.5 GB. Isso define o que cabe na sua placa.
- Treinamento exige ~3–4× a memória de inferência (gradientes + estados de otimizador).

> **Exemplo resolvido**: quanto de VRAM um modelo de 13B parâmetros precisa só para **inferência** em BF16? `13.000.000.000 parâmetros × 2 bytes = 26.000.000.000 bytes ≈ 26 GB` — não cabe numa GPU de consumo comum de 24GB, mas cabe (com folga apertada) numa A100 de 40GB. Em INT4, o mesmo modelo cai para `13B × 0.5 byte ≈ 6.5 GB` — cabe até em GPUs de laptop. Isso é a conta que decide, antes de qualquer outra consideração, se um modelo roda na sua máquina — o mod. [10](10_eficiencia_e_inferencia_local.md) aprofunda quantização (por que INT4 funciona sem destruir a qualidade do modelo) e outras técnicas para espremer modelos maiores em menos memória.
>
> **Checkpoint**: sem olhar o texto, calcule quanta VRAM um modelo de 7B parâmetros precisa em FP32 — e explique por que treinamento precisa de vários múltiplos a mais de memória que inferência.

> Quer ver essa conta de VRAM virar um benchmark real — o mesmo treino cronometrado em CPU, GPU e MPS, com a hipótese de speedup formulada antes de medir? A versão Acelerado tem o código completo no [Projeto 2.4 — Benchmark de hardware](/trilha-llm/02_programacao_ferramentas#projeto-24--benchmark-de-hardware).

### Cloud para quem não tem GPU
- Google Colab (free tier com T4).
- Kaggle Notebooks (free tier com P100/T4).
- Lambda Labs, RunPod, Vast.ai (alugar por hora, $0.30–$2/h em T4–A100).
- Hugging Face Spaces (deploy gratuito de demos).

### Referências
- `Curso` **NVIDIA Deep Learning Institute** (cursos gratuitos sobre CUDA, otimização). https://www.nvidia.com/en-us/training/online/
- `Paper` **Mixed Precision Training** — Micikevicius et al. (2017). https://arxiv.org/abs/1710.03740
- `Livro` **Programming Massively Parallel Processors** — Kirk & Hwu (CUDA, opcional, profundidade).

---

## Projetos práticos

### Projeto 2.1 — Setup completo reprodutível
- Crie um repo com: `pyproject.toml`, lockfile (`uv` ou `pip-tools`), Dockerfile, `Makefile`.
- Configure pre-commit (`black`, `ruff`, `mypy`).
- Adicione um workflow CI básico (GitHub Actions) que rode testes.
- **Teste**: clone em outra máquina, rode tudo do zero.

> **Variante guiada**: valide o "clone em outra máquina" literalmente, não como formalidade — peça pra alguém (ou use uma VM limpa) clonar seu repo e rodar só os comandos documentados no README. Qualquer passo que você "esqueceu" de documentar porque já estava configurado na sua máquina vai aparecer aqui.

### Projeto 2.2 — Análise exploratória profissional
- Escolha um dataset (Kaggle ou UCI ML Repo).
- Faça EDA (Exploratory Data Analysis) com Pandas + Plotly em Jupyter.
- Documente hipóteses, anomalias, decisões de limpeza.
- Exporte como notebook + relatório em Markdown.

> Quer ver essa EDA implementada passo a passo — visão geral, distribuições, matriz de correlação e relação de cada feature com o target, cada bloco com a pergunta de diagnóstico que ele responde? A versão Acelerado tem o código completo no [Projeto 2.2 — Análise exploratória profissional](/trilha-llm/02_programacao_ferramentas#projeto-22--análise-exploratória-profissional).

### Projeto 2.3 — Mesmo problema em PyTorch e em TS (transformers.js)
- Carregue um modelo pequeno (ex.: `distilbert-base-uncased` para classificação).
- Faça inferência em Python (PyTorch) e em TS (transformers.js).
- Compare latência, output, DX.
- **Objetivo**: sentir as diferenças concretas dos dois ecossistemas.

> **Variante guiada**: rode exatamente a mesma entrada nos dois e confirme que os outputs batem (ou são consistentes) antes de comparar latência — sem essa validação, uma diferença de latência pode estar escondendo uma diferença de comportamento entre as duas implementações.

### Projeto 2.4 — Benchmark de hardware
- Rode o mesmo treinamento (rede pequena no MNIST) em CPU e GPU.
- Meça tempo, uso de memória.
- Se tiver Apple Silicon, faça também em MPS.

> **Variante guiada**: antes de medir, formule uma hipótese de quantas vezes mais rápido a GPU deve ser (baseado na intuição de paralelismo desta seção) — depois compare com o número real e reflita sobre a diferença (overhead de transferência de dados CPU↔GPU costuma explicar parte do gap em modelos pequenos).

---

## Erros comuns

- **Misturar `pip` e `conda` no mesmo ambiente.** Caos garantido.
- **Ignorar versões de CUDA.** PyTorch + CUDA + driver têm matriz de compatibilidade restrita.
- **Não fixar seeds.** Resultados não reproduzíveis = não-ciência.
- **Subestimar Docker.** Sem container, "funciona na minha máquina" é a regra.
- **Tentar treinar tudo localmente.** Saiba quando subir para cloud.

---

## Saiba mais

Alguns tópicos deste módulo foram citados de propósito sem profundidade — são áreas inteiras por si só, e o objetivo aqui é saber que existem e onde procurar quando precisar:

- **Kubernetes** (2.3) — orquestração de containers em escala; você não precisa disso para treinar ou servir um modelo sozinho, mas é o que empresas usam para rodar dezenas de serviços (incluindo modelos) de forma resiliente. `Curso` **Kubernetes Basics** (oficial, gratuito). https://kubernetes.io/docs/tutorials/kubernetes-basics/
- **Programação CUDA de baixo nível** (2.4) — escrever kernels customizados em C++/CUDA, além de só chamar operações do PyTorch que já usam CUDA por baixo; relevante se você for otimizar uma operação que nenhuma biblioteca ainda implementou eficientemente. `Livro` **Programming Massively Parallel Processors** — Kirk & Hwu.
- **Intel oneAPI e Vulkan compute** (2.4) — alternativas de nicho a CUDA/ROCm/MPS, relevantes só em hardware específico (Intel Arc, mobile/embarcado). Sem curadoria de referência única aqui — pesquise a documentação oficial do fabricante quando o hardware exigir.
- **JAX a fundo** (2.1) — a alternativa funcional ao PyTorch, com `jit`, `vmap` e `grad` como primitivas compostas em vez de um framework orientado a objetos; usado pesado em pesquisa (DeepMind) e quando TPU é o alvo. `Curso` **JAX Documentation — Quickstart** (oficial). https://jax.readthedocs.io/

---

## Checklist de saída

- [ ] Tenho um setup Python e TS reprodutíveis em qualquer máquina (se não, revise a seção 2.1 e o Projeto 2.1).
- [ ] Sei diferenciar quando usar Python vs TS para uma tarefa de IA (se não, revise a seção 2.2).
- [ ] Entendo o que é VRAM e sei calcular quanta um modelo precisa (se não, revise a seção 2.4).
- [ ] Consigo rodar um modelo Hugging Face em CPU, GPU local, e Colab (se não, revise o Projeto 2.4).
- [ ] Tenho um workflow de versionamento de código + dados + modelos (se não, revise a seção 2.3).
