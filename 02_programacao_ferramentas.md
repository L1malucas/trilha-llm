# Módulo 02 — Programação e Ferramentas

> **Objetivo**: ter um ambiente de trabalho profissional em Python e TypeScript, dominar as bibliotecas numéricas, e entender o hardware sob o qual ML roda.
>
> **Pré-requisitos**: Módulo 01 (matemática), conhecimento prévio de programação.
>
> **Tempo de referência**: 2–4 semanas.

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

### Gerenciamento de ambiente
- `venv` (padrão), `conda` (científico), **`uv`** (moderno, rapidíssimo, recomendado).
- `pyproject.toml` em vez de `requirements.txt` quando possível.
- `pip-tools` ou `uv` para lockfiles.

### Referências
- 📚 **Fluent Python (2nd ed.)** — Luciano Ramalho.
- 🎓 **NumPy Documentation — User Guide** (oficial). https://numpy.org/doc/stable/user/
- 🎓 **PyTorch Tutorials** (oficial). https://pytorch.org/tutorials/
- 🛠 **uv** (gerenciador de pacotes/ambientes). https://docs.astral.sh/uv/

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
| **transformers.js** | Inferência de modelos no browser/Node via ONNX |
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

### Referências
- 📚 **Effective TypeScript** — Dan Vanderkam.
- 🛠 **Vercel AI SDK Documentation**. https://ai-sdk.dev/
- 🛠 **transformers.js**. https://huggingface.co/docs/transformers.js
- 🛠 **MCP TypeScript SDK**. https://github.com/modelcontextprotocol/typescript-sdk

---

## 2.3 Engenharia de software para ML

### Controle de versão
- Git avançado: rebase, cherry-pick, bisect, hooks.
- **DVC** (Data Version Control) — versionar datasets e modelos. https://dvc.org/
- Git LFS para modelos pequenos.
- **Hugging Face Hub** como repositório de modelos/datasets (usa Git+LFS).

### Containers e orquestração
- Docker: imagens, multi-stage builds, otimização de cache.
- Docker Compose para pilhas locais (LLM + vector DB + app).
- Conhecimento básico de Kubernetes (não precisa ser expert).

### Reprodutibilidade
- **Weights & Biases (W&B)**, **MLflow**, ou **Aim** — tracking de experimentos.
- Seeds determinísticos.
- Documentação de versões de CUDA, drivers, bibliotecas.

### Referências
- 📚 **Designing Machine Learning Systems** — Chip Huyen.
- 🎓 **Made With ML — MLOps Course**. https://madewithml.com/
- 🛠 **MLflow Documentation**. https://mlflow.org/

---

## 2.4 Hardware e aceleração

### CPU vs GPU vs TPU
- Por que GPU acelera ML: paralelismo massivo em operações matriciais.
- Quando CPU basta: inferência de modelos pequenos, dados tabulares.
- TPU (Google) — relevante se for usar JAX em larga escala.

### Plataformas
- **NVIDIA CUDA** (padrão de fato, melhor suporte).
- **AMD ROCm** (alternativa em Linux, suporte crescente).
- **Apple Silicon (MPS)** — `torch.backends.mps`, suporte parcial mas funcional.
- **Intel oneAPI**, **Vulkan compute** (nichos).

### VRAM como limite
- Regra de bolso: parâmetros × bytes por parâmetro.
- FP32 = 4 bytes, FP16/BF16 = 2 bytes, INT8 = 1 byte, INT4 = 0.5 byte.
- Modelo 7B em FP16 = ~14 GB; em INT4 = ~3.5 GB. Isso define o que cabe na sua placa.
- Treinamento exige ~3–4× a memória de inferência (gradientes + estados de otimizador).

### Cloud para quem não tem GPU
- Google Colab (free tier com T4).
- Kaggle Notebooks (free tier com P100/T4).
- Lambda Labs, RunPod, Vast.ai (alugar por hora, $0.30–$2/h em T4–A100).
- Hugging Face Spaces (deploy gratuito de demos).

### Referências
- 🎓 **NVIDIA Deep Learning Institute** (cursos gratuitos sobre CUDA, otimização). https://www.nvidia.com/en-us/training/online/
- 📄 **Mixed Precision Training** — Micikevicius et al. (2017). https://arxiv.org/abs/1710.03740
- 📚 **Programming Massively Parallel Processors** — Kirk & Hwu (CUDA, opcional, profundidade).

---

## 🧪 Projetos práticos

### Projeto 2.1 — Setup completo reprodutível
- Crie um repo com: `pyproject.toml`, lockfile (`uv` ou `pip-tools`), Dockerfile, `Makefile`.
- Configure pre-commit (`black`, `ruff`, `mypy`).
- Adicione um workflow CI básico (GitHub Actions) que rode testes.
- **Teste**: clone em outra máquina, rode tudo do zero.

### Projeto 2.2 — Análise exploratória profissional
- Escolha um dataset (Kaggle ou UCI ML Repo).
- Faça EDA com Pandas + Plotly em Jupyter.
- Documente hipóteses, anomalias, decisões de limpeza.
- Exporte como notebook + relatório em Markdown.

### Projeto 2.3 — Mesmo problema em PyTorch e em TS (transformers.js)
- Carregue um modelo pequeno (ex.: `distilbert-base-uncased` para classificação).
- Faça inferência em Python (PyTorch) e em TS (transformers.js).
- Compare latência, output, DX.
- **Objetivo**: sentir as diferenças concretas dos dois ecossistemas.

### Projeto 2.4 — Benchmark de hardware
- Rode o mesmo treinamento (rede pequena no MNIST) em CPU e GPU.
- Meça tempo, uso de memória.
- Se tiver Apple Silicon, faça também em MPS.

---

## ⚠️ Erros comuns

- **Misturar `pip` e `conda` no mesmo ambiente.** Caos garantido.
- **Ignorar versões de CUDA.** PyTorch + CUDA + driver têm matriz de compatibilidade restrita.
- **Não fixar seeds.** Resultados não reproduzíveis = não-ciência.
- **Subestimar Docker.** Sem container, "funciona na minha máquina" é a regra.
- **Tentar treinar tudo localmente.** Saiba quando subir para cloud.

---

## Checklist de saída

- [ ] Tenho um setup Python e TS reprodutíveis em qualquer máquina.
- [ ] Sei diferenciar quando usar Python vs TS para uma tarefa de IA.
- [ ] Entendo o que é VRAM e sei calcular quanta um modelo precisa.
- [ ] Consigo rodar um modelo Hugging Face em CPU, GPU local, e Colab.
- [ ] Tenho um workflow de versionamento de código + dados + modelos.
