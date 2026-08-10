# Trilha de Estudo — LLM, IA e Engenharia de Sistemas Inteligentes

> Currículo acadêmico + engenharia real, sem viés de fornecedor, com referências canônicas.
> Linguagens: **Python** (núcleo) e **TypeScript** (onde há ecossistema maduro).
> Saída: arquivos `.md` por módulo, este arquivo é o índice e o mapa.

---

## Filosofia da trilha

Esta trilha tem formato **T/Y**:

- **Base larga** (módulos 1–7): matemática, programação, ML clássico/moderno, DL, NLP clássico. Sem essa base, tudo o que vier depois é uso de caixa-preta.
- **Tronco profundo em LLMs** (módulos 8–15): Transformers, arquiteturas, treinamento, alinhamento, eficiência, prompting, RAG, agentes, avaliação, produção.
- **Galhos de especialização** (módulos 16–19): Visão Computacional, RL, Multimodal, Tópicos Avançados.
- **Síntese** (módulo 20): projetos integradores.

**Regra de ouro**: cada conceito teórico deve "virar código" pelo menos uma vez. Implementar do zero (mesmo que mal) é o que separa quem entende de quem reconhece.

---

## Mapa dos módulos

```
                    [00] Índice (este arquivo)
                              │
        ┌─────────────────────┼─────────────────────┐
        │  BASE (T/Y horizontal)                    │
        ├─[01] Matemática                          │
        ├─[02] Programação e Ferramentas           │
        ├─[03] ML Clássico                         │
        ├─[04] ML Moderno                          │
        ├─[05] Deep Learning                       │
        ├─[06] NLP Clássico                        │
        ├─[07] Transformers                        │
        │                                          │
        │  TRONCO LLM (vertical, profundidade)     │
        ├─[08] Arquiteturas de LLMs                │
        ├─[09] Treinamento e Alinhamento          │
        ├─[10] Eficiência e Inferência Local      │
        ├─[11] Prompt Engineering                  │
        ├─[12] RAG                                 │
        ├─[13] Agentes, Tools e Protocolos (MCP)  │
        ├─[14] Avaliação e Segurança              │
        ├─[15] Engenharia de Produção             │
        │                                          │
        │  GALHOS (especializações)                │
        ├─[16] Visão Computacional                 │
        ├─[17] Aprendizado por Reforço            │
        ├─[18] Multimodal                          │
        ├─[19] Tópicos Avançados                   │
        │                                          │
        └─[20] Projetos Integradores
```

---

## Como usar

1. **Leia o `Objetivos` e `Pré-requisitos`** de cada módulo antes de começar. Se o pré-requisito não está consolidado, volte.
2. **Não pule a matemática.** Os módulos 1–2 não são opcionais.
3. **Cada módulo tem três partes**:
   - **Obrigatório**: o mínimo para dominar o tópico.
   - **Opcional/Aprofundamento**: para quem quer ir além.
   - **Projeto prático**: curto, focado, valida a teoria.
4. **Referências são curadas**: papers originais, livros canônicos, cursos universitários públicos. Sem blogposts genéricos, sem fórum.
5. **Sem prazo.** Avance quando o módulo atual estiver consolidado, não pelo calendário.

---

## Convenções

- **🐍 Python**: ferramenta principal para pesquisa, treinamento, fine-tuning.
- **📘 TypeScript**: ferramenta principal para aplicações, agentes em produção, integrações web.
- **📄 Paper**: artigo acadêmico (com link arXiv ou venue).
- **📚 Livro**: livro canônico, geralmente com versão pública gratuita.
- **🎓 Curso**: curso universitário público (Stanford, MIT, CMU, Berkeley, etc.).
- **🛠 Ferramenta**: framework ou biblioteca relevante.
- **🧪 Projeto**: projeto prático curto.
- **⚠️ Crítico**: ponto onde a maioria erra.

---

## Quando Python é incontornável (e o que buscar em TS)

| Tarefa | Python | Equivalente/Caminho em TS |
|---|---|---|
| Treinamento from scratch | PyTorch, JAX | Não há equivalente sério → use Python |
| Fine-tuning (LoRA, QLoRA) | `peft`, `trl` | Não há equivalente → use Python |
| Pesquisa/papers | Padrão da área | Idem |
| Inferência client-side | — | `transformers.js`, ONNX Runtime Web |
| Aplicações com LLM | LangChain, LlamaIndex | `langchain.js`, `llamaindex` (TS), Vercel AI SDK |
| RAG em produção | LlamaIndex, LangChain | LlamaIndex.TS, LangChain.js |
| Agentes | LangGraph, CrewAI | LangGraph.js, Mastra |
| Vector DBs | Todos têm SDK Python | Quase todos têm SDK TS (Pinecone, Weaviate, Qdrant, Chroma) |
| Inferência local (servir) | vLLM, Ollama | Ollama tem cliente TS |
| MCP (Model Context Protocol) | SDK oficial Python | SDK oficial TS |

**Regra prática**: para **treinar/pesquisar**, Python. Para **integrar/servir/agentes em produção web**, TS é viável e às vezes superior (DX, deploy serverless, edge).

---

## Referências transversais (válidas para a trilha inteira)

### Livros canônicos (versões públicas gratuitas marcadas com ✅)

- ✅ **"Mathematics for Machine Learning"** — Deisenroth, Faisal, Ong. https://mml-book.github.io/
- ✅ **"Deep Learning"** — Goodfellow, Bengio, Courville. https://www.deeplearningbook.org/
- ✅ **"Dive into Deep Learning"** — Zhang, Lipton, Li, Smola. https://d2l.ai/ (com código em PyTorch, MXNet, JAX, TF)
- ✅ **"The Elements of Statistical Learning"** — Hastie, Tibshirani, Friedman. https://hastie.su.domains/ElemStatLearn/
- ✅ **"Pattern Recognition and Machine Learning"** — Bishop. PDF oficial liberado pela Microsoft Research.
- ✅ **"Speech and Language Processing"** (3ª ed., draft) — Jurafsky & Martin. https://web.stanford.edu/~jurafsky/slp3/
- ✅ **"Reinforcement Learning: An Introduction"** — Sutton & Barto. http://incompleteideas.net/book/the-book-2nd.html

### Cursos universitários públicos (todos com vídeos no YouTube/site oficial)

- 🎓 **MIT 18.06 — Linear Algebra** (Gilbert Strang). https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/
- 🎓 **Stanford CS229 — Machine Learning** (Andrew Ng). https://cs229.stanford.edu/
- 🎓 **Stanford CS230 — Deep Learning**. https://cs230.stanford.edu/
- 🎓 **Stanford CS224N — NLP with Deep Learning** (Manning). https://web.stanford.edu/class/cs224n/
- 🎓 **Stanford CS231N — CNNs for Visual Recognition**. http://cs231n.stanford.edu/
- 🎓 **Stanford CS25 — Transformers United**. https://web.stanford.edu/class/cs25/
- 🎓 **Stanford CS336 — Language Modeling from Scratch** (2024+). https://stanford-cs336.github.io/
- 🎓 **MIT 6.S191 — Introduction to Deep Learning**. http://introtodeeplearning.com/
- 🎓 **DeepMind x UCL — Reinforcement Learning Course** (David Silver). https://www.davidsilver.uk/teaching/
- 🎓 **CMU 11-747 — Neural Networks for NLP**. http://phontron.com/class/nn4nlp/
- 🎓 **3Blue1Brown — Essence of Linear Algebra / Calculus / Neural Networks**. https://www.3blue1brown.com/

### Hubs de papers e benchmarks

- **arXiv** (cs.CL, cs.LG, cs.AI, stat.ML). https://arxiv.org/
- **Papers with Code**. https://paperswithcode.com/
- **Hugging Face Papers** (curadoria diária). https://huggingface.co/papers
- **Distill.pub** (visualizações didáticas, arquivado mas excelente). https://distill.pub/

---

## Próximo passo

Comece pelo `01_matematica.md`. Não pule, mesmo que ache que sabe.
