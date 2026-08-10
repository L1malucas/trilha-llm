# Módulo 12 — RAG (Retrieval-Augmented Generation)

> **Objetivo**: dominar a arquitetura RAG do paper original às variantes modernas (hybrid, graph, agentic). Embeddings, chunking, vector DBs, retrievers, rerankers, avaliação.
>
> **Pré-requisitos**: Módulos 06, 07, 08, 11.
>
> **Tempo de referência**: 3–5 semanas.

---

## Por que isso importa

LLMs têm conhecimento limitado, desatualizado e podem alucinar. RAG é a abordagem padrão para:
- **Atualidade** (dados que mudam).
- **Privacidade** (dados que não estão na web).
- **Atribuição** (citar fontes).
- **Custo** (modelo pequeno + bom retrieval pode bater modelo gigante puro).

Mas RAG mal-feito é pior que LLM puro: documentos errados → respostas erradas com confiança.

---

## 12.1 Paper fundador e conceito

📄 **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** — Lewis et al. (2020). https://arxiv.org/abs/2005.11401

### Pipeline canônico
```
Query → Embed → Vector Search → Top-K docs
                                    ↓
                    Prompt(query + docs) → LLM → Resposta
```

### Variações modernas
- **Pre-retrieval**: query rewriting, HyDE, sub-query decomposition.
- **Retrieval**: hybrid (dense + sparse), graph, multi-hop.
- **Post-retrieval**: rerank, compression, fusion.
- **Generation**: citação, fallback, refinamento iterativo.

---

## 12.2 Embeddings

### Conceito (revisão do mod. 06+08)
Embeddings = vetores densos que codificam significado. Documentos próximos em significado têm embeddings próximos por similaridade (cosseno, dot product).

### Modelos de embedding (atual: 2024–2025)
- **OpenAI text-embedding-3-large/small** (proprietário).
- **Cohere Embed v3**.
- **Voyage AI**.
- ✅ **E5** (Microsoft, multilingual). https://huggingface.co/intfloat/multilingual-e5-large
- ✅ **BGE-M3** (BAAI, multifuncional). https://huggingface.co/BAAI/bge-m3
- ✅ **Nomic Embed**. https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
- ✅ **Jina Embeddings v3**. https://huggingface.co/jinaai/jina-embeddings-v3
- ✅ **Stella, GTE, Snowflake Arctic**, todos competitivos.

### Como escolher
Consulte **MTEB** (Massive Text Embedding Benchmark): https://huggingface.co/spaces/mteb/leaderboard
- Filtre por idioma (PT-BR é importante).
- Considere tamanho do embedding (384 vs 768 vs 1024 vs 4096).
- Considere licença e custo.

### Embeddings multilíngues
- **multilingual-e5**, **BGE-M3**, **Cohere multilingual** funcionam bem em PT-BR.
- Modelos só-em-inglês degradam em PT, mas funcionam parcialmente.

### Matryoshka Representation Learning
Embeddings que permitem truncar dimensão sem retreino — economia de armazenamento.
- 📄 **Matryoshka Representation Learning** — Kusupati et al. (2022). https://arxiv.org/abs/2205.13147

---

## 12.3 Chunking: a arte de dividir documentos

### Estratégias
- **Fixed size**: N caracteres/tokens, com overlap. Simples, frequentemente sub-ótimo.
- **Recursive**: dividir por separadores em prioridade decrescente (parágrafo → frase → palavra).
- **Semantic chunking**: agrupar por similaridade entre frases adjacentes.
- **Document-aware**: respeitar headings (Markdown), seções (LaTeX), tabelas.
- **Sentence-based**: por frase, com janela.
- **Token-based**: respeitar limite do modelo de embedding e do LLM.

### Trade-offs
- **Chunks pequenos**: mais precisão, menos contexto.
- **Chunks grandes**: mais contexto, menos precisão (e pode estourar contexto do LLM).
- **Overlap** ajuda na fronteira mas duplica armazenamento.

### Truques modernos
- **Late Chunking** (Jina) — chunkar **depois** de embeddar o documento todo. https://jina.ai/news/late-chunking-in-long-context-embedding-models/
- **Contextual Retrieval** (Anthropic) — anexar contexto extraído de cada chunk antes de embeddá-lo. https://www.anthropic.com/news/contextual-retrieval
- **Parent-Child Chunking** — embedar trechos pequenos, retornar trecho-pai grande na recuperação.

### Ferramentas
- 🛠 **LangChain Text Splitters** (Python e TS).
- 🛠 **LlamaIndex NodeParsers**.
- 🛠 **Unstructured.io** — parsers ricos para PDF, HTML, DOCX. https://unstructured.io/

---

## 12.4 Vector Databases

### Categorias
- **Bibliotecas embarcadas**: FAISS (Meta), USearch, ScaNN, Annoy, hnswlib.
- **Bancos open-source self-hosted**: Qdrant, Weaviate, Milvus, Chroma, Vespa, pgvector (Postgres extension).
- **SaaS**: Pinecone, Vespa Cloud, Qdrant Cloud, Weaviate Cloud.
- **Multi-modal/full-text**: Vespa, Elasticsearch (com kNN search), OpenSearch.

### Algoritmos de busca aproximada (ANN)
- **HNSW** (Hierarchical Navigable Small World) — padrão moderno. 📄 https://arxiv.org/abs/1603.09320
- **IVF + PQ** (Inverted File + Product Quantization) — usado em FAISS para escalar bilhões.
- **DiskANN** — disco em vez de RAM, para datasets gigantes. 📄 (Microsoft Research)

### Critérios de escolha
| Necessidade | Sugestão |
|---|---|
| Local, simples, embeded | Chroma, FAISS |
| Postgres-based, transacional | pgvector |
| Self-hosted produção | Qdrant, Weaviate, Milvus |
| Hybrid (BM25 + vetor) nativo | Vespa, Weaviate, Elasticsearch |
| Escala massiva managed | Pinecone, Vespa Cloud |
| Edge/serverless | Turbopuffer, LanceDB |

### TS / JS
- **Chroma** (cliente JS), **Qdrant** (cliente JS), **Pinecone** (SDK TS), **Weaviate** (TS).
- **LanceDB** (TS-friendly, embedded). https://lancedb.com/

### Referências
- 📚 **FAISS Documentation**. https://faiss.ai/
- 📚 **Qdrant Documentation**. https://qdrant.tech/documentation/
- 🛠 **pgvector**. https://github.com/pgvector/pgvector

---

## 12.5 Retrieval avançado

### Sparse retrieval (não é "antigo", é complementar)
- **BM25** — variante clássica de TF-IDF, ainda excelente baseline.
- **SPLADE** — sparse aprendido. 📄 https://arxiv.org/abs/2107.05720
- **uniCOIL**, **TILDE**.

### Hybrid retrieval
Combinar dense + sparse via:
- **Reciprocal Rank Fusion (RRF)** — método simples e eficaz.
- **Pesos linearmente combinados**.
- **Learned fusion**.

### Multi-vector / Late interaction
- **ColBERT** — embedda cada token, combina por max-sim. 📄 https://arxiv.org/abs/2004.12832
- **ColBERTv2** — eficiente. https://arxiv.org/abs/2112.01488

### Query transformation
- **HyDE (Hypothetical Document Embeddings)** — gerar resposta hipotética, embeddá-la, buscar. 📄 https://arxiv.org/abs/2212.10496
- **Query rewriting** com LLM.
- **Multi-query retrieval** — gerar variações da query e fundir.
- **Step-back prompting** — abstrair antes de buscar.

### Multi-hop retrieval
Para perguntas que exigem combinar múltiplos documentos.
- 📄 **MuSiQue**, **HotpotQA** (datasets).
- **IRCoT** — interleaving retrieval com CoT.

### Referências
- 📄 **Dense Passage Retrieval (DPR)** — Karpukhin et al. (2020). https://arxiv.org/abs/2004.04906
- 📄 **REALM** — Guu et al. (2020). https://arxiv.org/abs/2002.08909

---

## 12.6 Reranking

### Por quê
Retrieval rápido (top-100) com baixa precisão → rerank lento (top-10) com alta precisão.

### Modelos
- **Cross-encoders** (Sentence-BERT cross variants).
- **bge-reranker-v2-m3** (open). https://huggingface.co/BAAI/bge-reranker-v2-m3
- **Cohere Rerank** (API).
- **Jina Reranker**.
- **MonoT5**, **MonoBERT**.

### Custo
Reranker tipicamente é 10–100× mais caro que retrieval inicial. Use só no top-K (K = 50–200).

---

## 12.7 Geração com contexto

### Anti-padrões comuns
- Concatenar todos os docs sem hierarquia.
- Não citar fontes.
- Não tratar caso "nenhum doc relevante".
- Não validar a resposta vs docs.

### Padrões bons
- **Estrutura clara** no prompt (com XML/marcadores).
- **Instrução para citar IDs** dos docs.
- **Self-grounding**: pedir para o modelo apontar a parte do doc que sustenta cada afirmação.
- **Refusal** quando docs não suportam: "Diga 'não sei com base nos documentos' se não houver evidência."

### Long context vs RAG
"Contextos de 1M tokens" não eliminam RAG. Trade-offs:
- **Long context**: mais simples, mais caro por chamada, performance degrada no meio do contexto ("lost in the middle"). 📄 https://arxiv.org/abs/2307.03172
- **RAG**: mais infraestrutura, mais barato, melhor para corpora dinâmicos.

---

## 12.8 Graph RAG

Para perguntas que exigem **conexões** entre entidades.

### Conceito
- Extrair entidades + relações dos docs (com LLM).
- Construir grafo de conhecimento.
- Recuperar subgrafos relevantes em vez de chunks isolados.

### Implementações
- **Microsoft GraphRAG**. https://github.com/microsoft/graphrag (📄 https://arxiv.org/abs/2404.16130)
- **LightRAG**, **HippoRAG**.
- **LlamaIndex Property Graph Index**.

---

## 12.9 Agentic RAG

Em vez de pipeline fixo, agente decide:
- Se buscar.
- Que query fazer.
- Que ferramenta usar.
- Quando rebuscar com refinamento.

Cross-link com módulo 13 (Agentes).

### Referências
- 📄 **Self-RAG: Learning to Retrieve, Generate, and Critique** — Asai et al. (2023). https://arxiv.org/abs/2310.11511
- 📄 **CRAG: Corrective Retrieval Augmented Generation** — Yan et al. (2024). https://arxiv.org/abs/2401.15884
- 📄 **Agentic RAG: A Survey** (2025). https://arxiv.org/abs/2501.09136

---

## 12.10 Avaliação de RAG

### Métricas separadas para retrieval e generation
- **Retrieval**: Recall@K, MRR, NDCG, hit rate.
- **Generation**: faithfulness (resposta apoia-se nos docs?), answer relevance, context relevance.

### Frameworks
- 🛠 **RAGAS** — toolkit de eval, com LLM-as-judge. https://github.com/explodinggradients/ragas
- 🛠 **TruLens**. https://github.com/truera/trulens
- 🛠 **DeepEval**. https://github.com/confident-ai/deepeval
- 🛠 **LangSmith**, **Langfuse** (mod. 15).

### Datasets canônicos
- **Natural Questions**, **TriviaQA**, **HotpotQA**, **MS MARCO**.
- **BEIR** — benchmark heterogêneo de retrieval. https://github.com/beir-cellar/beir

### Referências
- 📄 **RAGAS: Automated Evaluation of Retrieval Augmented Generation** — Es et al. (2023). https://arxiv.org/abs/2309.15217

---

## 12.11 Frameworks de RAG

### Python
- **LlamaIndex** — focado em indexação e retrieval. https://docs.llamaindex.ai/
- **LangChain** — generalista, com componentes RAG. https://python.langchain.com/
- **Haystack** (deepset). https://haystack.deepset.ai/
- **txtai**.

### TypeScript
- **LlamaIndex.TS**. https://ts.llamaindex.ai/
- **LangChain.js**. https://js.langchain.com/
- **Vercel AI SDK** (tem hooks para RAG).
- **Mastra** (framework agentico que cobre RAG). https://mastra.ai/

### Quando *não* usar framework
- Pipelines simples → escreva direto. Frameworks têm overhead conceitual.
- Aprendizado → implemente do zero pelo menos uma vez.

---

## 🧪 Projetos práticos

### Projeto 12.1 — RAG mínimo do zero
- Sem framework. Apenas: embeddings + FAISS + LLM local.
- Corpus: ~100 documentos próprios (PDFs, MDs, etc.).
- Implemente: ingest → embed → search → prompt → response.
- Em Python E em TS (com transformers.js para embeddings, modelo via Ollama).

### Projeto 12.2 — Comparativo de chunking
- Mesmo corpus, mesmo modelo, mesmo retriever.
- Variar: fixed-size (3 tamanhos), recursive, semantic, late chunking.
- Avalie em conjunto de 30 perguntas com gabarito.

### Projeto 12.3 — Hybrid retrieval com RRF
- Combine BM25 (via `rank_bm25` ou Elasticsearch) com retrieval denso.
- Use RRF para fundir.
- Compare com cada método isolado.

### Projeto 12.4 — Reranker
- Adicione `bge-reranker-v2-m3` ao pipeline do projeto 12.3.
- Meça ganho em Recall@5 e qualidade final da geração.

### Projeto 12.5 — Avaliação rigorosa com RAGAS
- Construa eval set: 50 perguntas com (a) docs relevantes anotados, (b) respostas de ouro.
- Use RAGAS para faithfulness, answer relevance, context relevance.
- Documente onde RAG quebra.

### Projeto 12.6 — Graph RAG mini
- Use Microsoft GraphRAG ou implemente versão simplificada.
- Compare com RAG de chunks em perguntas de "conexões" (ex: "que pessoas mencionadas no corpus trabalharam juntas?").

### Projeto 12.7 — Self-RAG / CRAG
- Implemente lógica: o LLM avalia se o doc é relevante; se não, busca novamente com query refinada.
- Compare com RAG fixo.

---

## ⚠️ Erros comuns

- **Avaliar só "vibe"** — sem dataset, sem métricas, qualquer mudança parece melhoria.
- **Embedding model + LLM model em idiomas diferentes** — degradação em PT-BR clássica.
- **Chunks muito grandes** — diluição de relevância no retrieval.
- **Não normalizar embeddings** — alguns modelos exigem; cosine similarity exige.
- **Confundir "tem embedding" com "encontra documento"** — embeddings ruins → retrieval ruim → resposta ruim.
- **Não tratar "nada relevante encontrado"** — modelo alucina com confiança.
- **Misturar dimensões/modelos** ao migrar — reindexação total é necessária.

---

## Conexão com módulos seguintes

| Conceito daqui | Aparece em |
|---|---|
| Embeddings + retrieval | Agentes (mod. 13) — RAG como tool |
| Reranker | Avaliação (mod. 14) |
| Citações e grounding | Segurança (mod. 14) |
| Vector DB ops | Produção (mod. 15) |

---

## Checklist de saída

- [ ] Construí RAG do zero em Python e em TS.
- [ ] Avaliei pelo menos 3 estratégias de chunking objetivamente.
- [ ] Sei configurar hybrid retrieval (BM25 + vetorial).
- [ ] Uso reranker no pipeline e meço o ganho.
- [ ] Tenho conjunto de eval com RAGAS rodando.
- [ ] Sei distinguir quando usar RAG, long context, ou fine-tuning.
