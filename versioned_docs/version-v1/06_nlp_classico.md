---
id: 06_nlp_classico
title: "Módulo 06 — NLP Clássico (pré-Transformer)"
sidebar_position: 6
---

# Módulo 06 — NLP Clássico (pré-Transformer)

> **Objetivo**: entender como linguagem foi tratada *antes* dos Transformers — e por que isso ainda importa. Sem essa base, "embeddings", "tokenização" e "contexto" são só palavras.
>
> **Pré-requisitos**: Módulos [01](01_matematica.md)–[05](05_deep_learning.md).
>
> **Tempo de referência**: 3–5 semanas.

---

## Por que isso importa

A maioria pula NLP clássico e vai direto para LLMs. Resultado: trata embeddings como mágica, não entende limitações, não sabe debugar pipelines de texto. Os conceitos aqui (tokenização, n-gramas, embeddings distribucionais) **continuam vivos** dentro dos Transformers — só que escondidos.

---

## 6.1 Linguagem como dado

### O que é texto, computacionalmente
- Caracteres (Unicode, UTF-8, code points, grafemas, normalização NFC/NFD).
- Tokens (e a ambiguidade do que conta como token).
- Documentos, corpora.

### Pré-processamento clássico
- **Normalização**: lowercase, remoção de acentos, normalização Unicode.
- **Tokenização** clássica: por whitespace, por regex, por regras linguísticas.
- **Stop words**: quando remover, quando manter.
- **Stemming** (Porter, Snowball) e **Lemmatization** (com POS tagging).
- **Sentence segmentation**.
- **Part-of-Speech tagging**.
- **Named Entity Recognition (NER)**.
- **Parsing**: constituency, dependency.

### Por que isso ainda importa
Pipelines profissionais ainda fazem **muito** desse pré-processamento — para limpeza, para extração de metadados, para RAG. Não é "obsoleto", é "aposentado da posição central".

### Ferramentas
- **NLTK** (clássica, didática). https://www.nltk.org/
- **spaCy** (industrial, rápida, multilíngue). https://spacy.io/
- **Stanza** (Stanford NLP, vários idiomas). https://stanfordnlp.github.io/stanza/
- TS: spaCy não tem porte sério; para NLP clássico em TS use `compromise` ou serviço externo. Para tokenização/NER moderna, transformers.js cobre.

### Referências
- 📚 **Speech and Language Processing** — Jurafsky & Martin, 3ª ed., draft gratuito. https://web.stanford.edu/~jurafsky/slp3/ (caps. 1–8 cobrem a fundação)
- 🎓 **Stanford CS224N** — primeiras aulas. https://web.stanford.edu/class/cs224n/

---

## 6.2 Modelos estatísticos de linguagem (n-gramas)

### Conceito
Modelar P(w_n | w_1, ..., w_{n-1}) com a aproximação de Markov: P(w_n | w_{n-k+1}, ..., w_{n-1}).

### Tópicos
- **N-gramas**: unigram, bigram, trigram.
- **Maximum Likelihood Estimation** sobre contagens.
- **Smoothing**: Laplace, Good-Turing, **Kneser-Ney** (estado da arte pré-neural).
- **Perplexity** como métrica de modelo de linguagem.

### Por que isso importa
- Perplexity continua sendo a métrica usada para avaliar LLMs.
- Conceitos como "predict next token" começam aqui.
- A intuição de Markov ajuda a entender contexto limitado.

### Referências
- 📚 **Speech and Language Processing**, cap. 3 (N-gram Language Models). https://web.stanford.edu/~jurafsky/slp3/
- 📄 **An Empirical Study of Smoothing Techniques for Language Modeling** — Chen & Goodman (1998). https://aclanthology.org/J99-4007/

---

## 6.3 Representação de texto: do esparso ao denso

### Representações esparsas
- **Bag of Words (BoW)**.
- **TF-IDF**.
- **One-hot encoding** de vocabulário.

### Limitações
- Não captura semântica ("rei" e "monarca" são tão distantes quanto "rei" e "abacaxi").
- Vocabulário grande → vetores enormes e esparsos.
- Sem ordem (em BoW puro).

### Modelos de tópicos (cross-link mod. [04](04_ml_moderno.md))
- **Latent Semantic Analysis (LSA)** via SVD em matriz termo-documento.
- **Latent Dirichlet Allocation (LDA)**.
- **Non-negative Matrix Factorization (NMF)**.

### Referências
- 📄 **Latent Dirichlet Allocation** — Blei, Ng, Jordan (2003). https://www.jmlr.org/papers/v3/blei03a.html

---

## 6.4 Word Embeddings — a revolução pré-Transformer

### A hipótese distribucional
"You shall know a word by the company it keeps" (Firth, 1957). **Palavras com contextos similares têm significados similares.** Isso é a base de todo embedding.

### Word2Vec
- **CBOW** (Continuous Bag of Words): prediz palavra-alvo a partir do contexto.
- **Skip-gram**: prediz contexto a partir da palavra-alvo.
- **Negative Sampling** (truque que torna o treinamento viável).
- **Vector arithmetic**: rei − homem + mulher ≈ rainha.

### GloVe
- Combina co-ocorrência global com janelas locais.
- Treinado em matriz de co-ocorrências.

### FastText
- Embeddings em nível de subpalavra (n-gramas de caracteres).
- Lida com palavras OOV (out-of-vocabulary).

### Limitações
- **Estáticos**: "banco" tem o mesmo embedding em "banco do parque" e "banco financeiro".
- Resolvido depois por embeddings contextuais (ELMo, BERT — mod. [08](08_llms_arquiteturas.md)).

### Papers fundadores (todos obrigatórios para ler)
- 📄 **Efficient Estimation of Word Representations in Vector Space (Word2Vec)** — Mikolov et al. (2013). https://arxiv.org/abs/1301.3781
- 📄 **Distributed Representations of Words and Phrases (Word2Vec, negative sampling)** — Mikolov et al. (2013). https://arxiv.org/abs/1310.4546
- 📄 **GloVe: Global Vectors for Word Representation** — Pennington, Socher, Manning (2014). https://nlp.stanford.edu/pubs/glove.pdf
- 📄 **Enriching Word Vectors with Subword Information (FastText)** — Bojanowski et al. (2016). https://arxiv.org/abs/1607.04606

### Ferramentas
- **gensim** (Python). https://radimrehurek.com/gensim/
- **fastText** (CLI + Python). https://fasttext.cc/

---

## 6.5 Tokenização para a era neural

### Por que tokenização avançada importa
- Vocabulário fixo grande consome memória.
- Palavras raras viram OOV.
- Solução: tokenizar em **subpalavras**.

### Algoritmos
- **Byte Pair Encoding (BPE)**: mescla pares mais frequentes iterativamente. Usado em GPT.
- **WordPiece**: variante do BPE usada em BERT.
- **Unigram Language Model**: treinamento probabilístico, usado em SentencePiece.
- **SentencePiece**: implementação que trata texto como bytes/caracteres puros (sem pré-tokenização). Usado em LLaMA, T5.
- **Tiktoken** (OpenAI BPE em Rust). https://github.com/openai/tiktoken

### Referências
- 📄 **Neural Machine Translation of Rare Words with Subword Units (BPE)** — Sennrich, Haddow, Birch (2015). https://arxiv.org/abs/1508.07909
- 📄 **SentencePiece** — Kudo & Richardson (2018). https://arxiv.org/abs/1808.06226
- 🎓 **Karpathy — Let's build the GPT Tokenizer** (vídeo + repo `minbpe`). https://www.youtube.com/watch?v=zduSFxRajkE
- 🎓 **Hugging Face Course — Tokenizers**. https://huggingface.co/learn/llm-course

---

## 6.6 Tarefas clássicas de NLP

### Classificação de texto
- Análise de sentimento.
- Detecção de spam.
- Categorização de tópicos.

### Sequência → sequência (Seq2Seq)
- Tradução automática.
- Sumarização.
- Question answering.

### Extração de informação
- NER, relation extraction, event extraction.

### Parsing
- Constituency parsing, dependency parsing.

### Conversação clássica
- Pipelines com NLU (Natural Language Understanding) + DM (Dialogue Manager) + NLG (Natural Language Generation) — substituídos por LLMs ponta-a-ponta hoje, mas o conceito permanece.

### Referências
- 📚 **Speech and Language Processing** — caps. dedicados a cada tarefa.
- 🎓 **Stanford CS224N**. https://web.stanford.edu/class/cs224n/

---

## 6.7 Avaliação em NLP

### Métricas
- **Accuracy / F1** (classificação).
- **BLEU** (tradução). Limitações conhecidas.
- **ROUGE** (sumarização).
- **METEOR**, **chrF** (alternativas a BLEU).
- **Perplexity** (modelagem de linguagem).
- **BERTScore** (avaliação semântica via embeddings).

### Conjuntos de avaliação clássicos
- **GLUE** e **SuperGLUE** (benchmarks de tarefas variadas).
- **SQuAD** (question answering).
- **CoNLL** (NER, parsing).

### Referências
- 📄 **GLUE** — Wang et al. (2018). https://arxiv.org/abs/1804.07461
- 📄 **BERTScore** — Zhang et al. (2019). https://arxiv.org/abs/1904.09675

---

## 🧪 Projetos práticos

### Projeto 6.1 — Pipeline clássico ponta-a-ponta
- Tarefa: classificação de sentimento em IMDb.
- Pipeline: tokenização → stopwords → stemming → TF-IDF → Logistic Regression.
- Faça em scikit-learn.
- **Meta**: >85% accuracy só com clássico.

### Projeto 6.2 — Treinar Word2Vec do zero
- Use gensim em corpus português (Wikipedia em PT, ou Folha-PT).
- Avalie qualitativamente: analogias (rei − homem + mulher), similaridades.
- **Compare** com embeddings pré-treinados em PT (NILC, fastText).

### Projeto 6.3 — Implementar BPE do zero
- Em Python puro, sem `tokenizers`.
- Treine em um corpus pequeno.
- Compare com `tiktoken` ou `tokenizers` da HF.
- **Referência guia**: o vídeo do Karpathy sobre tokenização.

### Projeto 6.4 — NER com spaCy + customização
- Use modelo PT-BR do spaCy.
- Adicione entidades customizadas (ex: nomes de produtos).
- Avalie com F1 sobre conjunto anotado por você.

### Projeto 6.5 — Comparativo: TF-IDF vs Word2Vec vs BERT-embeddings
- Mesma tarefa de classificação.
- Veja o ganho (ou não) de cada representação.
- **O que isso prova**: que embeddings melhoram, mas não são bala de prata.

---

## ⚠️ Erros comuns

- **Não normalizar Unicode** (NFC vs NFD) — bug clássico em PT-BR (acentos).
- **Stemming agressivo** que junta palavras semanticamente diferentes.
- **Ignorar tokenização** ao migrar entre modelos — cada modelo tem seu tokenizer.
- **Avaliação com leakage** (mesma frase em treino e teste).
- **Usar BLEU como verdade absoluta** — métrica falha, especialmente em PT-BR.

---

## Conexão com módulos seguintes

| Conceito daqui | Aparece em |
|---|---|
| Tokenização (BPE, SentencePiece) | LLMs (mod. [08](08_llms_arquiteturas.md), [09](09_treinamento_e_alinhamento.md)) |
| Word embeddings | Embeddings contextuais em LLMs |
| Hipótese distribucional | Toda IA moderna |
| Perplexity | Avaliação de LLMs (mod. [14](14_avaliacao_e_seguranca.md)) |
| Seq2Seq | Encoder-decoder em Transformers (mod. [07](07_transformers.md)) |
| Attention sobre RNN (Bahdanau) | Self-attention (mod. [07](07_transformers.md)) |
| TF-IDF | Sparse retrieval em RAG (mod. [12](12_rag.md)) |

---

## Checklist de saída

- [ ] Implementei BPE do zero.
- [ ] Treinei e avaliei Word2Vec.
- [ ] Sei explicar a hipótese distribucional.
- [ ] Conheço os trade-offs de tokenização: BPE vs WordPiece vs SentencePiece.
- [ ] Posso construir pipeline NLP clássico ponta-a-ponta sem framework "mágico".
- [ ] Entendo limitações de embeddings estáticos — e qual problema BERT/GPT resolvem.
