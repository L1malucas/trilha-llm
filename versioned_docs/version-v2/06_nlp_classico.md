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

## Objetivos de aprendizagem

Ao final deste módulo você deve ser capaz de:

- Explicar a hipótese distribucional e por que ela é a base de todo embedding, inclusive os de LLMs.
- Explicar por que BPE/SentencePiece resolvem o problema de vocabulário aberto (OOV), com um exemplo.
- Calcular perplexity a partir de probabilidades de um modelo de linguagem simples.
- Explicar por que embeddings estáticos (Word2Vec) falham em palavras polissêmicas — e o que embeddings contextuais mudam.
- Escolher a métrica de avaliação certa para uma tarefa de NLP (classificação, tradução, sumarização) e explicar sua limitação.

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

> Um detalhe fácil de subestimar: NFC vs NFD são duas formas *diferentes* de representar em bytes o mesmo caractere acentuado ("é" pode ser um único code point ou "e" + acento combinante) — visualmente idênticos, mas strings diferentes, o que quebra comparação exata e busca se o pipeline não normaliza consistentemente. É um dos bugs mais silenciosos em pipelines PT-BR.

### Ferramentas
- **NLTK** (clássica, didática). https://www.nltk.org/
- **spaCy** (industrial, rápida, multilíngue). https://spacy.io/
- **Stanza** (Stanford NLP, vários idiomas). https://stanfordnlp.github.io/stanza/
- TS: spaCy não tem porte sério; para NLP clássico em TS use `compromise` ou serviço externo. Para tokenização/NER moderna, transformers.js cobre.

### Referências
- `Livro` **Speech and Language Processing** — Jurafsky & Martin, 3ª ed., draft gratuito. https://web.stanford.edu/~jurafsky/slp3/ (caps. 1–8 cobrem a fundação)
- `Curso` **Stanford CS224N** — primeiras aulas. https://web.stanford.edu/class/cs224n/

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

> **Intuição**: a aproximação de Markov diz "pra prever a próxima palavra, você não precisa da frase inteira até aqui, só das últimas k palavras" — uma simplificação necessária porque contar ocorrências exatas de frases inteiras é inviável (a maioria das frases longas nunca se repete no corpus). Smoothing existe porque contagem pura (MLE) atribui probabilidade zero a qualquer n-grama nunca visto no treino — mesmo que seja uma combinação perfeitamente razoável de palavras. Kneser-Ney redistribui probabilidade dos n-gramas vistos para os não-vistos de forma mais inteligente que só somar 1 a cada contagem (Laplace).
>
> **Exemplo resolvido — perplexity**: perplexity é `2^(entropia cruzada)`, e intuitivamente mede "quantas opções o modelo está, em média, hesitando entre" a cada palavra. Se um modelo atribui probabilidade média de 0.5 à palavra correta a cada passo, sua perplexity é `2^(-log2(0.5)) = 2^1 = 2` — como se estivesse escolhendo entre 2 opções igualmente prováveis a cada palavra. Um modelo pior, com probabilidade média 0.1 na palavra correta, tem perplexity `2^(-log2(0.1)) ≈ 2^3.32 ≈ 10` — como escolher entre 10 opções. Perplexity menor = modelo mais confiante nas palavras certas = melhor modelo de linguagem.
>
> **Aplicação real**: essa mesma métrica (perplexity) é reportada até hoje para comparar LLMs modernos no mesmo corpus de teste — a mecânica de "quão surpreso o modelo fica com o próximo token real" não mudou, só a arquitetura que produz as probabilidades.
>
> **Checkpoint**: sem olhar o texto, explique por que MLE puro (contagem sem smoothing) falha para n-gramas nunca vistos. Depois, explique em uma frase o que perplexity mede intuitivamente.

### Referências
- `Livro` **Speech and Language Processing**, cap. 3 (N-gram Language Models). https://web.stanford.edu/~jurafsky/slp3/
- `Paper` **An Empirical Study of Smoothing Techniques for Language Modeling** — Chen & Goodman (1998). https://aclanthology.org/J99-4007/

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

> **Intuição**: TF-IDF pondera cada palavra por duas forças opostas — frequência no documento (TF, quanto mais aparece ali, mais relevante para aquele documento) contra raridade no corpus inteiro (IDF, palavras que aparecem em todo documento, tipo "de"/"o"/"que", carregam pouca informação distintiva). O resultado é um vetor esparso (a maioria das posições é zero, já que a maioria das palavras do vocabulário não aparece em um documento específico) onde palavras raras-mas-relevantes pesam mais que palavras comuns. A limitação central — "rei" e "monarca" são tratados como tão diferentes quanto "rei" e "abacaxi" — é exatamente o que embeddings densos (seção 6.4) resolvem.

### Modelos de tópicos (cross-link mod. [04](04_ml_moderno.md))
- **Latent Semantic Analysis (LSA)** via SVD em matriz termo-documento.
- **Latent Dirichlet Allocation (LDA)**.
- **Non-negative Matrix Factorization (NMF)**.

### Referências
- `Paper` **Latent Dirichlet Allocation** — Blei, Ng, Jordan (2003). https://www.jmlr.org/papers/v3/blei03a.html

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

> **Intuição**: a hipótese distribucional é, na prática, uma tarefa de pretexto self-supervised (mod. [04](04_ml_moderno.md#41-self-supervised-learning-ssl)) antes desse termo existir formalmente — Word2Vec treina prevendo palavra a partir de contexto (CBOW) ou contexto a partir de palavra (skip-gram), sem nenhum rótulo humano, e o subproduto útil é o vetor intermediário aprendido, não a previsão em si. Palavras que aparecem em contextos parecidos (ex.: "cachorro" e "gato" costumam ter vizinhos parecidos: "meu ___ late/mia", "alimentar o ___") acabam com vetores próximos no espaço aprendido, sem que ninguém tenha dito explicitamente que são semanticamente relacionadas.
>
> **Exemplo resolvido — vector arithmetic**: `rei − homem + mulher ≈ rainha` funciona porque a direção "rei → homem" (o que distingue masculino de feminino, aproximadamente) acaba codificada de forma consistente no espaço vetorial — subtrair "homem" de "rei" isola aproximadamente essa direção de gênero, e somar "mulher" aplica essa mesma direção a partir de outro ponto. Isso não é perfeito nem garantido (é uma propriedade emergente do treinamento, não projetada), mas funciona surpreendentemente bem para relações bem representadas no corpus.
>
> **Aplicação real**: a limitação central de Word2Vec/GloVe — "banco" tem um único vetor fixo, não importa o contexto ("banco do parque" vs "banco financeiro") — é exatamente o problema que embeddings *contextuais* (BERT, GPT — onde o embedding de cada token muda dependendo dos tokens ao redor, via self-attention do mod. [07](07_transformers.mdx)) resolvem. Entender essa limitação é entender por que Transformers foram um salto, não só uma arquitetura mais complexa por complexidade.
>
> **Checkpoint**: sem olhar o texto, explique a hipótese distribucional com suas próprias palavras. Depois, explique por que "banco" ter um único embedding fixo é um problema real (dê um exemplo de frase onde isso causaria erro).

### Papers fundadores (todos obrigatórios para ler)
- `Paper` **Efficient Estimation of Word Representations in Vector Space (Word2Vec)** — Mikolov et al. (2013). https://arxiv.org/abs/1301.3781
- `Paper` **Distributed Representations of Words and Phrases (Word2Vec, negative sampling)** — Mikolov et al. (2013). https://arxiv.org/abs/1310.4546
- `Paper` **GloVe: Global Vectors for Word Representation** — Pennington, Socher, Manning (2014). https://nlp.stanford.edu/pubs/glove.pdf
- `Paper` **Enriching Word Vectors with Subword Information (FastText)** — Bojanowski et al. (2016). https://arxiv.org/abs/1607.04606

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

> **Intuição — BPE**: comece com o texto quebrado em caracteres individuais. Encontre o par de símbolos adjacentes mais frequente no corpus inteiro (ex.: "e" seguido de "s" em muitas palavras) e mescle esse par num novo símbolo único. Repita milhares de vezes. O resultado é um vocabulário onde palavras muito comuns viram um único token ("the", "de") e palavras raras ou desconhecidas se decompõem em subpalavras menores, mas nunca ficam "sem token" — não existe OOV, no pior caso a palavra cai de volta em caracteres individuais, que sempre estão no vocabulário base.
>
> **Exemplo (ilustrativo)**: um corpus com muitas ocorrências de "lower", "lowest", "newer" tende a mesclar "e"+"r" → "er" cedo (aparece em vários lugares), depois talvez "low"+"er" → "lower" se essa combinação for frequente o bastante — mas uma palavra nova e rara como "flowerpot" (nunca vista) ainda seria tokenizável combinando pedaços já conhecidos ("flow", "er", "pot", por exemplo), em vez de virar um único token `<UNK>` desconhecido como aconteceria com um vocabulário fixo de palavras inteiras.
>
> **Aplicação real**: todo LLM moderno usa alguma variante de tokenização por subpalavra — é o que permite um vocabulário gerenciável (dezenas de milhares de tokens, não milhões de palavras possíveis) sem nunca "travar" num texto de entrada, seja código, uma URL, ou uma palavra inventada.
>
> **Checkpoint**: sem olhar o texto, explique por que tokenização por subpalavra elimina o problema de OOV (out-of-vocabulary) que existia com vocabulário de palavras inteiras.

### Referências
- `Paper` **Neural Machine Translation of Rare Words with Subword Units (BPE)** — Sennrich, Haddow, Birch (2015). https://arxiv.org/abs/1508.07909
- `Paper` **SentencePiece** — Kudo & Richardson (2018). https://arxiv.org/abs/1808.06226
- `Curso` **Karpathy — Let's build the GPT Tokenizer** (vídeo + repo `minbpe`). https://www.youtube.com/watch?v=zduSFxRajkE
- `Curso` **Hugging Face Course — Tokenizers**. https://huggingface.co/learn/llm-course

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
- `Livro` **Speech and Language Processing** — caps. dedicados a cada tarefa.
- `Curso` **Stanford CS224N**. https://web.stanford.edu/class/cs224n/

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

> **Intuição**: BLEU e ROUGE são fundamentalmente métricas de *sobreposição de n-gramas* entre a saída gerada e uma referência — contam quantos pedaços de texto batem literalmente. Isso as torna cegas a paráfrases corretas ("o gato está no tapete" vs "há um gato sobre o tapete" — mesmo significado, baixa sobreposição de n-gramas) e vulneráveis a saídas que "colam" trechos da referência sem realmente resolver a tarefa. BERTScore ataca essa limitação comparando *embeddings* em vez de tokens literais — duas frases semanticamente parecidas pontuam alto mesmo com vocabulário diferente, mais alinhado com o julgamento humano de qualidade.
>
> **Checkpoint**: sem olhar o texto, explique por que duas traduções corretas mas com palavras diferentes podem receber BLEU baixo — e por que isso é uma limitação, não um recurso.

### Referências
- `Paper` **GLUE** — Wang et al. (2018). https://arxiv.org/abs/1804.07461
- `Paper` **BERTScore** — Zhang et al. (2019). https://arxiv.org/abs/1904.09675

---

## Projetos práticos

### Projeto 6.1 — Pipeline clássico ponta-a-ponta
- Tarefa: classificação de sentimento em IMDb.
- Pipeline: tokenização → stopwords → stemming → TF-IDF → Logistic Regression.
- Faça em scikit-learn.
- **Meta**: >85% accuracy só com clássico.

### Projeto 6.2 — Treinar Word2Vec do zero
- Use gensim em corpus português (Wikipedia em PT, ou Folha-PT).
- Avalie qualitativamente: analogias (rei − homem + mulher), similaridades.
- **Compare** com embeddings pré-treinados em PT (NILC, fastText).

> **Variante guiada**: antes de testar analogias, verifique similaridades simples primeiro (palavras que deveriam ficar próximas: "cachorro"/"gato", "rei"/"rainha") — se isso já falhar, o problema é o treino (corpus pequeno, poucas épocas), e testar analogias mais complexas antes de resolver isso é perda de tempo.

### Projeto 6.3 — Implementar BPE do zero
- Em Python puro, sem `tokenizers`.
- Treine em um corpus pequeno.
- Compare com `tiktoken` ou `tokenizers` da HF.
- **Referência guia**: o vídeo do Karpathy sobre tokenização.

> **Variante guiada**: rode seu BPE com poucas iterações de merge primeiro (ex.: 10) e imprima o vocabulário resultante — confirme que os merges fazem sentido (pares frequentes tipo "e"+"s") antes de rodar as milhares de iterações necessárias para um tokenizer de verdade.

### Projeto 6.4 — NER com spaCy + customização
- Use modelo PT-BR do spaCy.
- Adicione entidades customizadas (ex: nomes de produtos).
- Avalie com F1 sobre conjunto anotado por você.

### Projeto 6.5 — Comparativo: TF-IDF vs Word2Vec vs BERT-embeddings
- Mesma tarefa de classificação.
- Veja o ganho (ou não) de cada representação.
- **O que isso prova**: que embeddings melhoram, mas não são bala de prata.

---

## Erros comuns

- **Não normalizar Unicode** (NFC vs NFD) — bug clássico em PT-BR (acentos).
- **Stemming agressivo** que junta palavras semanticamente diferentes.
- **Ignorar tokenização** ao migrar entre modelos — cada modelo tem seu tokenizer.
- **Avaliação com leakage** (mesma frase em treino e teste).
- **Usar BLEU como verdade absoluta** — métrica falha, especialmente em PT-BR.

---

## Conexão com módulos seguintes

| Conceito daqui | Aparece em |
|---|---|
| Tokenização (BPE, SentencePiece) | LLMs (mod. [08](08_llms_arquiteturas.md), [09](09_treinamento_e_alinhamento.mdx)) |
| Word embeddings | Embeddings contextuais em LLMs |
| Hipótese distribucional | Toda IA moderna |
| Perplexity | Avaliação de LLMs (mod. [14](14_avaliacao_e_seguranca.md)) |
| Seq2Seq | Encoder-decoder em Transformers (mod. [07](07_transformers.mdx)) |
| Attention sobre RNN (Bahdanau) | Self-attention (mod. [07](07_transformers.mdx)) |
| TF-IDF | Sparse retrieval em RAG (mod. [12](12_rag.mdx)) |

---

## Checklist de saída

- [ ] Implementei BPE do zero.
- [ ] Treinei e avaliei Word2Vec.
- [ ] Sei explicar a hipótese distribucional.
- [ ] Conheço os trade-offs de tokenização: BPE vs WordPiece vs SentencePiece.
- [ ] Posso construir pipeline NLP clássico ponta-a-ponta sem framework "mágico".
- [ ] Entendo limitações de embeddings estáticos — e qual problema BERT/GPT resolvem.
