---
id: 06_nlp_classico
title: "Módulo 06 — NLP Clássico (pré-Transformer)"
sidebar_position: 15
---

# Módulo 06 — NLP Clássico (pré-Transformer)

> **Objetivo**: entender como linguagem foi tratada *antes* dos Transformers — e por que isso ainda importa. Sem essa base, "embeddings", "tokenização" e "contexto" são só palavras que você já usou sem examinar de perto.
>
> **Pré-requisitos**: toda a trilha até aqui (módulos [08](08_llms_arquiteturas.md)–[20](20_projetos_integradores.md) e [07](07_transformers.mdx)) — em particular, o BPE via biblioteca do Projeto 20.2 (aqui você implementa o algoritmo do zero) e a similaridade de cosseno usada desde o Projeto 8.5.
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

É comum pular NLP clássico e ir direto para LLMs. O resultado é tratar embeddings como mágica, não entender suas limitações, não saber debugar um pipeline de texto quando algo dá errado de um jeito que "conversar com o modelo" não resolve. Os conceitos aqui (tokenização, n-gramas, embeddings distribucionais) continuam vivos dentro dos Transformers — você já os usou (embeddings desde o Projeto 8.5, perplexity no Projeto 9.3, BPE via biblioteca no Projeto 20.2) sem necessariamente ter examinado a origem deles. Este módulo faz essa origem explícita.

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
Pipelines profissionais ainda fazem **muito** desse pré-processamento — para limpeza, para extração de metadados, para RAG (mod. 12). Não é "obsoleto", é "aposentado da posição central".

> Um detalhe fácil de subestimar: NFC vs NFD são duas formas *diferentes* de representar em bytes o mesmo caractere acentuado ("é" pode ser um único code point ou "e" + acento combinante) — visualmente idênticos, mas strings diferentes, o que quebra comparação exata e busca se o pipeline não normaliza consistentemente. É um dos bugs mais silenciosos em pipelines PT-BR.

NLTK (clássica, didática), spaCy (industrial, rápida, multilíngue — usada no Projeto 6.4) e Stanza (Stanford NLP) são as bibliotecas de referência em Python; em TS, não há porte sério de spaCy — `compromise` cobre o básico, e para tokenização/NER moderna, `transformers.js` (já usado desde o Projeto 10.5) resolve.

---

## 6.2 Modelos estatísticos de linguagem (n-gramas)

### Conceito
Modelar a probabilidade da próxima palavra dadas as anteriores, com a aproximação de Markov: em vez de condicionar em toda a frase até ali, condiciona só nas últimas `k` palavras.

### Tópicos
- **N-gramas**: unigram, bigram, trigram.
- **Maximum Likelihood Estimation** sobre contagens.
- **Smoothing**: Laplace, Good-Turing, **Kneser-Ney** (estado da arte pré-neural).
- **Perplexity** como métrica de modelo de linguagem — a mesma métrica que você já calculou no Projeto 9.3.

### Por que isso importa
- Perplexity continua sendo a métrica usada para avaliar LLMs.
- Conceitos como "predict next token" começam aqui — o mesmo princípio do `cross_entropy` do Projeto 8.3, numa formulação bem mais simples (contagem, não rede neural).
- A intuição de Markov ajuda a entender contexto limitado.

> **Intuição**: a aproximação de Markov diz "pra prever a próxima palavra, você não precisa da frase inteira até aqui, só das últimas k palavras" — uma simplificação necessária porque contar ocorrências exatas de frases inteiras é inviável (a maioria das frases longas nunca se repete no corpus). Smoothing existe porque contagem pura (MLE) atribui probabilidade zero a qualquer n-grama nunca visto no treino — mesmo que seja uma combinação perfeitamente razoável de palavras. Kneser-Ney redistribui probabilidade dos n-gramas vistos para os não-vistos de forma mais inteligente que só somar 1 a cada contagem (Laplace).
>
> **Exemplo resolvido — perplexity**: perplexity é `2^(entropia cruzada)`, e intuitivamente mede "quantas opções o modelo está, em média, hesitando entre" a cada palavra. Se um modelo atribui probabilidade média de 0.5 à palavra correta a cada passo, sua perplexity é `2^(-log2(0.5)) = 2^1 = 2` — como se estivesse escolhendo entre 2 opções igualmente prováveis a cada palavra. Um modelo pior, com probabilidade média 0.1 na palavra correta, tem perplexity `2^(-log2(0.1)) ≈ 2^3.32 ≈ 10` — como escolher entre 10 opções. Perplexity menor = modelo mais confiante nas palavras certas = melhor modelo de linguagem. É exatamente essa conta (só que com `exp` e log natural, em vez de `2^` e log base 2 — matematicamente equivalente, convenção diferente) que você já calculou em `compute_perplexity` no Projeto 9.3.
>
> **Checkpoint**: sem olhar o texto, explique por que MLE puro (contagem sem smoothing) falha para n-gramas nunca vistos. Depois, explique em uma frase o que perplexity mede intuitivamente.

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

> **Intuição**: TF-IDF pondera cada palavra por duas forças opostas — frequência no documento (TF, quanto mais aparece ali, mais relevante para aquele documento) contra raridade no corpus inteiro (IDF, palavras que aparecem em todo documento, tipo "de"/"o"/"que", carregam pouca informação distintiva). O resultado é um vetor esparso (a maioria das posições é zero, já que a maioria das palavras do vocabulário não aparece em um documento específico) onde palavras raras-mas-relevantes pesam mais que palavras comuns. A limitação central — "rei" e "monarca" são tratados como tão diferentes quanto "rei" e "abacaxi" — é exatamente o que embeddings densos (seção 6.4) resolvem. Você compara diretamente TF-IDF contra embeddings densos no Projeto 6.5, e já usa TF-IDF como metade do hybrid retrieval do Projeto 12.3 (via BM25, uma evolução de TF-IDF).

### Modelos de tópicos
Latent Semantic Analysis (LSA, via SVD sobre a matriz termo-documento), Latent Dirichlet Allocation (LDA, um modelo probabilístico de "tópicos" latentes) e Non-negative Matrix Factorization (NMF) são técnicas clássicas para descobrir estrutura temática num corpus sem supervisão — precursoras conceituais do que embeddings fazem hoje de forma mais rica.

---

## 6.4 Word Embeddings — a revolução pré-Transformer

### A hipótese distribucional
"You shall know a word by the company it keeps" (Firth, 1957). **Palavras com contextos similares têm significados similares.** Isso é a base de todo embedding, inclusive os que você já usa desde o Projeto 8.5.

### Word2Vec
- **CBOW** (Continuous Bag of Words): prediz palavra-alvo a partir do contexto.
- **Skip-gram**: prediz contexto a partir da palavra-alvo.
- **Negative Sampling** (truque que torna o treinamento viável).
- **Vector arithmetic**: rei − homem + mulher ≈ rainha.

### GloVe
Combina co-ocorrência global com janelas locais, treinado diretamente sobre uma matriz de co-ocorrências em vez de prever contexto token a token como Word2Vec.

### FastText
Embeddings em nível de subpalavra (n-gramas de caracteres) — o embedding de uma palavra é composto a partir dos embeddings dos seus pedaços, o que permite gerar um embedding razoável até para palavras nunca vistas no treino (OOV), diferente de Word2Vec/GloVe puros.

### Limitações
- **Estáticos**: "banco" tem o mesmo embedding em "banco do parque" e "banco financeiro".
- Resolvido depois por embeddings contextuais (ELMo, BERT — e o encoder que você já implementou no Projeto 7.3).

> **Intuição**: a hipótese distribucional é, na prática, uma tarefa de pretexto self-supervised antes desse termo existir formalmente — Word2Vec treina prevendo palavra a partir de contexto (CBOW) ou contexto a partir de palavra (skip-gram), sem nenhum rótulo humano, e o subproduto útil é o vetor intermediário aprendido, não a previsão em si. Palavras que aparecem em contextos parecidos (ex.: "cachorro" e "gato" costumam ter vizinhos parecidos: "meu ___ late/mia", "alimentar o ___") acabam com vetores próximos no espaço aprendido, sem que ninguém tenha dito explicitamente que são semanticamente relacionadas. Você treina isso do zero, num corpus real, no Projeto 6.2.
>
> **Exemplo resolvido — vector arithmetic**: `rei − homem + mulher ≈ rainha` funciona porque a direção "rei → homem" (o que distingue masculino de feminino, aproximadamente) acaba codificada de forma consistente no espaço vetorial — subtrair "homem" de "rei" isola aproximadamente essa direção de gênero, e somar "mulher" aplica essa mesma direção a partir de outro ponto. Isso não é perfeito nem garantido (é uma propriedade emergente do treinamento, não projetada), mas funciona surpreendentemente bem para relações bem representadas no corpus.
>
> **Aplicação real**: a limitação central de Word2Vec/GloVe — "banco" tem um único vetor fixo, não importa o contexto ("banco do parque" vs "banco financeiro") — é exatamente o problema que embeddings *contextuais* resolvem: o encoder bidirecional que você já implementou no Projeto 7.3 produz um vetor diferente para "banco" dependendo dos tokens ao redor, via self-attention, porque a representação de cada token é recalculada a cada forward pass, condicionada em todo o resto da frase. Entender essa limitação é entender por que Transformers foram um salto, não só uma arquitetura mais complexa por complexidade.
>
> **Checkpoint**: sem olhar o texto, explique a hipótese distribucional com suas próprias palavras. Depois, explique por que "banco" ter um único embedding fixo é um problema real (dê um exemplo de frase onde isso causaria erro).

---

## 6.5 Tokenização para a era neural

### Por que tokenização avançada importa
- Vocabulário fixo grande consome memória.
- Palavras raras viram OOV.
- Solução: tokenizar em **subpalavras**.

### Algoritmos
- **Byte Pair Encoding (BPE)**: mescla pares mais frequentes iterativamente. Usado em GPT. Detalhado abaixo, e implementado do zero no Projeto 6.3.
- **WordPiece**: variante do BPE (usada em BERT) que escolhe o par a mesclar por um critério de verossimilhança estatística, não só frequência bruta.
- **Unigram Language Model**: em vez de construir o vocabulário mesclando de baixo pra cima (como BPE/WordPiece), começa com um vocabulário grande e remove progressivamente os tokens menos úteis — usado em SentencePiece.
- **SentencePiece**: implementação que trata texto como bytes/caracteres puros (sem pré-tokenização por espaço). Usado em LLaMA, T5, e o mesmo princípio por trás do `pre_tokenizers.ByteLevel()` que você usou no Projeto 20.2.
- **Tiktoken**: implementação de BPE em Rust, usada pela OpenAI — otimizada para velocidade, não para flexibilidade de treino.

> **Intuição — BPE**: comece com o texto quebrado em caracteres individuais. Encontre o par de símbolos adjacentes mais frequente no corpus inteiro (ex.: "e" seguido de "s" em muitas palavras) e mescle esse par num novo símbolo único. Repita milhares de vezes. O resultado é um vocabulário onde palavras muito comuns viram um único token ("the", "de") e palavras raras ou desconhecidas se decompõem em subpalavras menores, mas nunca ficam "sem token" — não existe OOV, no pior caso a palavra cai de volta em caracteres individuais, que sempre estão no vocabulário base. Você já usou uma implementação pronta disso (`tokenizers.models.BPE`) no Projeto 20.2; aqui você implementa o algoritmo de treino em si, em Python puro.
>
> **Exemplo (ilustrativo)**: um corpus com muitas ocorrências de "lower", "lowest", "newer" tende a mesclar "e"+"r" → "er" cedo (aparece em vários lugares), depois talvez "low"+"er" → "lower" se essa combinação for frequente o bastante — mas uma palavra nova e rara como "flowerpot" (nunca vista) ainda seria tokenizável combinando pedaços já conhecidos ("flow", "er", "pot", por exemplo), em vez de virar um único token `<UNK>` desconhecido como aconteceria com um vocabulário fixo de palavras inteiras.
>
> **Aplicação real**: todo LLM moderno usa alguma variante de tokenização por subpalavra — é o que permite um vocabulário gerenciável (dezenas de milhares de tokens, não milhões de palavras possíveis) sem nunca "travar" num texto de entrada, seja código, uma URL, ou uma palavra inventada.
>
> **Checkpoint**: sem olhar o texto, explique por que tokenização por subpalavra elimina o problema de OOV (out-of-vocabulary) que existia com vocabulário de palavras inteiras.

---

## 6.6 Tarefas clássicas de NLP

### Classificação de texto
Análise de sentimento, detecção de spam, categorização de tópicos — a tarefa do Projeto 6.1 e do Projeto 6.5.

### Sequência → sequência (Seq2Seq)
Tradução automática, sumarização, question answering — resolvidos hoje por encoder-decoder (o que você já implementou no Projeto 7.6) ou por LLMs decoder-only tratando tudo como geração de texto.

### Extração de informação
NER (Projeto 6.4), relation extraction, event extraction.

### Parsing
Constituency parsing (a estrutura em árvore de uma frase, por regras gramaticais) e dependency parsing (relações de dependência entre palavras — "sujeito de", "modificador de") — hoje majoritariamente substituídos por modelos neurais ponta-a-ponta, mas ainda usados em pipelines que precisam de estrutura gramatical explícita.

### Conversação clássica
Pipelines com NLU (Natural Language Understanding) + DM (Dialogue Manager) + NLG (Natural Language Generation) — substituídos por LLMs ponta-a-ponta hoje, mas o conceito de separar "entender", "decidir o que fazer", e "gerar a resposta" permanece, inclusive dentro do loop ReAct que você já implementou no Projeto 11.3 (Thought separa entendimento/decisão, a resposta final é geração).

---

## 6.7 Avaliação em NLP

### Métricas
- **Accuracy / F1** (classificação).
- **BLEU** (tradução). Limitações detalhadas abaixo.
- **ROUGE** (sumarização).
- **METEOR**, **chrF** (alternativas a BLEU).
- **Perplexity** (modelagem de linguagem — seção 6.2).
- **BERTScore** (avaliação semântica via embeddings, detalhado abaixo).

### Conjuntos de avaliação clássicos
GLUE e SuperGLUE (benchmarks de tarefas variadas, já mencionados no mod. 14), SQuAD (question answering) e CoNLL (NER, parsing) são os benchmarks de referência da era pré-LLM.

> **Intuição**: BLEU e ROUGE são fundamentalmente métricas de *sobreposição de n-gramas* entre a saída gerada e uma referência — contam quantos pedaços de texto batem literalmente. Isso as torna cegas a paráfrases corretas ("o gato está no tapete" vs "há um gato sobre o tapete" — mesmo significado, baixa sobreposição de n-gramas) e vulneráveis a saídas que "colam" trechos da referência sem realmente resolver a tarefa. BERTScore ataca essa limitação comparando *embeddings* em vez de tokens literais (o mesmo princípio de similaridade de cosseno do Projeto 8.5, aplicado token a token entre a saída e a referência, em vez de comparar frases inteiras) — duas frases semanticamente parecidas pontuam alto mesmo com vocabulário diferente, mais alinhado com o julgamento humano de qualidade.
>
> **Checkpoint**: sem olhar o texto, explique por que duas traduções corretas mas com palavras diferentes podem receber BLEU baixo — e por que isso é uma limitação, não um recurso.

---

## Projetos práticos

### Projeto 6.1 — Pipeline clássico ponta-a-ponta

Você vai construir um classificador de sentimento inteiramente com técnicas pré-neurais, sem nenhuma rede neural, e ver o quão longe isso vai.

**Pré-requisitos**: `pip install scikit-learn nltk`.

```python
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer  # stemmer para português
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

nltk.download("stopwords")
nltk.download("rslp")
stopwords_pt = set(stopwords.words("portuguese"))
stemmer = RSLPStemmer()

def preprocessar(texto):
    palavras = texto.lower().split()
    palavras = [p for p in palavras if p not in stopwords_pt]
    palavras = [stemmer.stem(p) for p in palavras]
    return " ".join(palavras)

textos_processados = [preprocessar(t) for t in textos]  # seu corpus de reviews com rótulo positivo/negativo
X_treino, X_teste, y_treino, y_teste = train_test_split(textos_processados, rotulos, test_size=0.2, random_state=0)

vectorizer = TfidfVectorizer(max_features=10000)
X_treino_tfidf = vectorizer.fit_transform(X_treino)
X_teste_tfidf = vectorizer.transform(X_teste)

classificador = LogisticRegression(max_iter=1000)
classificador.fit(X_treino_tfidf, y_treino)

previsoes = classificador.predict(X_teste_tfidf)
print(f"Accuracy: {accuracy_score(y_teste, previsoes):.3f}")
print(classification_report(y_teste, previsoes))
```

O pipeline completo é: `preprocessar` (lowercase, remoção de stopwords, stemming — reduzindo palavras à sua raiz, então "gostei"/"gostaram"/"gostando" viram formas próximas) → `TfidfVectorizer` (transforma cada texto num vetor esparso, a representação da seção 6.3) → `LogisticRegression` (um classificador linear simples sobre esses vetores). Nenhuma rede neural, nenhum embedding denso — e ainda assim, em datasets como IMDb (reviews de filme em inglês; procure um equivalente em PT-BR, ou traduza um subconjunto), esse pipeline tipicamente ultrapassa 85% de accuracy. Isso não é surpreendente uma vez que você entende TF-IDF: sentimento em reviews é bastante correlacionado com a presença de palavras específicas ("ótimo", "péssimo", "recomendo"), exatamente o tipo de sinal que TF-IDF captura bem.

---

### Projeto 6.2 — Treinar Word2Vec do zero

Você vai treinar embeddings Word2Vec num corpus em português e avaliar qualitativamente se eles capturam relações semânticas reais.

**Pré-requisitos**: `pip install gensim`, um corpus em português (um dump da Wikipedia em PT, ou qualquer corpus de texto corrido de alguns milhões de palavras).

```python
from gensim.models import Word2Vec

corpus_tokenizado = [texto.lower().split() for texto in corpus_pt]  # lista de listas de palavras

modelo = Word2Vec(
    sentences=corpus_tokenizado,
    vector_size=100,
    window=5,
    min_count=5,       # ignora palavras que aparecem menos de 5 vezes — poucas ocorrências não dão sinal suficiente
    sg=1,               # 1 = skip-gram, 0 = CBOW
    epochs=10,
)

print(modelo.wv.most_similar("cachorro"))
```

`window=5` define quantas palavras à esquerda e à direita contam como "contexto" para cada palavra-alvo — a implementação prática da hipótese distribucional da seção 6.4. `sg=1` escolhe skip-gram (prevê contexto a partir da palavra) em vez de CBOW (o oposto); ambos aprendem embeddings, skip-gram costuma funcionar melhor com corpora menores.

**Verifique similaridades simples antes de testar analogias**: `modelo.wv.most_similar("cachorro")` deveria retornar palavras como "gato", "animal", "cão" no topo — se isso já não fizer sentido, o problema está no treino (corpus pequeno demais, poucas épocas, `min_count` mal ajustado), e testar `modelo.wv.most_similar(positive=["rei", "mulher"], negative=["homem"])` (a analogia rei-mulher-homem≈rainha) antes de resolver isso é perda de tempo.

**Compare com embeddings pré-treinados em português** (como os do repositório NILC, ou fastText em PT) nas mesmas consultas de similaridade — embeddings treinados num corpus muito maior tendem a capturar relações mais sutis e menos ruidosas que os seus, treinados num corpus provavelmente bem menor.

---

### Projeto 6.3 — Implementar BPE do zero

Você vai implementar o algoritmo de treino de BPE inteiramente em Python puro, sem a biblioteca `tokenizers` que você usou no Projeto 20.2.

**Pré-requisitos**: nenhuma biblioteca além da padrão.

```python
from collections import Counter, defaultdict

def treinar_bpe(corpus, n_merges):
    # começa com cada palavra quebrada em caracteres, marcando fim de palavra com </w>
    palavras = corpus.split()
    vocabulario = Counter(" ".join(list(palavra)) + " </w>" for palavra in palavras)

    merges = []
    for _ in range(n_merges):
        pares = defaultdict(int)
        for palavra, freq in vocabulario.items():
            simbolos = palavra.split()
            for i in range(len(simbolos) - 1):
                pares[(simbolos[i], simbolos[i + 1])] += freq

        if not pares:
            break

        melhor_par = max(pares, key=pares.get)
        merges.append(melhor_par)

        novo_vocabulario = {}
        bigrama = " ".join(melhor_par)
        substituicao = "".join(melhor_par)
        for palavra, freq in vocabulario.items():
            nova_palavra = palavra.replace(bigrama, substituicao)
            novo_vocabulario[nova_palavra] = freq
        vocabulario = novo_vocabulario

    return merges, vocabulario

corpus_exemplo = "lower lower lower lowest lowest newer newer newer newer wider wider"
merges, vocab_final = treinar_bpe(corpus_exemplo, n_merges=10)
for i, par in enumerate(merges):
    print(f"merge {i}: {par} -> {''.join(par)}")
```

`vocabulario` mapeia cada palavra (já quebrada em símbolos separados por espaço, com `</w>` marcando o fim) à sua frequência no corpus. A cada iteração, `pares` conta a frequência de todo par de símbolos *adjacentes* que aparece em qualquer palavra do vocabulário atual (ponderada pela frequência da palavra); `max(pares, key=pares.get)` encontra o par mais frequente — a mesma operação central descrita na Intuição da seção 6.5. `vocabulario` é então atualizado substituindo esse par pelo símbolo mesclado em toda palavra onde ele aparece, e o processo repete.

**Rode primeiro com poucos merges** (como no exemplo acima, `n_merges=10`) e imprima a lista de `merges` — confirme que os primeiros merges fazem sentido (pares frequentes como `('e', 'r')` no corpus de exemplo, que tem "lower"/"newer"/"wider" repetidos) antes de rodar milhares de iterações num corpus de verdade. Compare o vocabulário final e o comportamento de tokenização do seu BPE com o `tiktoken` ou o `tokenizers` da Hugging Face (do Projeto 20.2) no mesmo corpus — não devem ser idênticos (implementações de produção têm otimizações e detalhes de pré-tokenização diferentes), mas o princípio (mesclar pares frequentes iterativamente) deve produzir vocabulários com "cara" parecida.

---

### Projeto 6.4 — NER com spaCy + customização

Você vai usar um modelo de NER pronto em português e estendê-lo com uma categoria de entidade customizada.

**Pré-requisitos**: `pip install spacy && python -m spacy download pt_core_news_lg`.

```python
import spacy

nlp = spacy.load("pt_core_news_lg")

texto = "A Petrobras anunciou lucro recorde no terceiro trimestre, segundo o CEO João Silva."
doc = nlp(texto)
for ent in doc.ents:
    print(ent.text, ent.label_)
```

O modelo pré-treinado já reconhece entidades genéricas (organizações, pessoas, locais). **Para adicionar uma categoria customizada** (ex.: nomes de produtos específicos do seu domínio, que o modelo genérico não conhece), treine um componente adicional com exemplos anotados por você:

```python
import spacy
from spacy.tokens import DocBin
from spacy.training import Example

exemplos_anotados = [
    ("Comprei um iPhone 15 Pro na loja.", [(11, 24, "PRODUTO")]),  # (início, fim, rótulo) em caracteres
    ("O Galaxy S24 tem uma câmera excelente.", [(2, 12, "PRODUTO")]),
    # anote pelo menos 50-100 exemplos para um resultado razoável
]

nlp_customizado = spacy.blank("pt")
ner = nlp_customizado.add_pipe("ner")
ner.add_label("PRODUTO")

exemplos_treino = []
for texto, entidades in exemplos_anotados:
    doc = nlp_customizado.make_doc(texto)
    exemplos_treino.append(Example.from_dict(doc, {"entities": entidades}))

otimizador = nlp_customizado.begin_training()
for epoca in range(30):
    nlp_customizado.update(exemplos_treino, sgd=otimizador)
```

As tuplas `(início, fim, "PRODUTO")` marcam, por posição de caractere, onde na frase está a entidade de interesse — o formato de anotação que o spaCy espera para treinar um componente de NER do zero. `nlp_customizado.update` roda o equivalente de um passo de treino (forward + backward + step, o mesmo princípio do Projeto 8.3, encapsulado pela API do spaCy).

**Avalie com F1** sobre um conjunto de teste anotado separadamente (não usado no treino): para cada entidade prevista, é um verdadeiro-positivo se bate exatamente (span e rótulo) com uma entidade anotada; calcule precision (das previstas, quantas estavam certas) e recall (das reais, quantas foram encontradas), e F1 como a média harmônica dos dois.

---

### Projeto 6.5 — Comparativo: TF-IDF vs Word2Vec vs embeddings contextuais

Você vai medir, na mesma tarefa de classificação, o ganho (ou não) de cada nível de representação de texto — do mais simples ao mais sofisticado.

**Pré-requisitos**: os Projetos 6.1 e 6.2 completos, mais `sentence-transformers` (já instalado desde o Projeto 8.5).

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. TF-IDF (Projeto 6.1) — já calculado como X_treino_tfidf / X_teste_tfidf

# 2. Word2Vec (Projeto 6.2) — representa cada texto pela média dos embeddings das suas palavras
def texto_para_vetor_w2v(texto, modelo_w2v):
    vetores = [modelo_w2v.wv[palavra] for palavra in texto.lower().split() if palavra in modelo_w2v.wv]
    return np.mean(vetores, axis=0) if vetores else np.zeros(modelo_w2v.vector_size)

X_treino_w2v = np.array([texto_para_vetor_w2v(t, modelo) for t in X_treino])
X_teste_w2v = np.array([texto_para_vetor_w2v(t, modelo) for t in X_teste])

# 3. Embeddings contextuais (Sentence-BERT — a versão de produção do encoder que você implementou no Projeto 7.3)
embedder = SentenceTransformer("intfloat/multilingual-e5-large")
X_treino_bert = embedder.encode(X_treino)
X_teste_bert = embedder.encode(X_teste)

for nome, (X_tr, X_te) in [
    ("TF-IDF", (X_treino_tfidf, X_teste_tfidf)),
    ("Word2Vec (média)", (X_treino_w2v, X_teste_w2v)),
    ("Embeddings contextuais", (X_treino_bert, X_teste_bert)),
]:
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_tr, y_treino)
    acc = accuracy_score(y_teste, clf.predict(X_te))
    print(f"{nome}: {acc:.3f}")
```

As três representações alimentam exatamente o mesmo classificador (`LogisticRegression`) — a única variável é como o texto vira um vetor numérico antes disso. `texto_para_vetor_w2v` representa um texto inteiro pela *média* dos embeddings das palavras que o compõem — uma forma simples (e um tanto grosseira, já que perde ordem e pondera todas as palavras igualmente) de ir de embeddings de palavra para um vetor de documento.

**O que isso prova**: o ganho de accuracy de TF-IDF → Word2Vec → embeddings contextuais normalmente existe, mas raramente é dramático em tarefas de classificação de sentimento simples (onde palavras-chave já carregam a maior parte do sinal, como discutido no Projeto 6.1) — a vantagem de representações mais ricas fica mais visível em tarefas que dependem de nuance semântica real (paráfrase, ironia, ambiguidade), não em qualquer tarefa de texto. Documente em qual, se algum, dos seus testes a diferença foi grande, e tente explicar por quê.

---

## Erros comuns

- **Não normalizar Unicode** (NFC vs NFD) — bug clássico em PT-BR (acentos).
- **Stemming agressivo** que junta palavras semanticamente diferentes.
- **Ignorar tokenização** ao migrar entre modelos — cada modelo tem seu tokenizer (o mesmo erro descrito no mod. 09 sobre chat templates, aqui na camada de tokenização).
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
| TF-IDF | Sparse retrieval em RAG (mod. [12](12_rag.mdx)) |

---

## Checklist de saída

- [ ] Implementei BPE do zero e comparei com uma implementação de produção (se não, revise o Projeto 6.3).
- [ ] Treinei e avaliei Word2Vec num corpus real, validando similaridades antes de analogias (se não, revise o Projeto 6.2).
- [ ] Sei explicar a hipótese distribucional com um exemplo próprio (se não, revise a seção 6.4).
- [ ] Conheço os trade-offs de tokenização: BPE vs WordPiece vs SentencePiece/Unigram (se não, revise a seção 6.5).
- [ ] Construí um pipeline NLP clássico ponta-a-ponta sem nenhuma rede neural, e sei por que ele funciona tão bem quanto funciona (se não, revise o Projeto 6.1).
- [ ] Entendo limitações de embeddings estáticos — e medi, não só argumentei, a diferença para embeddings contextuais numa tarefa real (se não, revise o Projeto 6.5).
