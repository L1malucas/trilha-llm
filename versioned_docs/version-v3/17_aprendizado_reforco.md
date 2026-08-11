---
id: 17_aprendizado_reforco
title: "Módulo 17 — Aprendizado por Reforço"
sidebar_position: 10
---

# Módulo 17 — Aprendizado por Reforço

> **Objetivo**: dominar RL clássico (MDPs, Q-learning, policy gradient) e moderno (DQN, PPO, AlphaZero, RL para LLMs). Conexão direta com RLHF/GRPO do módulo [09](09_treinamento_e_alinhamento.mdx).
>
> **Pré-requisitos**: Módulos [08](08_llms_arquiteturas.md)–[16](16_visao_computacional.md) — em particular, o loop de treino do Projeto 8.3 e o GRPO do Projeto 9.5, que este módulo formaliza e estende. Os módulos de matemática e Deep Learning que aprofundam probabilidade e otimização só vêm mais adiante nesta trilha — a notação matemática necessária (somatórios, esperança, gradiente) é explicada aqui, no ponto em que aparece.
>
> **Tempo de referência**: 5–8 semanas.

## Objetivos de aprendizagem

Ao final deste módulo você deve ser capaz de:

- Formalizar um problema como MDP (estado, ação, recompensa, transição) e explicar a equação de Bellman com intuição, não só a fórmula.
- Explicar a diferença entre métodos value-based (Q-learning, DQN) e policy-based (REINFORCE, PPO).
- Explicar por que PPO usa "clipping" e o que isso evita.
- Explicar por que DPO (mod. 09) pode ser entendido como uma forma de offline RL.
- Escolher entre RL online, offline e RLHF/GRPO para um cenário de treinamento de LLM.

---

## Por que isso importa

RL é o que treina modelos de raciocínio (DeepSeek-R1, o1), agentes que aprendem interagindo com um ambiente (jogos, robótica), e qualquer sistema que precisa tomar uma sequência de decisões cuja recompensa só se revela depois, não imediatamente a cada passo. Você já usou RL na prática — GRPO, no Projeto 9.5 — sem o formalismo completo por trás. Mesmo que você nunca treine um agente de RL "puro" fora do contexto de LLMs, RLHF e GRPO usam exatamente as mesmas ferramentas matemáticas descritas aqui; sem esse formalismo, o que acontece durante alinhamento de um LLM continua sendo uma caixa-preta.

---

## 17.1 Formalismo: Markov Decision Process (MDP)

### Componentes
- **Estado** `S`.
- **Ação** `A`.
- **Transição** `P(s' | s, a)` — probabilidade de ir para o estado `s'` dado que você estava em `s` e tomou a ação `a`.
- **Recompensa** `R(s, a, s')`.
- **Desconto** `γ ∈ [0, 1]`.
- **Política** `π(a | s)` — a estratégia: probabilidade de tomar a ação `a` dado o estado `s`.

### Quantidades-chave
- **Retorno**: `G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ...` — a soma de todas as recompensas futuras, cada uma "descontada" por `γ` elevado ao número de passos à frente.
- **Value function** `V^π(s)`: o retorno *esperado* (a média, sobre toda a aleatoriedade possível do ambiente e da política) se você começar no estado `s` e seguir a política `π` daí em diante.
- **Action-value (Q-function)** `Q^π(s, a)`: o mesmo, mas fixando também a primeira ação `a`, antes de voltar a seguir `π`.
- **Equação de Bellman**: detalhada abaixo.

> **Intuição**: um MDP formaliza "tomar decisões em sequência, cada uma afetando o futuro". Estado é "onde você está agora"; ação é "o que você pode fazer"; a propriedade de Markov diz que o futuro depende só do estado atual, não de como chegou ali (o histórico completo não importa, só a situação presente); recompensa é o sinal de feedback, frequentemente atrasado (uma ação boa agora pode só compensar muitos passos depois). O desconto `γ` resolve um problema prático: sem ele, recompensas infinitas no futuro tornariam o retorno total indefinido; `γ < 1` faz recompensas futuras valerem progressivamente menos que recompensas imediatas — parecido com preferir dinheiro hoje a dinheiro daqui a 10 anos. "Esperado" (na definição de `V^π`) significa a média ponderada por probabilidade sobre todos os futuros possíveis — o mesmo conceito de valor esperado que aparece implicitamente sempre que você calcula uma média de resultados de amostragem aleatória (como o `torch.multinomial` do Projeto 8.3 fazendo amostragem, repetido muitas vezes, converge para esse valor médio).
>
> A **equação de Bellman** é a peça central: ela diz que o valor de um estado é a recompensa imediata mais o valor (descontado) de onde você vai parar depois — `V(s) = recompensa_imediata + γ × V(próximo_estado)`. É uma definição *recursiva*, e essa recursão é exatamente o que torna RL computacionalmente tratável: em vez de precisar simular o futuro infinito para saber o valor de um estado, você pode calcular valores por bootstrapping (usando a estimativa atual do valor do próximo estado para atualizar a estimativa do estado atual), que é a ideia central por trás de Temporal Difference learning (seção 17.2).
>
> **Checkpoint**: sem olhar o texto, explique a propriedade de Markov com suas palavras. Depois, explique por que a equação de Bellman ser recursiva é útil computacionalmente.

O livro de Sutton & Barto ("Reinforcement Learning: An Introduction", disponível gratuitamente online) é a referência canônica mais citada da área, e o curso de David Silver (DeepMind/UCL) cobre essencialmente o mesmo material em formato de aula.

---

## 17.2 RL clássico tabular

### Métodos
- **Dynamic Programming**: policy iteration, value iteration (assume modelo do ambiente conhecido — você sabe `P` e `R` de antemão).
- **Monte Carlo methods**: aprendem por amostragem de episódios completos.
- **Temporal Difference (TD)**: bootstrap, atualização passo a passo.
- **SARSA** (on-policy TD).
- **Q-learning** (off-policy TD) — implementado do zero no Projeto 17.1.
- **Eligibility traces** (TD(λ)).

> **Intuição — Monte Carlo vs TD**: Monte Carlo espera o episódio *terminar* pra então atualizar os valores com o retorno real observado — preciso, mas lento (só aprende depois de completar uma partida inteira, por exemplo) e alta variância. TD atualiza a cada passo, usando a estimativa atual do próximo estado como aproximação do retorno futuro (bootstrapping, via Bellman) — mais rápido para aprender, funciona até em tarefas contínuas sem fim natural, mas introduz viés (a estimativa usada pode estar errada no início do treino). Q-learning é *off-policy*: aprende sobre a melhor ação possível (o máximo sobre as ações disponíveis), mesmo enquanto explora com uma política diferente (ex.: aleatória parte do tempo) — permite aprender sobre a política ótima enquanto ainda explora o ambiente.
>
> **Checkpoint**: sem olhar o texto, explique a diferença entre Monte Carlo e TD learning — qual espera o episódio terminar?

---

## 17.3 Deep RL: combinando RL com redes neurais

### DQN (Deep Q-Network)
Generaliza Q-learning substituindo a tabela de valores Q (que só funciona quando estados e ações são finitos e enumeráveis) por uma rede neural que aproxima `Q(s, a)` — necessário assim que o espaço de estados é grande demais ou contínuo (como pixels de um jogo de Atari) para caber numa tabela. Implementado do zero no Projeto 17.2.

### Truques que tornaram DQN viável
- **Experience replay** (descorrelacionar amostras).
- **Target network** (estabilidade).
- **Reward clipping**.
- **Frame stacking**.

> **Intuição**: treinar uma rede neural assume implicitamente que os exemplos de treino são razoavelmente independentes — o mesmo tipo de embaralhamento que `get_batch` faz no Projeto 8.3, pegando pedaços aleatórios do corpus em vez de sempre a mesma sequência. Mas em RL, experiências consecutivas (estado, ação, recompensa, próximo estado) são altamente correlacionadas (o estado no passo `t+1` depende diretamente do estado em `t`). Treinar diretamente nessa sequência correlacionada desestabiliza o treino. Experience replay quebra essa correlação: guarda experiências num buffer e amostra *aleatoriamente* dele para cada passo de treino. Target network resolve um problema relacionado — usar a mesma rede que está sendo atualizada como alvo do próprio treino (bootstrapping) cria um "alvo em movimento" instável; congelar uma cópia (target network) por um período e só atualizá-la periodicamente estabiliza o treino.

### Variações
Double DQN, Dueling DQN e Prioritized Experience Replay atacam, cada uma, uma fonte específica de instabilidade ou ineficiência do DQN original; Rainbow combina várias dessas melhorias no mesmo agente.

---

## 17.4 Policy Gradient methods

### Conceito
Em vez de aprender Q e derivar a política dela (escolher sempre a ação de maior valor), aprenda diretamente uma política `π_θ` — uma rede que recebe o estado e produz uma distribuição de probabilidade sobre ações.

### REINFORCE
O algoritmo básico de policy gradient: gera um episódio completo, observa o retorno final, e ajusta os pesos da política para aumentar a probabilidade das ações que levaram a um retorno bom (e diminuir a das que levaram a um retorno ruim) — o mesmo mecanismo de `loss.backward()` + `optimizer.step()` do Projeto 8.3, só que o "sinal de erro" vem do retorno observado, não de um rótulo conhecido.

### Actor-Critic
Combina uma policy network (actor, que decide a ação) com uma value network (critic, que estima o valor esperado do estado) para reduzir a variância do sinal de treino.

> **Intuição — value-based vs policy-based**: Q-learning/DQN aprendem "o quanto vale cada ação em cada estado" e a política emerge implicitamente (escolha a ação de maior valor). Policy gradient aprende a política *diretamente* — uma rede que recebe estado e produz uma distribuição de probabilidade sobre ações, ajustada via gradiente pra aumentar a probabilidade de ações que levaram a bom retorno. Isso é mais natural para ações contínuas (robótica — não dá pra enumerar "o valor de cada ação possível" quando ações são um vetor contínuo) e permite políticas estocásticas genuínas (útil quando a estratégia ótima envolve aleatoriedade, como blefar num jogo). REINFORCE tem alta variância porque estima o gradiente a partir de retornos de episódios completos amostrados — Actor-Critic reduz essa variância usando uma rede "crítica" que estima o valor esperado, permitindo comparar "esse retorno foi melhor ou pior que o esperado" (a vantagem) em vez do retorno bruto, um sinal de treino mais estável.

### A2C / A3C
Asynchronous Advantage Actor-Critic — várias cópias do agente rodando em paralelo, em ambientes independentes, compartilhando atualizações da mesma rede.

### TRPO
Trust Region Policy Optimization — restringe formalmente (via KL-divergence, a mesma medida de "distância" entre distribuições de probabilidade usada em DPO no mod. 09) o quanto a política pode mudar a cada passo, para estabilidade.

### PPO (Proximal Policy Optimization)
O algoritmo padrão moderno — uma simplificação de TRPO usando "clipping" em vez de uma restrição formal de KL. Usado em RLHF (mod. 09), jogos, robótica. Implementado com Stable-Baselines3 no Projeto 17.3, e é o mesmo algoritmo por trás do PPO-based RLHF que você reproduz no Projeto 17.4.

> **Intuição — por que "clipping"**: um passo de gradiente grande demais numa política pode mudá-la drasticamente de uma vez — e diferente de supervised learning, onde um passo ruim só piora a loss temporariamente, em RL um passo ruim pode levar o agente a uma região do espaço de políticas da qual é difícil se recuperar (a política nova explora de forma tão diferente que os dados coletados com ela deixam de ser úteis para continuar melhorando). TRPO resolve isso restringindo formalmente o quanto a política pode mudar por passo — mas essa restrição é computacionalmente cara. PPO simplifica: em vez de uma restrição matemática rígida, "recorta" (clip) o quanto o gradiente pode empurrar a política pra mudar por passo — muito mais barato de computar, com estabilidade prática similar. É essa simplicidade e estabilidade que tornou PPO o algoritmo padrão tanto em RL clássico quanto em RLHF (mod. [09](09_treinamento_e_alinhamento.mdx#95-alinhamento-via-preferências)).
>
> **Checkpoint**: sem olhar o texto, explique por que um passo de gradiente grande demais é mais perigoso em RL do que em supervised learning comum.

### GRPO (Group Relative Policy Optimization)
A variante usada no DeepSeek-R1/Math, sem value model separado — você já implementou isso no Projeto 9.5, com `GRPOTrainer` da `trl`; a seção 17.8 revisita a conexão formal.

### DDPG, TD3, SAC
Algoritmos desenhados especificamente para espaços de ação contínuos (robótica) — fora do escopo prático deste módulo, mas relevantes se você seguir para RL aplicado a controle físico.

---

## 17.5 Model-based RL e planning

### Conceito
Em vez de aprender só a política ou os valores diretamente da experiência (model-free), aprender um modelo do próprio ambiente (`P` e `R`) e planejar dentro desse modelo aprendido.

### Métodos
Dyna-Q combina experiência real com experiência simulada gerada por um modelo aprendido; MuZero (que também alimenta AlphaZero, seção 17.6) planeja inteiramente dentro de um modelo aprendido, sem nem precisar conhecer as regras do ambiente explicitamente; Dreamer V3 é uma abordagem model-based moderna aplicada a uma gama ampla de domínios com os mesmos hiperparâmetros.

### Por que importa
Eficiência de amostra (sample efficiency) é frequentemente 10–100× melhor.

> Model-free (DQN, PPO) aprende só de experiência real, que é cara de coletar (cada passo exige interagir com o ambiente de verdade). Model-based aprende um "simulador" aproximado do ambiente e pode planejar/treinar dentro desse simulador, gerando experiência sintética barata — o ganho de sample efficiency vem daí, ao custo de depender da qualidade do modelo aprendido.

---

## 17.6 Marcos históricos: AlphaGo, AlphaZero, MuZero

AlphaGo (2016) foi o primeiro sistema a vencer campeões humanos de Go, combinando redes neurais treinadas em partidas humanas com Monte Carlo Tree Search (MCTS, implementado do zero no Projeto 17.6). AlphaGo Zero removeu a dependência de partidas humanas, aprendendo inteiramente por self-play. AlphaZero generalizou o método para outros jogos (xadrez, shogi) sem conhecimento específico de domínio além das regras. MuZero foi um passo além: nem as regras do jogo são dadas explicitamente — o próprio modelo do ambiente é aprendido.

### Lições
- **MCTS** combinado com deep learning é poderoso: a rede neural guia a busca (estimando quais ramos da árvore valem a pena explorar), e a busca refina a estimativa da rede.
- Self-play é um caminho para superar performance humana sem depender de dados humanos como teto de qualidade.

---

## 17.7 Multi-agent RL e RL em ambientes complexos

MARL (Multi-Agent RL) estende RL para cenários com múltiplos agentes aprendendo simultaneamente — cooperando, competindo, ou uma mistura das duas. PettingZoo é a biblioteca mais usada para ambientes multi-agente; Procgen e Crafter são benchmarks desenhados especificamente para testar generalização (em vez de memorização de um único ambiente fixo); MineRL explora RL em ambientes tipo Minecraft; Habitat 2.0 e Isaac Gym/Lab focam em robótica simulada.

---

## 17.8 RLHF, RLAIF e RL em LLMs

Com o formalismo deste módulo, o pipeline RLHF do mod. [09](09_treinamento_e_alinhamento.mdx#95-alinhamento-via-preferências) fica mais claro: o "estado" é o texto gerado até agora, a "ação" é o próximo token, a "recompensa" vem do reward model (ou, em reasoning RL, de um verificador automático de corretude, como a `reward_correct_answer` que você escreveu no Projeto 9.5), e PPO/GRPO são os mesmos algoritmos da seção 17.4, aplicados a esse MDP específico onde o "ambiente" é simplesmente o processo de geração de texto token a token.

### Pipeline RLHF
1. SFT base.
2. Reward Model treinado em pares de preferência.
3. PPO contra o RM, com KL-penalty contra SFT inicial.

Você reproduz esse pipeline completo (incluindo o treino do reward model, que o Projeto 9.2 pulou ao usar DPO diretamente) no Projeto 17.4.

### RLAIF
RL com feedback de **outro LLM** (o mesmo princípio de Constitutional AI, mod. 09) em vez de anotação humana — o "juiz" que você já usou nos Projetos 9.2 e 14.2 substituindo o papel do avaliador humano no pipeline de RLHF.

### Reasoning RL (R1-style)
Recompensa **direta** em corretude (math, code) — sem reward model, geralmente via GRPO, o mesmo que você já implementou no Projeto 9.5. A cadeia de raciocínio longa emerge organicamente do processo de treino, sem ter sido demonstrada explicitamente em nenhum exemplo.

---

## 17.9 Offline RL

### Definição
Aprender uma política a partir de um dataset fixo, sem interagir com o ambiente durante o treino.

### Por que importa
- Robótica/medicina onde explorar ativamente (testar ações no mundo real) é caro ou perigoso.
- Aproveitamento de logs existentes, já coletados por outra política.
- Conexão direta com LLMs: **DPO é, em essência, offline RL**.

### Métodos
Conservative Q-Learning (CQL), BCQ e IQL atacam o problema central de offline RL (detalhado abaixo) de formas distintas; Decision Transformer reformula RL inteiramente como um problema de modelagem de sequência — trata (retorno desejado, estado, ação) como uma sequência de tokens e usa um Transformer autoregressivo, literalmente a mesma arquitetura do Projeto 8.3, para prever a próxima ação — implementado no Projeto 17.7.

> **Intuição**: RL online (PPO, DQN) coleta experiência nova a cada passo de treino, interagindo com o ambiente real. Offline RL aprende só do que já está registrado num dataset fixo, sem poder testar ações novas no ambiente real — o desafio central é que a política aprendida pode "superestimar" o valor de ações raras ou ausentes no dataset (nunca foram testadas de verdade, então não há como saber se são boas). Essa é exatamente a situação de DPO (mod. [09](09_treinamento_e_alinhamento.mdx#95-alinhamento-via-preferências)): o modelo aprende de um dataset fixo de pares de preferência, sem gerar e testar novas respostas em tempo real contra um reward model — é offline RL aplicado a alinhamento de LLM, o que explica por que DPO é mais estável (mas também mais limitado pela qualidade/cobertura do dataset de preferências) que RLHF via PPO online.

---

## 17.10 Bibliotecas e ambientes

Gymnasium (sucessor do OpenAI Gym) é o padrão para ambientes de RL de propósito geral, usado nos Projetos 17.1 e 17.2; PettingZoo estende isso para multi-agente; MuJoCo cobre física contínua (hoje gratuito); Isaac Lab/Gym (NVIDIA) roda simulação de robótica em GPU, em escala.

Entre os frameworks de algoritmos, Stable-Baselines3 (PPO, SAC, DQN, entre outros, em PyTorch — usado no Projeto 17.3) é o mais usado para aplicar algoritmos já implementados; CleanRL, ao contrário, implementa cada algoritmo num único arquivo curto, sem abstrações escondidas atrás de classes genéricas — excelente para aprender o que cada linha faz, e uma referência que vale consultar (sem copiar) ao implementar DQN do zero no Projeto 17.2; RLlib (Ray) foca em produção e treino distribuído; TRL, que você já usa desde o mod. 09, é a ponte entre RL e LLMs especificamente.

---

## Projetos práticos

### Projeto 17.1 — Q-learning tabular do zero

Você vai implementar Value Iteration e Q-learning em NumPy puro, sem nenhuma biblioteca de RL, em dois ambientes simples do Gymnasium.

**Pré-requisitos**: `pip install gymnasium numpy matplotlib`.

**1. Value Iteration primeiro** (o ambiente é totalmente conhecido — `FrozenLake` expõe as probabilidades de transição — o que dá uma referência confiável para validar o Q-learning depois):

```python
import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", is_slippery=False)
n_estados, n_acoes = env.observation_space.n, env.action_space.n

def value_iteration(env, gamma=0.99, theta=1e-8):
    V = np.zeros(n_estados)
    while True:
        delta = 0
        for s in range(n_estados):
            valores_acao = []
            for a in range(n_acoes):
                valor = sum(
                    prob * (recompensa + gamma * V[proximo_estado] * (not terminou))
                    for prob, proximo_estado, recompensa, terminou in env.unwrapped.P[s][a]
                )
                valores_acao.append(valor)
            V[s] = max(valores_acao)
            delta = max(delta, abs(V[s] - max(valores_acao)))
        if delta < theta:
            break
    return V

V_otimo = value_iteration(env)
print(V_otimo.reshape(4, 4))
```

`env.unwrapped.P[s][a]` é a tabela de transições que o Gymnasium expõe para ambientes tabulares simples: para cada estado `s` e ação `a`, lista os `(probabilidade, próximo_estado, recompensa, terminou)` possíveis — exatamente `P(s'|s,a)` e `R(s,a,s')` da seção 17.1, dados explicitamente. O loop aplica a equação de Bellman repetidamente até os valores pararem de mudar significativamente (`delta < theta`) — é a definição recursiva da seção 17.1, calculada por iteração até convergência.

**2. Q-learning, que aprende só por interação** (sem acessar `env.unwrapped.P`):

```python
def q_learning(env, episodios=5000, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.999):
    Q = np.zeros((n_estados, n_acoes))
    retornos_por_episodio = []

    for ep in range(episodios):
        estado, _ = env.reset()
        retorno_total = 0
        terminou = False
        while not terminou:
            if np.random.rand() < epsilon:
                acao = env.action_space.sample()  # exploração
            else:
                acao = np.argmax(Q[estado])  # exploração da melhor ação conhecida até agora

            proximo_estado, recompensa, terminou, truncado, _ = env.step(acao)
            Q[estado, acao] += alpha * (recompensa + gamma * np.max(Q[proximo_estado]) - Q[estado, acao])
            estado = proximo_estado
            retorno_total += recompensa
            terminou = terminou or truncado

        epsilon *= epsilon_decay
        retornos_por_episodio.append(retorno_total)

    return Q, retornos_por_episodio

Q_aprendido, retornos = q_learning(env)
```

`Q[estado, acao] += alpha * (recompensa + gamma * np.max(Q[proximo_estado]) - Q[estado, acao])` é a atualização TD (Temporal Difference) da seção 17.2: `recompensa + gamma * np.max(Q[proximo_estado])` é a estimativa do valor da ação, usando o valor *atualmente conhecido* do próximo estado (bootstrapping); a diferença entre essa estimativa e o valor antigo, escalada por `alpha` (a taxa de aprendizado), é o ajuste aplicado. `epsilon` controla exploração vs aproveitamento (`epsilon`-greedy): começa alto (age aleatoriamente, explorando) e decai a cada episódio (passa a confiar mais no que já aprendeu).

**3. Compare** `Q_aprendido.max(axis=1)` (o valor de cada estado segundo Q-learning) com `V_otimo` da Value Iteration — devem ser próximos, já que ambos convergem para a mesma solução ótima nesse ambiente simples, apesar de terem acesso a informação completamente diferente (modelo conhecido vs só interação). Plote `retornos_por_episodio` (com `matplotlib`, o mesmo padrão do Projeto 8.4) para visualizar a curva de aprendizado — deve subir e estabilizar conforme `epsilon` decai.

---

### Projeto 17.2 — DQN no CartPole

Você vai implementar DQN do zero em PyTorch, com replay buffer e target network, seguindo CleanRL como referência de estrutura mas escrevendo o código você mesmo.

**Pré-requisitos**: `pip install gymnasium torch`.

```python
import torch
import torch.nn as nn
import random
from collections import deque

env = gym.make("CartPole-v1")
n_estados = env.observation_space.shape[0]
n_acoes = env.action_space.n

class QNetwork(nn.Module):
    def __init__(self, n_estados, n_acoes):
        super().__init__()
        self.rede = nn.Sequential(nn.Linear(n_estados, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, n_acoes))

    def forward(self, x):
        return self.rede(x)

q_network = QNetwork(n_estados, n_acoes)
target_network = QNetwork(n_estados, n_acoes)
target_network.load_state_dict(q_network.state_dict())  # começa idêntica à q_network
optimizer = torch.optim.AdamW(q_network.parameters(), lr=1e-3)

replay_buffer = deque(maxlen=10000)

def treinar_passo(batch_size=64, gamma=0.99):
    if len(replay_buffer) < batch_size:
        return
    batch = random.sample(replay_buffer, batch_size)
    estados, acoes, recompensas, proximos_estados, terminou = map(torch.tensor, zip(*batch))

    q_valores = q_network(estados.float()).gather(1, acoes.unsqueeze(1).long()).squeeze()
    with torch.no_grad():
        proximo_q_max = target_network(proximos_estados.float()).max(1).values
        alvo = recompensas + gamma * proximo_q_max * (~terminou)

    loss = nn.functional.mse_loss(q_valores, alvo)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

`replay_buffer` é uma `deque` de tamanho fixo (descarta as experiências mais antigas quando cheia) — o buffer de experience replay da seção 17.3. `treinar_passo` amostra um lote aleatório dele (`random.sample`), quebrando a correlação temporal entre experiências consecutivas. `q_network(...).gather(1, acoes...)` seleciona, para cada linha do batch, o Q-valor especificamente da ação que foi tomada (a rede produz um valor por ação possível; só nos importa o da ação realmente executada). `target_network` (não `q_network`) é usada para calcular `alvo` — a rede "congelada" que estabiliza o treino, como descrito na seção 17.3; `with torch.no_grad()` porque não queremos propagar gradiente através do cálculo do alvo, só através da previsão atual (`q_valores`).

O loop de treino completo intercala: rodar episódios com `epsilon`-greedy (igual ao Projeto 17.1, mas escolhendo ações via `q_network(estado).argmax()` em vez de uma tabela), guardar cada transição em `replay_buffer`, chamar `treinar_passo` a cada alguns passos, e copiar `q_network` para `target_network` periodicamente (a cada N episódios, não a cada passo). Valide primeiro em CartPole (rápido de rodar, poucos minutos para convergir); Atari (Pong, Breakout, via `ale-py`) é um passo natural seguinte, mas exige bem mais compute e frame stacking (empilhar os últimos N frames como entrada, para o agente perceber movimento).

---

### Projeto 17.3 — PPO em ambiente contínuo

Você vai treinar PPO com Stable-Baselines3 num ambiente de ação contínua, e plotar as curvas de treino.

**Pré-requisitos**: `pip install stable-baselines3 gymnasium[mujoco]`.

```python
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import matplotlib.pyplot as plt

env = Monitor(gym.make("Pendulum-v1"))
modelo = PPO("MlpPolicy", env, verbose=1, n_steps=2048, batch_size=64, learning_rate=3e-4)
modelo.learn(total_timesteps=100_000)

recompensas_por_episodio = env.get_episode_rewards()
plt.plot(recompensas_por_episodio)
plt.xlabel("Episódio")
plt.ylabel("Recompensa total")
plt.title("Curva de treino — PPO em Pendulum")
plt.savefig("ppo_curva.png")
```

`Monitor` envolve o ambiente e registra automaticamente a recompensa total de cada episódio, o que `env.get_episode_rewards()` expõe depois — o equivalente pronto ao que você calculou manualmente em `retornos_por_episodio` no Projeto 17.1. Diferente do DQN do Projeto 17.2 (que produz um único valor por ação discreta), `Pendulum-v1` tem ação contínua (o torque aplicado, um número real), e a política de PPO aprende diretamente uma distribuição de probabilidade sobre esse valor contínuo — exatamente a vantagem de policy-based sobre value-based mencionada na seção 17.4. Depois de validar em Pendulum, tente `BipedalWalker-v3` ou um ambiente MuJoCo (`Hopper-v4`) para uma tarefa de controle mais complexa.

---

### Projeto 17.4 — Reproduzir RLHF completo (com Reward Model + PPO)

Diferente do Projeto 9.2 (que usou DPO, pulando o reward model), você vai reproduzir o pipeline RLHF completo — treinar um reward model e depois otimizar a política contra ele via PPO — e comparar o resultado com DPO no mesmo dataset.

**Pré-requisitos**: os mesmos do Projeto 9.1/9.2 (`transformers`, `peft`, `trl`).

**1. Treine um reward model**, usando o mesmo dataset de preferências do Projeto 9.2 (`preferences.jsonl`, gerado ali via "juízes" no Ollama):

```python
from trl import RewardConfig, RewardTrainer
from transformers import AutoModelForSequenceClassification

reward_model = AutoModelForSequenceClassification.from_pretrained("Qwen/Qwen2.5-0.5B", num_labels=1)

reward_config = RewardConfig(output_dir="./reward-model", per_device_train_batch_size=4, num_train_epochs=1)
reward_trainer = RewardTrainer(model=reward_model, args=reward_config, train_dataset=dpo_dataset)  # do Projeto 9.2
reward_trainer.train()
```

`AutoModelForSequenceClassification` com `num_labels=1` transforma o modelo numa rede que produz um único número (o score de recompensa) em vez de uma distribuição sobre o vocabulário — a arquitetura muda de "gerar texto" para "avaliar texto". `RewardTrainer` treina esse modelo a atribuir um score maior à resposta `chosen` do que à `rejected`, para cada par do dataset — literalmente o passo 2 do pipeline RLHF da seção 17.8.

**2. Rode PPO contra o reward model treinado**:

```python
from trl import PPOConfig, PPOTrainer

ppo_config = PPOConfig(output_dir="./qwen-ppo", per_device_train_batch_size=4, learning_rate=1e-5)
ppo_trainer = PPOTrainer(
    args=ppo_config,
    model=model,  # o modelo pós-SFT do Projeto 9.1
    reward_model=reward_model,
    train_dataset=prompts_dataset,  # só os prompts, sem chosen/rejected — PPO gera as respostas ele mesmo
)
ppo_trainer.train()
```

A diferença estrutural em relação a DPO: PPO **gera** respostas ativamente durante o treino (usando o modelo atual) e pede ao `reward_model` para pontuá-las, ajustando a política para aumentar a pontuação esperada — é RL online de verdade, com o reward model no papel do "ambiente" que fornece recompensa, exatamente como formalizado na seção 17.8. DPO, em contraste, nunca gera nada durante o treino — só otimiza uma loss supervisionada sobre pares já fixados.

**3. Compare** os três modelos (SFT puro, SFT+DPO do Projeto 9.2, SFT+PPO deste projeto) no mesmo conjunto de prompts de teste, com a mesma função `generate_response` do Projeto 9.1. Documente qual foi mais estável de treinar (PPO tem mais hiperparâmetros sensíveis — veja "Erros comuns") e se a qualidade final percebida difere.

---

### Projeto 17.5 — Reasoning RL com GRPO em tarefa mais difícil

Você vai estender o Projeto 9.5 (GRPO em aritmética simples) para uma tarefa mais próxima de GSM8K, com problemas de texto multi-passo em vez de expressões aritméticas diretas.

**Pré-requisitos**: os mesmos do Projeto 9.5.

O código é estruturalmente o mesmo do Projeto 9.5 (`GRPOConfig`, `GRPOTrainer`, uma função de recompensa que verifica a resposta) — o que muda é o dataset e a função de recompensa:

```python
def gerar_problema_texto():
    nomes = ["Maria", "João", "Ana", "Pedro"]
    nome = random.choice(nomes)
    inicial = random.randint(5, 50)
    compra = random.randint(1, 10)
    preco_unitario = random.randint(2, 20)
    gasto = compra * preco_unitario
    resposta = inicial - gasto if inicial >= gasto else inicial + gasto
    operacao = "gastou" if inicial >= gasto else "ganhou"
    return {
        "prompt": f"{nome} tinha R$ {inicial}. Comprou {compra} itens a R$ {preco_unitario} cada. "
                   f"Quanto {nome} tem agora, sabendo que {operacao} nessa compra?",
        "answer": str(resposta),
    }

problemas_texto = [gerar_problema_texto() for _ in range(500)]
```

A função de recompensa (`reward_correct_answer`, do Projeto 9.5) não muda — ainda extrai um número da resposta e compara com o gabarito. O que muda é que, para resolver esses problemas, o modelo precisa primeiro *entender* o enunciado em texto (não só calcular uma expressão já dada), o que exige uma cadeia de raciocínio mais rica para chegar à resposta certa. Compare a taxa de acerto antes e depois do treino GRPO, e observe se as respostas geradas depois do treino mostram mais passos intermediários de raciocínio explícito do que antes — o comportamento emergente descrito na seção 17.8, em miniatura.

---

### Projeto 17.6 — MCTS + UCT em Tic-Tac-Toe

Você vai implementar Monte Carlo Tree Search do zero, sem nenhuma rede neural, para jogar Tic-Tac-Toe (ou Connect Four) de forma quase ótima.

**Pré-requisitos**: nenhuma biblioteca além de `numpy`.

```python
import math
import random

class NoMCTS:
    def __init__(self, estado, pai=None, acao_que_levou_aqui=None):
        self.estado = estado
        self.pai = pai
        self.acao_que_levou_aqui = acao_que_levou_aqui
        self.filhos = []
        self.visitas = 0
        self.vitorias = 0
        self.acoes_nao_expandidas = acoes_legais(estado)

    def ucb1(self, c=1.41):
        if self.visitas == 0:
            return float("inf")
        return (self.vitorias / self.visitas) + c * math.sqrt(math.log(self.pai.visitas) / self.visitas)

def mcts(estado_raiz, n_simulacoes=1000):
    raiz = NoMCTS(estado_raiz)

    for _ in range(n_simulacoes):
        no = raiz
        # 1. Seleção: desce a árvore escolhendo sempre o filho de maior UCB1, até achar um nó não totalmente expandido
        while not no.acoes_nao_expandidas and no.filhos:
            no = max(no.filhos, key=lambda filho: filho.ucb1())

        # 2. Expansão: adiciona um novo filho para uma ação ainda não testada
        if no.acoes_nao_expandidas:
            acao = no.acoes_nao_expandidas.pop()
            novo_estado = aplicar_acao(no.estado, acao)
            filho = NoMCTS(novo_estado, pai=no, acao_que_levou_aqui=acao)
            no.filhos.append(filho)
            no = filho

        # 3. Simulação (rollout): joga aleatoriamente até o fim da partida a partir daqui
        resultado = simular_partida_aleatoria(no.estado)

        # 4. Retropropagação: propaga o resultado da simulação de volta até a raiz
        while no is not None:
            no.visitas += 1
            no.vitorias += resultado
            no = no.pai

    melhor_filho = max(raiz.filhos, key=lambda filho: filho.visitas)
    return melhor_filho.acao_que_levou_aqui
```

As 4 fases (Seleção, Expansão, Simulação, Retropropagação) são o algoritmo MCTS completo. UCB1 (`ucb1`) é a fórmula que equilibra exploração e aproveitamento na fase de seleção: o primeiro termo (`vitorias/visitas`) favorece ramos historicamente bons; o segundo termo cresce quando um nó foi pouco visitado relativo ao pai, empurrando a busca a experimentar ramos menos explorados — o mesmo dilema exploração-vs-aproveitamento do `epsilon`-greedy do Projeto 17.1, só que resolvido de forma mais refinada, sem precisar de um `epsilon` fixo. `simular_partida_aleatoria` joga até o fim escolhendo ações aleatórias — é o "Monte Carlo" no nome: em vez de calcular exatamente o valor de cada posição (inviável em jogos maiores), estima por amostragem repetida.

`acoes_legais`, `aplicar_acao` e `simular_partida_aleatoria` são específicas do jogo (Tic-Tac-Toe: um tabuleiro 3×3, ações são posições vazias, o jogo termina em vitória/empate) — implemente-as você mesmo; são poucas linhas de lógica de tabuleiro, sem nada de RL nelas.

**Bônus — mini-AlphaZero**: substitua `simular_partida_aleatoria` por uma rede neural pequena (treinada via self-play, iterativamente) que estima diretamente o valor de uma posição, em vez de rodar uma partida aleatória até o fim — a mesma ideia central de AlphaZero, na escala de um jogo trivial.

---

### Projeto 17.7 — Decision Transformer

Você vai reproduzir a ideia central do Decision Transformer — RL como modelagem de sequência — reaproveitando quase diretamente a arquitetura `MiniLlama` do Projeto 8.3.

**Pré-requisitos**: os mesmos do Projeto 8.3, mais um dataset offline de trajetórias (colete rodando uma política aleatória ou o PPO do Projeto 17.3 em CartPole por algumas centenas de episódios, salvando `(estado, ação, recompensa)` de cada passo).

**1. Prepare os dados como sequências de `(retorno-a-ir, estado, ação)`**:

```python
def calcular_retornos_a_ir(recompensas, gamma=0.99):
    retornos = [0] * len(recompensas)
    acumulado = 0
    for i in reversed(range(len(recompensas))):
        acumulado = recompensas[i] + gamma * acumulado
        retornos[i] = acumulado
    return retornos

# para cada trajetória coletada: (retornos_a_ir[i], estados[i], acoes[i]) vira um "token" de 3 partes na sequência
```

`retorno-a-ir` (return-to-go) num passo `i` é a soma de todas as recompensas *daquele ponto em diante* na trajetória real — é isso que o modelo aprende a condicionar: "dado que eu quero atingir este retorno total, e estou neste estado, qual ação eu deveria ter tomado?".

**2. Adapte o `MiniLlama` do Projeto 8.3** para essa entrada estruturada em vez de tokens de texto — a mudança principal é a camada de embedding inicial, que agora projeta vetores de estado/ação/retorno (números contínuos) em vez de fazer lookup de tokens discretos:

```python
class DecisionTransformerEmbedding(nn.Module):
    def __init__(self, dim_estado, n_acoes, dim_modelo):
        super().__init__()
        self.embed_retorno = nn.Linear(1, dim_modelo)
        self.embed_estado = nn.Linear(dim_estado, dim_modelo)
        self.embed_acao = nn.Embedding(n_acoes, dim_modelo)

    def forward(self, retornos, estados, acoes):
        # intercala retorno, estado, ação como 3 "tokens" por passo de tempo — a mesma sequência
        # que depois passa pelos LlamaStyleBlock do Projeto 8.3, sem nenhuma mudança neles
        return self.embed_retorno(retornos) + self.embed_estado(estados) + self.embed_acao(acoes)
```

O restante do modelo (`LlamaStyleBlock`, `RMSNorm`, `GQACausalSelfAttention`) é reaproveitado sem alteração — a máscara causal continua fazendo sentido aqui (a ação no passo `t` não deveria "ver" o futuro), e a cabeça de saída final prevê a próxima ação em vez do próximo token de vocabulário.

**3. Compare com PPO no mesmo ambiente**: depois de treinado, gere uma trajetória condicionando no maior retorno observado no dataset de treino (`retorno-a-ir` inicial = o melhor retorno visto) e veja se o Decision Transformer consegue reproduzir ou superar esse desempenho — compare a recompensa média obtida contra o PPO do Projeto 17.3 treinado do zero no mesmo ambiente, e note a diferença fundamental: PPO explora ativamente o ambiente durante o treino; Decision Transformer nunca interage com o ambiente durante o treino, só aprende do dataset fixo (é offline RL, seção 17.9, assim como DPO).

---

## Erros comuns

- **Achar que policy = Q-function aprendida**: confunde value-based com policy-based.
- **Não normalizar advantages** em PPO/A2C — instabilidade.
- **Esquecer de detach value targets** — dupla retropropagação errada (o mesmo motivo do `with torch.no_grad()` no `treinar_passo` do Projeto 17.2).
- **Hiperparâmetros muito sensíveis** — RL é notoriamente difícil de tunar; Stable-Baselines3 e CleanRL têm boas defaults, comece delas.
- **Comparar runs com 1 seed** — variância gigante em RL; reporte ≥3 seeds.
- **Confundir RLHF/DPO com RL "puro"**. DPO é supervisionado (offline), RLHF é PPO (online) — a distinção da seção 17.9.

---

## Conexão com outros módulos

| Conceito daqui | Aparece em |
|---|---|
| PPO, GRPO | RLHF, reasoning RL (mod. [09](09_treinamento_e_alinhamento.mdx)) |
| Reward shaping | Fine-tuning para tasks específicas |
| MCTS + RL | Agentes com planejamento (mod. [13](13_agentes_tools_protocolos.md)) |
| Offline RL / DPO | Alinhamento (mod. [09](09_treinamento_e_alinhamento.mdx)) |
| Decision Transformer | Tópicos avançados (mod. [19](19_topicos_avancados.md)) |

---

## Checklist de saída

- [ ] Implementei Q-learning tabular do zero e validei contra Value Iteration (se não, revise o Projeto 17.1).
- [ ] Implementei DQN do zero, com replay buffer e target network (se não, revise o Projeto 17.2 e a seção 17.3).
- [ ] Treinei PPO em ambiente contínuo com sucesso (se não, revise o Projeto 17.3).
- [ ] Reproduzi RLHF completo (reward model + PPO) e comparei com DPO no mesmo dataset (se não, revise o Projeto 17.4).
- [ ] Sei explicar diferença entre on-policy, off-policy, offline RL (se não, revise as seções 17.2 e 17.9).
- [ ] Entendo as conexões matemáticas entre RLHF, DPO, PPO, GRPO com exemplos concretos, não só de nome (se não, revise a seção 17.8).
- [ ] Implementei MCTS do zero e entendo o trade-off exploração/aproveitamento em UCB1 (se não, revise o Projeto 17.6).
