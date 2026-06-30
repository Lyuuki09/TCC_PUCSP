# Sistema de Recomendação Híbrido para Lugares — TCC PUCSP

**Collaborative Filtering + Zero-Shot Learning com Embeddings BERT**

Uma abordagem acadêmica simples, reprodutível e eficaz para jovens e adultos.

**Autores:** Anderson, Juan, Leandro e Ygor

---

## Objetivo

Desenvolver um sistema de recomendação híbrido para sugestão de lugares (cafés, bares e espaços culturais) direcionado ao público jovem pós-faculdade (20–30 anos), com foco em socialização, networking e lazer.

O sistema combina duas abordagens complementares:

- **Collaborative Filtering (CF)** — explora padrões de preferência baseados em interações históricas de usuários.
- **Zero-Shot Learning com embeddings BERT** — permite recomendações de novos lugares sem histórico de avaliações (cold-start).

O **MovieLens 100K** foi utilizado como proxy, mapeando filmes para lugares e gêneros cinematográficos para atributos dos locais. Além disso, 31 locais reais da região PUC-SP Monte Alegre foram catalogados para demonstração prática.

---

## Etapas do Projeto

### Etapa 1 — Introdução e Justificativa Técnica

A escolha por uma arquitetura híbrida se justifica pelos seguintes motivos:

- **Collaborative Filtering** é excelente para capturar similaridades latentes entre usuários e itens, porém sofre com o problema de cold-start (novos usuários ou novos itens sem avaliações).
- **Zero-Shot Learning** via embeddings semânticos do BERT permite inferir similaridade entre descrições textuais sem necessidade de treinamento adicional, resolvendo o cold-start de forma eficiente.

A combinação das duas abordagens produz um sistema mais robusto, escalável e alinhado com o estado da arte em sistemas de recomendação (Ricci et al., 2022; Zhang et al., 2023).

Evitamos o uso de Graph Neural Networks (GNNs) por adicionarem complexidade computacional significativa sem ganhos proporcionais neste cenário de protótipo. Experimentos exploratórios com GNN estão na pasta `Testes_GNN/`.

### Etapa 2 — Configuração do Ambiente e Bibliotecas

| Biblioteca | Função |
|---|---|
| `scikit-surprise` | Algoritmos de Collaborative Filtering |
| `transformers` + BERTimbau | Embeddings semânticos em português |
| `scikit-learn` | Similaridade de cosseno |
| `pandas` / `numpy` | Manipulação de dados |
| `torch` | Inferência do modelo BERT |
| `matplotlib` | Visualização de métricas |
| `streamlit` | Interface web interativa |

O **BERTimbau** (`neuralmind/bert-base-portuguese-cased`) foi escolhido por capturar melhor as nuances de descrições em português, superando abordagens tradicionais como TF-IDF. O mean pooling sobre a última camada oculta produz vetores de 768 dimensões (Reimers & Gurevych, 2019).

### Etapa 3 — Preparação dos Dados

1. Download automático do MovieLens 100K
2. Carregamento de ratings e metadados dos filmes
3. Uso de **100% dos dados** (943 usuários, 1.682 itens)
4. Integração de 31 locais reais da região PUC-SP com descrições ricas para o BERT

### Etapa 4 — Collaborative Filtering (KNN Item-Based)

- Algoritmo: **KNN item-based** com similaridade de cosseno
- Parâmetros: k=50, split 80/20 para treino/teste
- Métrica: **RMSE** (~1.02 no conjunto de teste)

Optamos pelo KNN item-based por ser simples, interpretável e eficiente quando o número de itens é menor que o de usuários.

### Etapa 5 — Zero-Shot Learning com Embeddings BERT

1. Geração de embeddings para todos os itens (gêneros dos filmes como proxy de atributos)
2. Processamento em batch para eficiência
3. Recomendação por similaridade semântica entre descrições textuais

### Etapa 6 — Avaliação do Sistema

| Métrica | Componente | O que mede |
|---|---|---|
| **RMSE** | Collaborative Filtering | Precisão das predições numéricas |
| **Precision@5** | Zero-Shot (BERT) | Relevância das recomendações top-K |

### Etapa 7 — Demonstração Prática

Simulação de recomendação para o público-alvo: dado um perfil de preferência textual (ex.: *"bares descontraídos, música ao vivo"*), o sistema sugere os locais mais semanticamente similares entre candidatos.

Foi desenvolvida também uma **interface web interativa** (`app.py`) com Streamlit, permitindo que o usuário descreva sua *vibe*, opcionalmente informe um ID de usuário para ativar o KNN, e receba **um local recomendado** em tempo real.

### Etapa 8 — Fusão Híbrida e Interface Web

A estratégia de fusão ponderada combina os dois componentes:

```
score_final = (1 − α) · score_BERT + α · score_KNN
```

| α | Comportamento |
|---|---|
| `0.0` | 100% BERTimbau (texto / vibe) |
| `0.6` | Padrão — 40% texto + 60% histórico |
| `1.0` | 100% KNN (histórico colaborativo) |

O módulo `tcc_recommender/recommendation/hybrid.py` expõe as funções principais:

- `generate_embeddings()` — embeddings BERTimbau dos locais
- `get_bert_scores()` — similaridade textual (Zero-Shot)
- `get_knn_recommendations()` — perfil colaborativo via histórico MovieLens
- `hybrid_recommend()` — fusão híbrida com parâmetro α

### Etapa 9 — Arquitetura do Sistema

```mermaid
flowchart TD
    A[MovieLens 100K] --> B[Pré-processamento]
    B --> C[Collaborative Filtering<br/>KNN Item-Based]
    B --> D[Zero-Shot Learning<br/>BERTimbau]
    C --> E[RMSE]
    D --> F[Precision@K]
    E --> G[Sistema Híbrido]
    F --> G
    G --> H[Recomendações para jovens 20-30 anos]
    G --> I[Interface Streamlit<br/>app.py]
```

### Etapa 10 — Considerações Finais

O sistema demonstra viabilidade técnica para recomendação de lugares utilizando técnicas híbridas. O componente zero-shot se mostrou especialmente promissor para cenários com novos itens.

**Próximos passos sugeridos:**

- Fine-tuning com modelo multilíngue (BERTimbau ou mDeBERTa)
- Coleta de dados reais via API do Google Places ou Foursquare
- Avaliação online com usuários reais (A/B testing)

---

## Estrutura do Repositório

```
TCC_PUCSP/
├── app.py                      # Interface web Streamlit (recomendado para demo)
├── main.py                     # Executa o pipeline completo
├── TCC-PUC.ipynb               # Notebook de demonstração (usa os módulos)
├── requirements.txt
├── README.md
├── tcc_recommender/            # Código-fonte organizado
│   ├── config.py               # Hiperparâmetros e caminhos
│   ├── data/
│   │   ├── download.py         # Download MovieLens
│   │   ├── loader.py           # Carregamento dos dados
│   │   └── places_puc.py       # Locais reais PUC-SP
│   ├── models/
│   │   ├── collaborative.py    # KNN item-based (CF)
│   │   └── embeddings.py       # BERTimbau embedder
│   ├── evaluation/
│   │   ├── metrics.py          # Precision@K
│   │   └── ab_test.py          # Simulador A/B
│   ├── recommendation/
│   │   ├── hybrid.py           # Fusão híbrida (BERT + KNN)
│   │   └── zero_shot.py        # Recomendação por texto
│   └── visualization/
│       └── plots.py            # Gráficos de métricas
└── Testes_GNN/                 # Experimentos com GNN
```

---

## Como Executar

> **Importante:** sempre use o **ambiente virtual (`venv`)** do projeto. Rodar comandos no Python global causa erros como `ModuleNotFoundError: No module named 'surprise'`, pois o pacote `scikit-surprise` (importado como `surprise`) está instalado apenas no venv.

### 1. Ambiente virtual (obrigatório)

```powershell
cd C:\Users\Leco\Desktop\Estudo\TCC\TCC_PUCSP

# Criar o venv (apenas na primeira vez)
python -m venv venv

# Ativar o venv — o prompt deve exibir (venv)
.\venv\Scripts\Activate.ps1

# Instalar dependências
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Se o PowerShell bloquear a ativação do venv:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**Atalho sem ativar manualmente** (sempre usa o Python do venv):

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m streamlit run app.py
```

### 2. Interface web Streamlit (recomendado)

Com o venv ativo:

```powershell
streamlit run app.py
```

Ou, sem ativar o venv:

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

O app abrirá em **http://localhost:8501**.

**Fluxo de uso:**

1. Na sidebar, clique em **Retreinar Modelo / Carregar Dados** (primeira execução pode levar alguns minutos — download do MovieLens, carga do BERTimbau e treino do KNN).
2. Descreva sua *vibe* no campo principal (ex.: *"Quero um bar agitado com música ao vivo"*).
3. (Opcional) Informe um **ID de usuário** entre 1 e 943 para ativar o componente KNN.
4. Clique em **Buscar Locais** e veja o local recomendado.
5. (Opcional) Em **Configurações avançadas** na sidebar, ajuste o peso **α** da fusão híbrida.

### 3. Pipeline completo (Python)

```powershell
python main.py
```

### 4. Notebook interativo

```powershell
jupyter notebook TCC-PUC.ipynb
```

O dataset MovieLens 100K é baixado automaticamente na primeira execução.

### 5. Testes com GNN

```powershell
pip install -r Testes_GNN/requirements.txt
```

---

## Solução de Problemas

### `ModuleNotFoundError: No module named 'surprise'`

Esse erro ocorre quando o Streamlit (ou outro comando) roda **fora do venv**. O pacote correto é **`scikit-surprise`**, importado no código como `surprise`.

**Como corrigir:**

```powershell
# 1. Ative o venv
.\venv\Scripts\Activate.ps1

# 2. Reinstale as dependências no ambiente correto
pip install "numpy<2" scikit-surprise -r requirements.txt

# 3. Suba o app usando o Python do venv
.\venv\Scripts\python.exe -m streamlit run app.py
```

**Como verificar se está no ambiente certo:**

```powershell
# Deve apontar para ...\TCC_PUCSP\venv\Scripts\python.exe
where python

# Deve imprimir a versão sem erro
python -c "import surprise; print('scikit-surprise OK')"
```

### Conflito com NumPy 2.x

O `scikit-surprise` não é compatível com NumPy 2.x. O `requirements.txt` já fixa `numpy<2`. Se houver erro relacionado ao NumPy:

```powershell
pip install "numpy<2" --force-reinstall
pip install scikit-surprise --force-reinstall
```

---

## Dependências

- `pandas`, `numpy<2`
- `scikit-surprise` (importado como `surprise` — Collaborative Filtering)
- `torch`, `transformers`
- `scikit-learn`, `matplotlib`, `jupyter`
- `streamlit` (interface web)

> **Nota:** `scikit-surprise` pode falhar com `numpy` 2.x. Sempre instale via `pip install -r requirements.txt` **dentro do venv** para garantir versões compatíveis.
