# Sistema de Recomendação Híbrido — TCC PUCSP

Sistema híbrido de recomendação que combina **Collaborative Filtering** (`scikit-surprise`) e **Zero-Shot Learning** (embeddings BERT) para sugerir lugares, simulados com o dataset MovieLens 100K.

**Autores:** Anderson, Juan, Leandro e Ygor

## Estrutura do repositório

```
TCC_PUCSP/
├── TCC-PUC.ipynb          # Notebook principal do TCC
├── requirements.txt       # Dependências do notebook principal
├── README.md
└── Testes_GNN/            # Experimentos com Graph Neural Networks
    ├── TCC_GNN.ipynb
    ├── teste1.ipynb
    ├── requirements.txt
    └── fontes/Links.txt
```

## Como reproduzir (Windows / PowerShell)

1. Criar e ativar um ambiente virtual:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

2. Instalar dependências:

```powershell
pip install -r requirements.txt
```

3. Executar o notebook principal:

```powershell
jupyter notebook TCC-PUC.ipynb
```

O notebook baixa automaticamente o dataset MovieLens 100K na primeira execução.

## Dependências principais

- `pandas`, `numpy`
- `scikit-surprise` (importado como `surprise`)
- `torch`, `transformers`
- `scikit-learn`, `matplotlib`

> **Nota:** `scikit-surprise` pode ter problemas com `numpy` 2.x. O `requirements.txt` já restringe `numpy<2`.

## Testes com GNN

A pasta `Testes_GNN/` contém experimentos exploratórios com Graph Neural Networks. Para rodá-los, use o `requirements.txt` específico dessa pasta:

```powershell
pip install -r Testes_GNN/requirements.txt
```
