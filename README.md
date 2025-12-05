# Recomendação Zero-Shot

Resumo do projeto e passos rápidos para reproduzir o notebook principal `recomendacao_zeroshot_ex.ipynb`.

**Objetivo**
- Implementar um sistema híbrido de recomendação que combina Collaborative Filtering (via `scikit-surprise`) e Zero-Shot Learning (via embeddings BERT) para sugerir lugares (simulados com MovieLens 100k).

**Conteúdo**
- Notebook principal: `recomendacao_zeroshot_ex.ipynb` — código e explicações passo a passo.
- Exemplo de dependências: `Testes_GNN/requirements.txt`.

## Estrutura do repositório
- `recomendacao_zeroshot_ex.ipynb` — notebook com todo o pipeline (download MovieLens, preprocessamento, CF, embeddings BERT, avaliação e visualização de grafo).
- `Testes_GNN/requirements.txt` — arquivo de dependências existente (referência).

## Dependências principais
O notebook usa (diretamente ou indiretamente) bibliotecas comuns de Data Science e NLP. Principais pacotes:

- `pandas`
- `numpy` (atenção a compatibilidades com `scikit-surprise`)
- `scikit-surprise` (importa no notebook como `surprise`)
- `torch`
- `transformers`
- `scikit-learn`
- `networkx`
- `matplotlib`

Veja `Testes_GNN/requirements.txt` para uma lista completa com versões encontradas no ambiente original.

## Passo a passo para reproduzir (PowerShell — Windows)

1) Criar e ativar um ambiente virtual (recomendado):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

2) Instalar dependências básicas (exemplo mínimo):

```powershell
pip install pandas numpy scikit-surprise torch transformers scikit-learn networkx matplotlib
```

Observação: instale `scikit-surprise` usando `pip install scikit-surprise`. Se encontrar problemas com compilação, prefira usar uma distribuição binária compatível (ou conda).

3) Rodar o notebook (opções):

- Interativo: abrir Jupyter e executar as células.
  ```powershell
  jupyter notebook recomendacao_zeroshot_ex.ipynb
  ```

- Executar todas as células via nbconvert (não interativo):
  ```powershell
  jupyter nbconvert --to notebook --execute recomendacao_zeroshot_ex.ipynb --output executed.ipynb --ExecutePreprocessor.timeout=600
  ```

4) Gerar um `requirements2.txt` reproducível (opções):

- Método rápido (lista enxuta a partir dos imports):
  - Converter notebook para script e usar `pipreqs`:
    ```powershell
    jupyter nbconvert --to script recomendacao_zeroshot_ex.ipynb --output tmp_nb_script.py
    pip install pipreqs
    pipreqs . --force --savepath requirements2.txt
    ```

- Método exato (captura versões do ambiente onde o notebook foi executado):
  ```powershell
  python -m pip freeze > requirements2.txt
  ```

5) Testar que os imports funcionam:

```powershell
python -c "import pandas, numpy, surprise, torch, transformers, sklearn, networkx, matplotlib; print('OK')"
```

## Notas específicas do notebook
- O notebook baixa o dataset MovieLens 100k, faz preprocessamento de ratings e filmes, treina um KNN item-based via `surprise` e extrai embeddings textuais via `transformers` (BERT) para recomendações zero-shot.
- Há operações de visualização de grafo usando `networkx` e `matplotlib`.
- Atenção ao `numpy` se usar `scikit-surprise`: versões muito recentes de `numpy` (2.x) podem causar problemas; prefira `numpy` da série 1.26.x se necessário.

## Problemas comuns e soluções rápidas
- Erro ao rodar `pip freeze` (launcher quebrado): use `python -m pip freeze > requirements2.txt` ou `py -m pip freeze > requirements2.txt`.
- Se `scikit-surprise` falhar na instalação via pip, tente instalar com conda ou procurar wheels pré-compilados.

## Próximos passos sugeridos
- Gerar `requirements2.txt` específico para este notebook (posso gerar automaticamente a partir do notebook e salvar como `requirements2.txt`).
- Opcional: criar um `environment.yml` para conda com dependências binárias mais simples de instalar (`scikit-surprise`, `pytorch`/`cpu` ou `cuda`).

## Créditos
- Autor: código e notebook presentes em `recomendacao_zeroshot_ex.ipynb`.
