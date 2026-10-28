# GANs - Trabalho Avaliativo

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Implementação de uma **Conditional GAN (cGAN)** condicional treinada no dataset **Fashion-MNIST**, com uma interface Gradio para gerar peças de roupa "sob encomenda" a partir da categoria escolhida pelo usuário. O projeto segue a estrutura do [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/), com gerenciamento de dependências via `uv`.

## Como rodar

### 1. Ambiente

```bash
make requirements
# ou diretamente:
uv sync
```

Ative o ambiente virtual criado em `.venv/` (`source .venv/bin/activate`) ou use `uv run <comando>` na frente de cada passo abaixo. O código detecta GPU (CUDA) automaticamente e usa CPU como alternativa.

### 2. Preparar o dataset

```bash
uv run python -m gans.dataset
```

Baixa o Fashion-MNIST para `data/raw/` via `torchvision` e monta um subconjunto (default: 1.000 imagens por classe, configurável por quantidade e por lista de classes, com remapeamento de rótulos), normalizado para `[-1, 1]`.

### 3. Treinar a cGAN

```bash
uv run python -m gans.train
```

Parâmetros configuráveis via CLI (defaults seguem a especificação do trabalho):

| Parâmetro | Descrição | Default |
|---|---|---|
| `--epochs` | número de épocas de treino | 20 |
| `--batch-size` | tamanho do batch | 64 |
| `--lr` | taxa de aprendizado (Adam) | 2e-4 |
| `--latent-dim` | dimensão do vetor latente `z` | 100 |
| `--samples-per-class` | tamanho do subconjunto por classe | 1000 |

Durante o treino, a cada época:
- as perdas de G e D são registradas (CSV)
- uma grade de amostras é salva em `reports/figures/` usando ruído fixo, uma linha por classe
- checkpoints do gerador são salvos em `models/`

### 4. Avaliar

```bash
uv run python -m gans.evaluate
```

Gera o gráfico de perdas G/D ao longo das épocas, um GIF com a evolução das imagens geradas época a época, e (opcional) a métrica de um classificador auxiliar sobre as imagens geradas.

### 5. Interface web

```bash
uv run uvicorn gans.api:app --reload
```

Sobe uma API (FastAPI) que carrega o gerador treinado e expõe um endpoint de geração. O frontend estático em `frontend/` (estilizado com o design system VOID/03 — `frontend/assets/`) consome essa API: menu suspenso para a categoria (nomes em português), controle de quantidade de imagens, campo opcional de seed e botão "Gerar".

### Outros comandos úteis

```bash
make lint      # ruff check
make format    # ruff check --fix + ruff format
make test      # pytest tests/
```

## O que precisa ser feito

Checklist de implementação, por fase (acompanhar também as issues no GitHub):

- [ ] **Fase 1 — Dataset** (`gans/dataset.py`): Fashion-MNIST via `torchvision.datasets`, subconjunto configurável (N por classe, classes escolhidas, remapeamento de rótulos), normalização `[-1, 1]`
- [ ] **Fase 2 — Modelos** (`gans/models.py`): `Generator` (ruído + `nn.Embedding` do rótulo, `ConvTranspose2d`+`BatchNorm`+`ReLU`, saída 28x28 com `Tanh`) e `Discriminator` (imagem + rótulo, `Conv2d`+`LeakyReLU`, sem `BatchNorm` na primeira camada, saída de probabilidade)
- [ ] **Fase 3 — Treinamento** (`gans/train.py`): loop de treino D/G com `BCELoss`, Adam (`lr=2e-4`, `betas=(0.5, 0.999)`), batch 64, ~20 épocas; detecção automática de GPU; checkpoints e grades de amostras por época
- [ ] **Fase 4 — Avaliação** (`gans/evaluate.py`): curva de loss G/D, GIF da evolução das amostras, métrica opcional via classificador auxiliar
- [ ] **Fase 5 — Interface web** (`gans/api.py` + `frontend/`): API FastAPI que serve o gerador treinado; frontend estático estilizado com o design system VOID/03 (`frontend/assets/`), com seleção de categoria (PT-BR), quantidade, seed opcional, botão "Gerar", exibição ampliada
- [ ] **Fase 6 — Documentação e empacotamento**: dependências no `pyproject.toml` (fastapi, uvicorn, matplotlib, imageio/pillow), README com instruções de uso, testes básicos em `tests/`

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         y and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── y   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes y a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

