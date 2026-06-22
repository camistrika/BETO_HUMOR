# BETO_HUMOR

Detección de humor en español con BETO (NBSVM / BETO congelado / BETO + LoRA).

## Instalación

```bash
pip install -r requirements.txt
pip install -e .
```

El `pip install -e .` instala el paquete `betohumor` en modo editable, así
los notebooks pueden hacer `from betohumor.config import ...` sin necesitar
agregar nada al `sys.path`.

## Estructura

```
BETO_HUMOR/
├── data/
│   ├── raw/            <- poné ahí haha_2019_train.csv
│   └── processed/
├── notebooks/
├── results/
│   ├── checkpoints/
│   └── figures/
├── src/
│   └── betohumor/       <- el paquete
├── setup.py
└── requirements.txt
```

## Configuración

No se usa YAML. Todos los hiperparámetros están en dataclasses con valores
por defecto, en `src/betohumor/config.py`. Para usar otros valores:

```python
from betohumor.config import DataConfig, BetoConfig, LoraConfig

data_config = DataConfig()                      # defaults
beto_config = BetoConfig(learning_rate=5e-4)     # override puntual
lora_config = LoraConfig(r=32, lora_alpha=64)
```

## Orden sugerido de notebooks/scripts

1. `notebooks/baseline_nbsvm.ipynb`
2. `notebooks/hyperparam_search_baseline.ipynb`
3. `notebooks/hyperparam_search_lora.ipynb`
4. `notebooks/cross_validation_baseline.ipynb`
5. `notebooks/cross_validation_lora.ipynb`
6. **Ablation de hashtags** (con la mejor config ya encontrada, sin nueva búsqueda):
   ```bash
   python3 -m betohumor.ablation_nbsvm_quick      # rápido, sin GPU
   python3 -m betohumor.ablation_baseline_quick    # liviano, corre local en CPU/MPS
   python3 -m betohumor.ablation_lora_quick        # requiere GPU (Colab)
   ```
   Antes de correrlos, revisar que las constantes `*_BEST_*` al principio de
   cada script tengan los valores reales del grid search / CV.

