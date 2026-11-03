# Notebooks

Notebooks are for exploration and the LSTM. Keep reusable code in `src/pitwall/` and
import it, so the notebooks and the pipeline never disagree.

Suggested names:
- `01_eda.ipynb`: degradation by compound, circuit, temperature (Anand)
- `02_lstm.ipynb`: PyTorch LSTM on `pitwall.data.stint_sequences`, same `cv_fold` values

Start every notebook with:

```python
from pitwall.data import load_features
train = load_features("train")
```
