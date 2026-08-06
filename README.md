# ML Learning 

Motive - hands on of all ML algos 

---

## Topics Covered

| Folder | Topics |
|---|---|
| `1-linear-regression/` | NumPy basics, linear regression (single & multi-feature), feature scaling, polynomial regression, scikit-learn |
| `2-logistic-regression/` | Classification, sigmoid function, decision boundary, logistic loss, gradient descent, regularisation |

---

## Requirements

| Dependency | Version | Notes |
|---|---|---|
| **Python** | `3.10.x` ||
| **numpy** | `>=1.24, <3.0` | |
| **matplotlib** | `>=3.7, <4.0` |
| **scipy** | `>=1.10, <2.0` | |
| **scikit-learn** | `>=1.2, <2.0` | |
| **ipykernel** | `>=6.0` | Jupyter kernel |
| **ipywidgets** | `>=8.0` | Required by interactive plot utilities |
| **ipympl** | `>=0.9` | Required for `%matplotlib widget` (interactive plots) |

---

## Setup

```bash
# 1. Create and activate a Python 3.10 virtual environment
python3.10 -m venv .venv310
source .venv310/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Register the kernel with Jupyter
python -m ipykernel install --user --name=venv310 --display-name "Python 3.10 (ml)"
```

---
