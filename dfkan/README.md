# dfkan

**DualFlex KAN** – A Kolmogorov-Arnold Network library with flexible polynomial activations, multiple basis functions, explainability tools, and smart initialization strategies.

## Installation

```bash
pip install dfkan
```

Or from source:

```bash
git clone https://github.com/your-username/dfkan.git
cd dfkan
pip install -e .
```

## Quick start

```python
from dfkan import DualFlexKAN, smart_kan_initialization, KANExplainability

model = DualFlexKAN(...)
smart_kan_initialization(model)
```

## Main components

- **Models**: `DualFlexKAN`, `DualFlexKANLinear`, `SelectiveAttentiveKAN`
- **Activations**: `PolynomialActivation`, `LegendreActivation`, `ChebyshevActivation`, `BSplineActivation`, `RBFActivation`, and more
- **Initialization**: `KANInitializer`, `smart_kan_initialization`
- **Explainability**: `KANExplainability`, `KANInteractionAnalysis`, `quick_explain`
- **Save/Load**: `save_model_auto`, `load_model_auto`, `save_experiment`

## Requirements

Python ≥ 3.10, PyTorch, NumPy, SciPy, scikit-learn, Matplotlib, Seaborn, Dill.

## License

MIT
