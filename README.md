# Crossbar Parasitic Models Simulator
## Overview
The Crossbar Parasitics Simulator is a comprehensive tool for analyzing parasitic effects in crossbar arrays. It consists of two main components:
1. **Model Comparison**: Benchmarks different parasitic resistance models from literature
2. **Hardware-Aware Training**: Evaluates trained neural networks under realistic crossbar conditions

## Features
### Model Comparison
- **Multiple Models Support**:
  - JeongModel, IdealModel, DMR, Gamma, CrossSim, LTSpice, NgSpice, Memtorch
  - Configurable reference model (default: NgSpice)

- **Parametric Analysis**:
  - Array size (default: 32x32)
  - Parasitic resistance values
  - Memory window ratios (HRS/LRS)
  - Sparsity (% of HRS devices in the crossbar)
  - Device variability and non-linearities

- **Performance Metrics**:
  - Execution time benchmarking
  - Current and voltage relative errors
  - Robustness metrics

### Hardware-Aware Training
- **Neural Network Features**:
  - Two-layer fully connected architecture
  - MNIST dataset compatibility
  - Configurable hidden layer size

- **Hardware Modeling**:
  - Linear conductance mapping for weights
  - Configurable quantization schemes
  - Parasitic resistance incorporation
  - Array size partitioning for large networks
  - Adjustable resistance ratios (HRS/LRS)
  - Positive-only weight constraints option

- **Analysis Tools**:
  - Real-time accuracy monitoring
  - Multi-model comparison plots
  - Weight/bias distribution heatmaps
  - Confusion matrix visualization
  - CSV logging of performance metrics
  - 3D surface plots for parameter sweeps

---

## **Getting Started**

1. **Clone the repository**
  ```bash
  git clone https://github.com/alambertini01/Crossbar_Models_Comparison
  cd Crossbar_Models_Comparison
  ```

2. **Set up a virtual environment (optional):**

  For Windows:

  
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```
  For macOS/Linux:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```


3. **Install the package:**

```bash
pip install -e .
```

---

### Usage
1. **Model Comparison**
```bash
python Crossbar_Simulator.py
```

2. **Neural Network Training**
```bash
python NN_Training.py
```

3. **Trained network Evaluation**
```bash
python NN_testing.py
```

---

## Citation
```bibtex
@software{crossbar_simulator,
  author = {Lambertini, Alessandro},
  title = {Crossbar Parasitics Simulator},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/alambertini01/Crossbar_Models_Comparison}
}
```
