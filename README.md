# **Crossbar Parasitics Simulator**

## **Overview**
The Crossbar Parasitics Simulator benchmarks and compares various models of parasitic resistances found in the literature. It evaluates each model in terms of:
- **Execution times**  
- **Relative error** (compared to a reference model, configurable by the user)

The simulator is modular, allowing activation or deactivation of models and customization of key parameters for flexibility.

---

## **Features**
- **Customizable Models**:  
  Includes a variety of models, such as:
  - **JeongModel**
  - **IdealModel**
  - **DMR (multiple versions)**
  - **Gamma (multiple versions)**
  - **CrossSim (multiple versions)**
  - **LTSpice**
  - **NgSpice (linear and nonlinear)**
  - **Memtorch (Python and C++ implementations)**


  Choose any model as the reference for comparisons. Default: `NgSpice`.
- **Parameters**:
  - Crossbar array dimensions (default: `32x32`)
  - Parasitic resistance
  - Memory window
  - Device variability.
  - Device non-linearities.
 
    
  The simulator allows the parameters to be range of values and performs a simulation sweep.
- **Multiple Visualizations**:
  - Simulation times
  - Relative and absolute output currents
  - Voltage drops Heatmaps
  - Accuracy trends (e.g., vs. parasitic resistance, memory window)
  - Scatter plots

---

## **Getting Started**
### **1. Requirements**
- Python 3.x
- Dependencies: Install via `requirements.txt`
  ```bash
  pip install -r requirements.txt
