Evolutionary Optimization of Differentially Private LSTM-Dense Architectures for IoT Intrusion Detection

This repository provides the official implementation for the paper: "Evolutionary Hyperparameter Optimization for Differentially Private Deep Learning in IoT Security". It utilizes a Genetic Algorithm (GA) to navigate the complex trade-off between model utility (F1-Score) and privacy leakage (Epsilon) within an LSTM-Dense neural network framework.

## 1. Methodology Overview
The core of this research is a dual-objective optimization problem. We utilize DP-SGD to provide formal privacy guarantees and the DEAP (Distributed Evolutionary Algorithms in Python) framework to automate the selection of privacy-sensitive hyperparameters.

### Experimental Pipeline:Data Curating: Focuses on high-impact IoT attack vectors: DDoS-ICMP_Flood, DDoS-UDP_Flood, and DDoS-TCP_Flood.
Evolutionary Search: A Genetic Algorithm evolves individuals (hyperparameter sets) including layer depth, unit counts, activation functions, and L2-norm clipping bounds.

Privacy-Aware Fitness Function: The fitness of an individual is defined as:$$Fitness = \text{Mean CV F1-Score} - \epsilon$$where $\epsilon$ is the privacy budget consumed, calculated via the Moments Accountant.

DP-SGD Integration: Implementation of DPKerasSGDOptimizer to ensure that the contribution of individual training examples is hidden.

## 2. Repository StructurePlaintext├── data/                   # Dataset directory (IoT_Intrusion.csv)(can change to specific datasets
├── src/
│   ├── preprocessing.py    # Scaling, Label Encoding, and Reshaping
│   ├── ga_optimizer.py     # DEAP setup and custom mutation/crossover
│   └── dp_model.py         # DP-LSTM architecture and TF-Privacy integration
├── results/                # Confusion matrices and Loss curves
├── requirements.txt        # Exact versioning for reproducibility
└── main.ipynb              # Complete end-to-end execution script

## 3. Installation & Environment Setup To ensure reproducibility, specific versions of tensorflow and tensorflow-privacy are required.

Using mismatched versions may result in incompatible GradientTape operations.Bash# Recommended Python 3.10-
pip install tensorflow==2.14.0---ensure same
pip install tensorflow-privacy==0.8.0---- ensure is same
pip install deap imblearn pandas scikit-learn matplotlib seaborn

# unistall any previuos version

## 4. Data PreparationThe model expects a CSV format following the IoT_Intrusion schema.
Target: label (encoded to 0, 1, 2).Input Scaling: StandardScaler is applied to categorical/numerical features.
LSTM Reshaping: Input vectors are reshaped to $(Samples, Timesteps=1, Features)$ to accommodate the temporal requirements of the LSTM layers.

## 5. Genetic Algorithm Configuration
The search space is defined as follows:LSTM Layers: 1 to 3 layers (Units: 16 to 256)(can vary as needed ).
Dense Layers: 1 to 4 layers (Units: 16 to 256).

Activations: gelu, swish, silu, relu.
DP Parameters: L2-norm clip [0.1, 5.0],
Fixed Noise Multiplier: 1.3.
Evolutionary Params: Population: 10, 
            Generations: 5, CXPB: 0.5, MUTPB: 0.2.

## 6. Results and Privacy Accounting Upon completion, the script outputs a Privacy Statement derived from the Moments Accountant. 
MetricGA-Optimized DP Model Accuracy[]F1-Score (Macro)
[Insert %]Epsilon ($\epsilon$)Calculated based on Steps/Batch Size
Delta ($\delta$)$10^{-5}$### Visualization  
The script generates:Dual Loss Curves: Comparing standard vs. 
privacy-preserving training trajectories.
Confusion Matrix: Evaluated on the held-out test set (approx. 40,000 samples).

## 7. Reproducibility Checklist
To replicate the results presented in the manuscript:Set gene_alg = True in the configuration block.
Ensure noise_multiplier remains constant at 1.3 to validate the $\epsilon$ comparisons.
The StratifiedKFold (n=6) ensures the results are robust against class imbalance in the IoT subsets.
