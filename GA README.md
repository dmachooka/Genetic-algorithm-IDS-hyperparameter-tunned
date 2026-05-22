The project focuses on:

Privacy-preserving deep learning using DP-SGD

Hyperparameter optimization using:

Genetic Algorithm (GA)

Random / Keras-Tuner search

Privacy–utility tradeoff optimization

Formal privacy accounting (ε, δ)

Evaluation on IoT traffic datasets

This work targets research in:

IoT Security

Cyber-Physical Systems (CPS)

Privacy-Preserving Machine Learning

Industrial IDS
Research Objective

We optimize:

𝜃
∗
=
arg
⁡
max
⁡
𝜃
∈
Θ
(
𝐹
1
‾
(
𝜃
)
−
𝜀
(
𝜃
)
)
θ
∗
=arg
θ∈Θ
max
	​

(
F1
(θ)−ε(θ))

Where:

𝐹
1
‾
F1
 = cross-validated detection performance

𝜀
ε = privacy budget

𝜃
θ = network hyperparameters

The system balances intrusion detection accuracy against privacy leakage.
.
├── notebook_1_ga_dp_mlp.ipynb        # GA-optimized DP-MLP
├── notebook_2_training_eval.ipynb    # Final DP training + evaluation
├── notebook_3_random_search.ipynb    # Random / Keras-Tuner DP search
├── data/
│   └── data.csv
├── figures/
├── requirements.txt
└── README.md
Data Processing

Remove timestamp column

Standardize numerical features

Stratified train / validation / test split

Binary classification: attack vs normal
Differential Privacy Training

We use:

tensorflow_privacy.DPKerasSGDOptimizer

DP Mechanism

For each minibatch:

Compute per-example gradients

Clip gradients:

𝑔
~
𝑖
=
𝑔
𝑖
⋅
min
⁡
(
1
,
𝐶
∥
𝑔
𝑖
∥
2
)
g
~
	​

i
	​

=g
i
	​

⋅min(1,
∥g
i
	​

∥
2
	​

C
	​

)

Add Gaussian noise:

𝑁
(
0
,
𝜎
2
𝐶
2
)
N(0,σ
2
C
2
)

Update model parameters

Privacy guarantee:

(
𝜀
,
𝛿
)
-DP
(ε,δ)-DP
🧠 Model Architecture (MLP)

Final architecture:

Input layer: feature dimension

4–8 hidden layers (ReLU)

Dropout regularization

Output: Sigmoid (binary classification)

No CNN or LSTM — pure MLP architecture.

⚙️ Hyperparameter Search Space
ACTIVATIONS = ['gelu', 'swish', 'silu', 'relu']
OPTIMIZERS = ['sgd', 'adam']
LOSSES = ['binary_crossentropy']
LEARNING_RATES = [0.001, 0.0001]
BATCH_SIZES = [64, 128, 256]
EPOCHS = [20, 30, 50, 100]

LAYERS = [4, 6, 8]
UNITS = [64, 128, 256]

noise_multiplier = 1.3

🧬 Genetic Algorithm Optimization
Fitness Function
𝐹
𝑖
𝑡
𝑛
𝑒
𝑠
𝑠
=
𝐹
1
‾
−
𝜀
Fitness=
F1
−ε
GA Parameters

Population size: 10

Generations: 5

Tournament selection

Two-point crossover

Random mutation

Advantages:

Handles non-convex search

Balances privacy and performance

Robust in imbalanced IoT attack datasets

🎲 Random / Keras-Tuner Search

Alternative optimization:

Uniform sampling from search space

Objective:

𝐽
(
𝜃
)
=
𝐹
1
‾
−
𝜀
J(θ)=
F1
−ε

Faster but less globally optimal than GA.

📊 Evaluation Metrics

Accuracy

Precision

Recall

F1-score

Confusion Matrix

Precision–Recall Curve

Privacy budget (ε)

🔐 Privacy Accounting RDP

We compute privacy loss using:

compute_dp_sgd_privacy_statement(...)


Typical ε range in experiments:

1.5 ≤ ε ≤ 4.5


Interpretation:

ε < 1 → Strong privacy, lower recall

ε ≈ 2–4 → Practical IDS deployment

ε > 8 → Weak privacy guarantee

📈 Expected Outcomes in IoT IDS

Under DP constraints:

Slight drop in recall

Strong resistance to:

Membership inference attacks

Model inversion attacks

Traffic fingerprint reconstruction

Recommended deployment configuration:

Parameter	Recommended
Layers	4–6
Units	128–256
Noise Multiplier	1.0–1.5
ε	≤ 4
🖥️ Installation
Requirements

Python 3.9+

TensorFlow 2.14

tensorflow-privacy 0.9.0

scikit-learn

deap

pandas

matplotlib

Install
pip install -r requirements.txt


Or manually:

pip install tensorflow==2.14.0
pip install tensorflow-privacy==0.9.0
pip install deap scikit-learn pandas matplotlib seaborn

🚀 Running the Notebooks

Place dataset inside /data/

Update file path in notebook

Run:

notebook_1_ga_dp_mlp.ipynb


or

notebook_3_random_search.ipynb

🧪 Reproducibility

To ensure consistent results:

import numpy as np
import tensorflow as tf
import random

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)


Stratified splitting is used for class balance preservation.

🧮 Computational Complexity

Let:

N = number of samples

L = number of layers

U = units per layer

Training complexity per epoch:

𝑂
(
𝑁
⋅
𝐿
⋅
𝑈
2
)
O(N⋅L⋅U
2
)

GA complexity:

𝑂
(
𝐺
⋅
𝑃
⋅
𝐶
𝑉
⋅
𝑇
𝑟
𝑎
𝑖
𝑛
𝑖
𝑛
𝑔
𝐶
𝑜
𝑠
𝑡
)
O(G⋅P⋅CV⋅TrainingCost)

Where:

G = generations

P = population size

CV = cross-validation folds

📚 Target Journals

Suitable for:

IEEE Transactions on Industrial Informatics

IEEE Internet of Things Journal

Cyber-Physical Systems (Taylor & Francis)

Journal of Network and Computer Applications

Computers & Security

📌 Key Contribution

This repository demonstrates:

Practical deployment of DP in IoT IDS

Privacy–utility optimization via evolutionary search

Formal ε accounting

End-to-end CPS security pipeline

📬 Citation

If you use this code, please cite:

@article{dp_mlp_iot_ids,
  title={Multi-objective Differentially Private MLP for IoT Intrusion Detection in Cyber-Physical Systems},
  author={Daniel Machooka},
  journal={Under Review},
  year={2026}
}

🏁 Bottom Line

This project provides a deployable, privacy-aware intrusion detection system for IoT and CPS environments that:

Protects sensitive device logs

Preserves regulatory compliance

Maintains competitive detection performance

Resists modern privacy attacks
