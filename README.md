# Neural Network for Digit Classification (MNIST) using PyTorch

This project implements a feedforward neural network using PyTorch to classify handwritten digits from the MNIST dataset. This project is intermediate in my study of neural networks and PyTorch in particular and was created to compare with [my past project](https://github.com/ash1ra/Numpy-FFNN-Classification-MNIST), an implementation of a similar neural network on Numpy.  

I tried to write the code as quickly as possible to be able to move on to further study, and therefore did not implement part of the functionality, as it will be identical to the previous project.
## Model Architecture
The neural network is a two-layer feedforward network:
- **Input Layer**: 784 neurons (28x28 pixel images flattened);
- **Hidden Layer**: 10 neurons;
- **Output Layer**: 10 neurons (one for each digit, 0-9);
- **Loss Function**: Cross-entropy loss;
- **Output Activation**: Softmax (embedded in PyTorch's cross-entropy loss realization);
- **Hidden Layer Activation**: ReLU;
- **Optimizer**: Adam.

## Setting Up and Running the Project
### Using pip
1. Clone the repository:
```bash
git clone https://github.com/ash1rawtf/pytorch-ffnn-classification-mnist.git
cd pytorch-ffnn-classification-mnist
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Unix or MacOS
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the model:  
Create and customize the desired model in the `main.py` file and then run it.  
```bash
python main.py
```

### Using uv
1. Clone the repository:
```bash
git clone https://github.com/ash1rawtf/pytorch-ffnn-classification-mnist.git
cd pytorch-ffnn-classification-mnist
```

2. Create and activate a virtual environment:
```bash
uv venv
# On Windows
.venv\Scripts\activate
# On Unix or MacOS
source .venv/bin/activate
```

3. Install dependencies:
```bash
uv sync
```

4. Run the model:
Create and customize the desired model in the `main.py` file and then run it.  
```bash
python main.py
```
