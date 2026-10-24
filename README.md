# s2s-rnn

a sequence-to-sequence recurrent neural network chatbot built completely from scratch using pure python and numpy. no heavy deep learning frameworks like pytorch or tensorflowjust raw matrix math, hidden states, and backpropagation through time.

available as a python package on [pypi](https://pypi.org/project/s2s-rnn/0.1.0/).

## features

- **pure numpy architecture:** custom `S2S_RNN` class handling forward and backward passes.
- **regularization & optimization:** built-in L2 weight decay, learning rate decay schedules, and gradient clipping to prevent exploding gradients.
- **anti-gibberish filters:** output validation logic to catch repetition loops and ensure clean responses.
- **character-level sequence processing:** tokenizes strings into one-hot vectors across a defined vocabulary.

## installation

you can install the core rnn module directly via pip:

```bash
pip install s2s-rnn==0.1.0
```

or install requirements locally for running the chatbot script:

```bash
pip install numpy
```

## project structure

```text
 RNN.py        # contains the custom S2S_RNN matrix logic and backprop
 chatbot.py    # main script with training loop, dataset, and inference generator
 README.md     # project documentation
```

## core API usage (`S2S_RNN`)

the core of the library is the `S2S_RNN` class, which handles sequence processing, hidden states, forward passes, and backward passes.

### initialization parameters

when creating an instance of `S2S_RNN`, pass three key dimensions:
- `inp_r`: input dimension (typically your vocabulary size or one-hot vector size).
- `s_len`: hidden layer size (e.g., `256` or `384` units for memory capacity).
- `vocab_size`: output dimension (size of your output vocabulary or character set).

```python
import numpy as np
from RNN import S2S_RNN

# initialize the network
v_size = 50  # vocabulary size
h_size = 256 # hidden state / brain size

net = S2S_RNN(inp_r=v_size, s_len=h_size, vocab_size=v_size)
```

### forward pass

to run a forward pass over a sequence of one-hot vectors:

```python
# create a dummy input sequence of length 10 (each vector shaped [v_size, 1])
X_seq = [np.zeros((v_size, 1)) for _ in range(10)]
X_seq[0][5] = 1  # example active token

# initial hidden  state (zeros vector of shape [h_size, 1])
s0 = np.zeros((h_size, 1))

# run forward pass
predictions = net.forward_pass(X=X_seq, s0=s0)

# access hidden states stored during the pass
final_hidden_state = net.states[-1]
```

### backward pass & training

to compute gradients via backpropagation through time (bptt) using cross-entropy loss:

```python
# target sequence of one-hot vectors
Y_seq = [np.zeros((v_size, 1)) for _ in range(10)]
Y_seq[0][6] = 1  # example target token

# compute gradients and loss
derivs, loss = net.backward_pass(X=X_seq, Y=Y_seq, for_pass=True, loss="cross entropy")

# derivs contains gradients for [Wo, bo, Wss, Wsx, bs] ready for weight updates
```

## running the chatbot

1. clone the repository or place `RNN.py` and `chatbot.py` in the same directory.
2. start training and launch the interactive shell:

```bash
python chatbot.py
```

the model will train on the embedded conversational dataset, optimize its weights, and open an interactive prompt. type `quit` or `exit` to stop.