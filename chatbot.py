import json
import os
import numpy as np
from RNN import S2S_RNN


class DeepCorpusNeuralChatbot:

  def __init__(self, data_path="data/Validation.jsonl"):
    self.chars = list(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -?.!\"'"
    )
    self.tokens = ["<start>", "<end>"]
    self.vocab = self.tokens + self.chars
    self.c2i = {ch: i for i, ch in enumerate(self.vocab)}
    self.i2c = {i: ch for i, ch in enumerate(self.vocab)}
    self.v_size = len(self.vocab)
    self.h_size = 256
    self.net = S2S_RNN(
        inp_r=self.v_size, s_len=self.h_size, vocab_size=self.v_size
    )

    # 1. Load data from Validation.jsonl
    jsonl_data = self.load_jsonl(data_path)

    # 2. Generate 300+ custom conversational pairs (slang, u/r mapping, tech, casual)
    custom_corpus = self.generate_massive_corpus()

    # 3. Combine both datasets
    self.dataset = jsonl_data + custom_corpus
    print(
        f"Total combined dataset size: {len(self.dataset)} conversation pairs."
    )

  def load_jsonl(self, path):
    dataset = []
    if not os.path.exists(path):
      print(f"Notice: '{path}' not found. Proceeding with custom corpus only.")
      return dataset

    print(f"Loading external entries from '{path}'...")
    with open(path, "r", encoding="utf-8") as f:
      for line in f:
        line = line.strip()
        if not line:
          continue
        try:
          item = json.loads(line)
          q = (
              item.get("query")
              or item.get("prompt")
              or item.get("input")
              or item.get("question")
          )
          a = (
              item.get("response")
              or item.get("completion")
              or item.get("output")
              or item.get("answer")
          )
          if q and a:
            dataset.append((str(q).lower().strip(), str(a).lower().strip()))
        except Exception:
          pass
    print(f"Successfully loaded {len(dataset)} pairs from Validation.jsonl.")
    return dataset

  def generate_massive_corpus(self):
    """Generates 300+ rich conversation pairs featuring natural slang (u, r),

    technical AI definitions, greetings, and varied sentence structures.
    """
    base_pairs = [
        ("hi", "hi, how are you today?"),
        ("hello", "hello there! how can i help u?"),
        ("hey", "hey! what is up?"),
        ("how r u?", "i am doing great, thx for asking!"),
        ("how are u", "i am doing awesome, how about u?"),
        ("what r u doing?", "just chatting with u and learning sequence patterns."),
        ("who are you?", "i am an rnn chatbot built from scratch in numpy."),
        ("what is your name?", "my name is tiny rnn."),
        ("who made u?", "i was coded by u in pure python!"),
        ("are u human?", "no, i am an artificial intelligence model."),
        ("what is machine learning?", "it is a field of artificial intelligence."),
        ("what is an rnn?", "it is a recurrent neural network for sequences."),
        ("what is ai?", "ai stands for artificial intelligence."),
        (
            "what is deep learning?",
            "it is a subset of machine learning using neural nets.",
        ),
        (
            "what is numpy?",
            "numpy is a powerful library for matrix math in python.",
        ),
        ("what is code?", "code is instructions written for computers."),
        (
            "what is a neural network?",
            "it is a network inspired by biological brains.",
        ),
        (
            "what is training?",
            "the process of optimizing weights using loss gradients.",
        ),
        (
            "what is loss?",
            "a measure of how wrong the model predictions are.",
        ),
        (
            "what is gradient descent?",
            "an optimization algorithm to minimize loss.",
        ),
        (
            "what is an epoch?",
            "one complete pass through the entire training dataset.",
        ),
        ("what is a weight?", "a parameter adjusted during training."),
        ("what is a bias?", "an offset value added to network pre-activations."),
        (
            "what is softmax?",
            "a function turning raw scores into probability distributions.",
        ),
        (
            "what is overfitting?",
            "when a model memorizes training data instead of generalizing.",
        ),
        (
            "what is regularization?",
            "a technique to prevent overfitting by penalizing large weights.",
        ),
        (
            "tell me a joke",
            (
                "why did the neural network cross the road? to get to the other"
                " dataset!"
            ),
        ),
        (
            "another joke",
            (
                "why do programmers prefer dark mode? because light attracts"
                " bugs!"
            ),
        ),
        ("bye", "goodbye! see u later."),
        ("thanks", "you are very welcome!"),
        ("thx", "np! let me know if u need more help."),
    ]

    expanded = []
    prefixes = ["hey ", "yo ", "please ", "can u tell me ", "tell me "]
    suffixes = ["?", "!", ".", " please."]

    # Systematic expansion to create a robust multi-hundred pattern distribution
    for q, a in base_pairs:
      expanded.append((q, a))
      for p in prefixes:
        expanded.append((f"{p}{q}", a))
      for s in suffixes:
        cleaned_q = q.rstrip("?!.")
        expanded.append((f"{cleaned_q}{s}", a))

    return expanded

  def vectorize(self, text):
    t_list = []
    i = 0
    while i < len(text):
      if text.startswith("<start>", i):
        t_list.append("<start>")
        i += 7
      elif text.startswith("<end>", i):
        t_list.append("<end>")
        i += 5
      else:
        ch = text[i]
        t_list.append(ch if ch in self.c2i else " ")
        i += 1
    seq = []
    for t in t_list:
      oh = np.zeros((self.v_size, 1))
      oh[self.c2i[t]] = 1
      seq.append(oh)
    return seq

  def train(self, epochs=800, lr=0.005):
    if not self.dataset:
      print("Error: No training data available.")
      return

    X, Y = [], []
    for q, a in self.dataset:
      full = f"<start>{q} -> {a}<end>"
      s = self.vectorize(full)
      if len(s) > 1:
        X.append(s[:-1])
        Y.append(s[1:])

    print(
        f"\nStarting deep training on {len(X)} conversational pairs for"
        f" {epochs} epochs...\n"
    )

    l2 = 2e-4
    clip = 3.0

    for ep in range(1, epochs + 1):
      # Extended decay schedule suited for 800 epochs
      cur_lr = lr * (0.95 ** (ep // 30))
      loss_sum = 0

      combined = list(zip(X, Y))
      np.random.shuffle(combined)
      shuffled_X, shuffled_Y = zip(*combined)

      for x_seq, y_seq in zip(shuffled_X, shuffled_Y):
        derivs, loss = self.net.backward_pass(
            X=x_seq, Y=y_seq, for_pass=True, loss="cross entropy"
        )

        for d in derivs:
          np.clip(d, -clip, clip, out=d)

        self.net.Wo -= cur_lr * (derivs[0] + l2 * self.net.Wo)
        self.net.bo -= cur_lr * derivs[1]
        self.net.Wss -= cur_lr * (derivs[2] + l2 * self.net.Wss)
        self.net.Wsx -= cur_lr * (derivs[3] + l2 * self.net.Wsx)
        self.net.bs -= cur_lr * derivs[4]

        loss_sum += loss

      avg_loss = loss_sum / len(X)
      if ep == 1 or ep % 50 == 0 or ep == epochs:
        print(
            f"Epoch {ep:3d}/{epochs} | LR: {cur_lr:.5f} | Loss:"
            f" {avg_loss:.4f}"
        )

    print("\nDeep training and weight convergence complete!\n")

  def sanitize_input(self, raw_text):
    return "".join(
        [c for c in raw_text.lower().strip() if c in self.c2i or c == " "]
    )

  def generate(self, prompt, max_len=60, top_k=3, temp=0.2):
    clean_prompt = self.sanitize_input(prompt)
    if not clean_prompt:
      return "please enter a valid prompt."

    p_text = f"<start>{clean_prompt} -> "
    p_seq = self.vectorize(p_text)

    s0 = np.zeros((self.h_size, 1))
    self.net.forward_pass(X=p_seq, s0=s0)
    cur_h = self.net.states[-1]
    cur_in = p_seq[-1]

    out_chars = []

    for _ in range(max_len):
      preds = self.net.forward_pass(X=[cur_in], s0=cur_h)
      cur_h = self.net.states[-1]

      probs = preds[0].ravel()
      probs = np.clip(probs, 1e-15, 1.0)

      log_p = np.log(probs) / temp
      exp_p = np.exp(log_p - np.max(log_p))
      probs = exp_p / np.sum(exp_p)

      top_indices = np.argsort(probs)[-top_k:]
      f_probs = np.zeros_like(probs)
      f_probs[top_indices] = probs[top_indices]

      p_sum = np.sum(f_probs)
      if p_sum > 0:
        f_probs /= p_sum
      else:
        f_probs[top_indices[-1]] = 1.0

      idx = np.random.choice(self.v_size, p=f_probs)
      token = self.i2c[idx]

      if token == "<end>":
        break
      if token == "<start>":
        continue

      out_chars.append(token)

      curr_str = "".join(out_chars)
      if "->" in curr_str:
        out_chars = []
        continue

      nxt = np.zeros((self.v_size, 1))
      nxt[idx] = 1
      cur_in = nxt

    return "".join(out_chars).strip() or "i am thinking..."


if __name__ == "__main__":
  # Points directly to your path: data/Validation.jsonl
  bot = DeepCorpusNeuralChatbot(data_path="data/Validation.jsonl")
  bot.train(epochs=800, lr=0.005)

  print("--- Deep Corpus RNN Chatbot Online ---")
  while True:
    try:
      user_in = input("You: ")
      if user_in.lower() == "quit":
        break
      if not user_in.strip():
        continue
      print(f"RNN: {bot.generate(user_in)}\n")
    except KeyboardInterrupt:
      break