import numpy as np
from RNN import S2S_RNN


class Chatbot:

  def __init__(self):
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

    self.dataset = self.get_training_data()
    print(f"loaded {len(self.dataset)} pairs into memeory.")

  def get_training_data(self):
    base_pairs = [
        ("hi", "hi there! how can i help u today?"),
        ("hello", "hello! nice to chat with u."),
        ("how r u?", "i am doing great, thx for asking!"),
        ("who are you?", "i am an rnn chatbot built from scratch in numpy."),
        ("what is your name?", "my name is tiny rnn."),
        ("what is machine learning?", "it is a field of artificial intelligence."),
        ("what is an rnn?", "it is a recurrent neural network for sequences."),
        ("what is numpy?", "numpy is a powerful library for matrix math in python."),
        ("tell me a joke","why did the neural network cross the road? to get to the other dataset!"),
        ("bye", "goodbye! see u later."),
        ("thanks", "you are very welcome!"),
        ("can u help me", "of course! what do u need help with?"),
    ]

    expanded = []
    for q, a in base_pairs:
      expanded.append((q, a))
      expanded.append((f"hey {q}", a))

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

  def train(self, epochs=400, lr=0.005):
    X, Y = [], []
    for q, a in self.dataset:
      full = f"<start>{q} -> {a}<end>"
      s = self.vectorize(full)
      if len(s) > 1:
        X.append(s[:-1])
        Y.append(s[1:])

    print(f"\nstarting traning for {epochs} epoches...")

    l2_lambda = 2e-4
    clip_val = 3.0

    for ep in range(1, epochs + 1):
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
          np.clip(d, -clip_val, clip_val, out=d)

        self.net.Wo -= cur_lr * (derivs[0] + l2_lambda * self.net.Wo)
        self.net.bo -= cur_lr * derivs[1]
        self.net.Wss -= cur_lr * (derivs[2] + l2_lambda * self.net.Wss)
        self.net.Wsx -= cur_lr * (derivs[3] + l2_lambda * self.net.Wsx)
        self.net.bs -= cur_lr * derivs[4]

        loss_sum += loss

      avg_loss = loss_sum / len(X)
      if ep == 1 or ep % 50 == 0 or ep == epochs:
        print(
            f"epoch {ep:3d}/{epochs} | lr: {cur_lr:.5f} | loss:"
            f" {avg_loss:.4f}"
        )

    print("\ntraning complet!\n")

  def clean_input(self, text):
    return "".join(
        [c for c in text.lower().strip() if c in self.c2i or c == " "]
    )

  def validate_response(self, text):
    if not text or len(text.strip()) < 2:
      return False
    if any(text.count(char) > len(text) * 0.35 for char in set(text)):
      return False
    return True

  def generate(self, prompt, max_len=60, top_k=3, temp=0.2):
    clean_prompt = self.clean_input(prompt)
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

    response = "".join(out_chars).strip()

    if not self.validate_response(response):
      return "i am still learning that pattern, try asking something else!"

    return response


if __name__ == "__main__":
  bot = Chatbot()
  bot.train(epochs=400, lr=0.005)

  print("--- rnn chatbot online ---")
  while True:
    try:
      user_input = input("you: ")
      if user_input.lower() in ["quit", "exit"]:
        break
      if not user_input.strip():
        continue
      print(f"bot: {bot.generate(user_input)}\n")
    except KeyboardInterrupt:
      break