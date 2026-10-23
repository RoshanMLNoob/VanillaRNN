import numpy as np
from RNN import S2S_RNN

# 1. Define your vocabulary safely as single tokens
# We extract individual characters from standard strings, but keep special tokens whole.
base_chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ,?.!'\"")
special_tokens = ["<start>", "<end>"]

vocab = special_tokens + base_chars
char_to_idx = {ch: i for i, ch in enumerate(vocab)}
idx_to_char = {i: ch for i, ch in enumerate(vocab)}
vocab_size = len(vocab)

def text_to_sequence(text, add_start_end=True):
    """
    Converts a raw string into a sequence of one-hot vectors.
    Automatically wraps the text with <start> and <end> tokens if requested.
    """
    tokens = []
    if add_start_end:
        tokens.append("<start>")
    
    # Simple character-level tokenization for the rest of the text
    for char in text:
        if char in char_to_idx:
            tokens.append(char)
        else:
            tokens.append(" ") # Fallback for unknown characters
            
    if add_start_end:
        tokens.append("<end>")
        
    # Convert tokens to one-hot vector sequence (list of arrays)
    seq = []
    for token in tokens:
        idx = char_to_idx[token]
        one_hot = np.zeros((vocab_size, 1))
        one_hot[idx] = 1
        seq.append(one_hot)
        
    return seq

# 2. Initialize the RNN model
hidden_size = 64
rnn = S2S_RNN(inp_r=vocab_size, s_len=hidden_size, vocab_size=vocab_size)

def train_chatbot(dataset, epochs=100, lr_val=0.01):
    """
    Takes dataset as a list of (query_string, response_string) pairs,
    prepares the sequences with <start> and <end> tags, and trains the RNN.
    """
    print("\nPreparing training data...")
    X_train = []
    Y_train = []
    
    for query, response in dataset:
        # Encoder input: query wrapped with <start> and <end>
        x_seq = text_to_sequence(query, add_start_end=True)
        # Decoder target: response wrapped with <start> and <end>
        y_seq = text_to_sequence(response, add_start_end=True)
        
        X_train.append(x_seq)
        Y_train.append(y_seq)
        
    print(f"Training on {len(dataset)} conversation pairs...")
    
    # Custom learning rate lambda
    lr_func = lambda t: lr_val
    
    # Train using your class's built-in training loop
    rnn.Train_RNN(
        X=X_train, 
        Y=Y_train, 
        epoch=epochs, 
        lr=lr_func, 
        f1="relu", 
        f2="softmax", 
        loss="cross entropy", 
        one_hot=False, 
        stocastic=True, 
        interval=50
    )
    print("Training finished!")

def generate_response(prompt_text, max_len=60):
    """
    Encodes the prompt query and decodes an autoregressive response.
    """
    # Convert prompt string to sequence (with start/end tags)
    query_seq = text_to_sequence(prompt_text, add_start_end=True)
    
    # Phase 1: Encoding the Query
    rnn.forward_pass(X=query_seq, s0=None)
    final_hidden_state = rnn.states[-1]  # Capture context vector
    
    # Phase 2: Decoding the Response starting with <start>
    start_idx = char_to_idx["<start>"]
    current_input_char = np.zeros((vocab_size, 1))
    current_input_char[start_idx] = 1
    
    current_hidden = final_hidden_state
    generated_chars = []
    
    for _ in range(max_len):
        preds = rnn.forward_pass(X=[current_input_char], s0=current_hidden)
        current_hidden = rnn.states[-1]
        
        pred_probs = preds[0]
        predicted_idx = np.argmax(pred_probs)
        predicted_token = idx_to_char[predicted_idx]
        
        if predicted_token == "<end>":
            break
            
        generated_chars.append(predicted_token)
        
        # Feed predicted token back as next input
        next_input = np.zeros((vocab_size, 1))
        next_input[predicted_idx] = 1
        current_input_char = next_input
        
    return "".join(generated_chars)

# --- Main Execution Example ---
if __name__ == "__main__":
    # Define your data simply as pairs of x, y strings!
    Data = [
        ("hi", "hi, how are you?"),
        ("who are you?", "i am an rnn chatbot built from scratch."),
        ("what is your name?", "my name is tiny rnn.")
    ]
    
    # Train the model (uncomment the line below when ready to train)
    train_chatbot(Data, epochs=50, lr_val=0.01)
    
    # Or try loading a pre-saved model if you have one:
    # rnn.load_model("rnn_chatbot_model.pkl")
    
    print("\n--- Chatbot Ready ---")
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                break
            
            response = generate_response(user_input)
            print(f"RNN: {response}\n")
            
        except KeyboardInterrupt:
            break