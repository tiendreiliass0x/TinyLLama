import torch
from tinyllama import TinyLLama

# params
vocab_size = 10000
embed_dim = 128
num_layers = 2
num_heads = 4
seq_length = 128
batch_size = 32

# inference
model = TinyLLama(vocab_size, embed_dim, num_layers, num_heads)  # Re-create the model structure
model.load_state_dict(torch.load('tinyllama_model_parameters.pth', weights_only=True))
model.eval()  # Set the model to evaluation mode
print(model)