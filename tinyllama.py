import torch
import torch.nn as nn

class TinyLLama(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, num_heads, embed_dim * 4, dropout=0.1),
            num_layers
        )
        self.output = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        x = self.transformer(x)
        x = self.output(x)
        return x

def generate_test_data(vocab_size, seq_length, batch_size):
    """Generate random input data for testing the model."""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), dtype=torch.long)
    return input_ids

def train_model(model, train_data, epochs, lr):
    """Train the Tiny LLaMA model for one epoch."""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for batch in train_data:
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output.view(-1, output.size(-1)), batch.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} loss: {total_loss/len(train_data)}")

# Example usage
vocab_size = 10000
embed_dim = 128
num_layers = 2
num_heads = 4
seq_length = 128
batch_size = 32

model = TinyLLama(vocab_size, embed_dim, num_layers, num_heads)
train_data = [generate_test_data(vocab_size, seq_length, batch_size) for _ in range(100)]

train_model(model, train_data, epochs=1, lr=1e-3)

