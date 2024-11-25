import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from xformers.ops import memory_efficient_attention

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset generation: small sequences with 3 coordinates and point-wise continuous targets
def generate_random_data(num_events, num_targets):
    data = []
    for _ in range(num_events):
        seq_len = torch.randint(3, 10, (1,)).item()  # Random sequence length
        coords = torch.randn(seq_len, 3)  # Random 3D points
        targets = torch.randn(seq_len, num_targets)  # Continuous regression targets for each point
        data.append((coords, targets))
    return data

# Custom Dataset and DataLoader
class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Generate data
num_targets = 5  # Number of regression targets per point
num_events = 10  # Total number of events
data = generate_random_data(num_events=num_events, num_targets=num_targets)
dataset = RandomDataset(data)
dataloader = DataLoader(dataset, batch_size=1, collate_fn=lambda x: x, shuffle=True)

# xFormers-based Attention Layer
class XFormersAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(XFormersAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # Reshape for multi-head attention
        batch_size, seq_len, embed_dim = q.shape
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply memory-efficient attention
        attn_output = memory_efficient_attention(q, k, v, attn_bias=mask)

        # Reshape and project back to embedding dimension
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        return self.output_proj(attn_output)

# Transformer Encoder Layer with xFormers Attention
class XFormersEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super(XFormersEncoderLayer, self).__init__()
        self.attention = XFormersAttention(d_model, nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention block
        attn_output = self.attention(x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # Feedforward block
        ff_output = self.feedforward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        return x

# Transformer Model for Point-wise Regression
class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, dim_feedforward, num_layers, num_targets):
        super(SimpleTransformer, self).__init__()
        self.input_layer = nn.Linear(input_dim, d_model)
        self.encoder = nn.ModuleList(
            [XFormersEncoderLayer(d_model, nhead, dim_feedforward, dropout=0.1) for _ in range(num_layers)]
        )
        self.output_layer = nn.Linear(d_model, num_targets)

    def forward(self, x, padding_mask=None):
        x = self.input_layer(x)
        for layer in self.encoder:
            x = layer(x, mask=padding_mask)
        x = self.output_layer(x)  # Regression output per point
        return x

# Instantiate the model
model = SimpleTransformer(input_dim=3, d_model=8, nhead=1, dim_feedforward=16, num_layers=1, num_targets=num_targets).to(device)

# Loss and optimizer
criterion = nn.MSELoss()  # Mean Squared Error Loss for point-wise regression
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(2):  # Small number of epochs for demonstration
    for batch in dataloader:
        # Preprocess the batch
        inputs = [item[0] for item in batch]
        labels = [item[1] for item in batch]

        # Pad sequences to the maximum length in the batch
        lengths = [seq.size(0) for seq in inputs]
        max_len = max(lengths)
        padded_inputs = torch.zeros(len(inputs), max_len, 3).to(device)
        padded_labels = torch.zeros(len(inputs), max_len, num_targets).to(device)
        for i, (seq, target) in enumerate(zip(inputs, labels)):
            padded_inputs[i, :seq.size(0), :] = seq
            padded_labels[i, :target.size(0), :] = target

        # Create padding mask
        padding_mask = torch.zeros(len(inputs), max_len, dtype=torch.bool).to(device)
        for i, seq_len in enumerate(lengths):
            padding_mask[i, seq_len:] = True

        # Forward pass
        optimizer.zero_grad()
        outputs = model(padded_inputs, padding_mask=padding_mask)

        # Compute loss and backpropagation
        loss = criterion(outputs[~padding_mask], padded_labels[~padding_mask])
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

print("Training complete!")
