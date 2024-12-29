import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :].requires_grad_(True).to(x.device)
        return self.dropout(x)



class TypeEmbedding(nn.Module):
    def __init__(self, embedding_dim, num_type):
        super(TypeEmbedding, self).__init__()
        self.num_type = num_type
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_type, embedding_dim)


    def forward(self, x):
        # x: [batch_size, seq_len]
        # x: [batch_size, seq_len, num_type] # one-hot encoding conversion
        # x: [batch_size, seq_len, embed_dim] # from one-hot encoding to embedding
        # U (Learnable embedding) Y (One-hot encoding)
        print("x size: ", x.size())  # [batch_size, seq_len] (3, 7)
        Y = F.one_hot(x, self.num_type)
        print(Y)
        print("Y size: ", Y.size())  # [batch_size, seq_len, num_type]
        event_embedding = self.embedding(Y)
        print("event embedding size: ", event_embedding.size())
        return event_embedding


# Sample index tensor (must be of integer type)
indices = torch.tensor([0, 2, 1, 3])

# Apply one-hot encoding
one_hot_encoded = F.one_hot(indices, num_classes=4)

print(one_hot_encoded)

te = TypeEmbedding(5, 4)

x = torch.tensor([[1, 2, 3, 0, 3, 2, 1], [0, 3, 2, 1, 3, 2, 1], [0, 3, 2, 1, 1, 2, 3]])
print(te(x))