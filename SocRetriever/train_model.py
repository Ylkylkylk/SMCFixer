from sentence_transformers import SentenceTransformer, InputExample, losses, util, evaluation, models
from torch.utils.data import DataLoader, Dataset
import os
import torch
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, average_precision_score
import re
import torch.nn as nn
import torch.nn.functional as F


# Data cleaning function
def clean_query(query):
    query = re.sub(r'Error: ', '', query)
    query = re.sub(r'\n', ' ', query)
    query = re.sub(r'--> .*?:\d+:\d+:', '', query)
    return query.strip()


# Function to extract code snippets
def extract_code_snippet(text):
    code_snippet = re.findall(r'<code>(.*?)</code>', text)
    return ' '.join(code_snippet) if code_snippet else text


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.per_head_dim = hidden_dim // num_heads

        self.linear_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(3)
        ])
        self.final_linear = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()

        query, key, value = [l(x).view(batch_size, seq_length, self.num_heads, self.per_head_dim).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (x, x, x))]

        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.per_head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, value).transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_length, self.hidden_dim)
        attention_output = self.final_linear(context)

        return attention_output


# Custom Attention Layer
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.attention = MultiHeadSelfAttention(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, 256)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: [batch_size, seq_len, hidden_dim]
        x = self.attention(x)
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)

        attn_weights = torch.softmax(x, dim=1)
        attn_output = torch.bmm(x.transpose(1, 2), attn_weights).squeeze(2)
        return attn_output

    def save(self, output_path):
        torch.save(self.state_dict(), os.path.join(output_path, 'self_attention.pt'))

    def load(self, input_path):
        self.load_state_dict(torch.load(os.path.join(input_path, 'self_attention.pt')))


# Data set class definition
class RetrievalDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        query = clean_query(entry['query'])
        answer = extract_code_snippet(entry['answer'])
        return InputExample(texts=[query, answer], label=float(entry['label']))


# Custom collate function
def custom_collate(batch):
    texts = [example.texts for example in batch]
    labels = torch.tensor([example.label for example in batch])
    return texts, labels


# Load data function
def load_data(data_path, test_size=0.1):
    with open(data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    train_data, val_data = train_test_split(data, test_size=test_size, random_state=42)
    return RetrievalDataset(train_data), RetrievalDataset(val_data)


# Custom evaluator class
class CustomEvaluator(evaluation.SentenceEvaluator):
    def __init__(self, dataloader: DataLoader, name: str = '', write_csv: bool = True):
        self.dataloader = dataloader
        self.name = name
        self.write_csv = write_csv

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        all_preds, all_labels = [], []
        for texts, labels in self.dataloader:
            labels = labels.to(torch.float32)
            embeddings = model.encode([text[0] for text in texts])
            similarities = util.pytorch_cos_sim(embeddings, embeddings)
            preds = torch.argmax(similarities, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        mar = average_precision_score(all_labels, all_preds)
        mrr = sum(1.0 / (rank + 1) for rank in range(len(all_preds)) if all_labels[rank] == 1) / len(all_preds)

        print(f"Accuracy: {acc:.4f}, MAR: {mar:.4f}, MRR: {mrr:.4f}")
        return acc  # You can choose to return acc, mar, or mrr depending on your preference


os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained model
model_path = './all-MiniLM-L6-v2'

# Create model components
word_embedding_model = models.Transformer(model_path)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
attention_model = SelfAttention(hidden_dim=word_embedding_model.get_word_embedding_dimension())


class SentenceTransformerWithAttention(SentenceTransformer):
    def __init__(self, model_path, attention_model, device):
        word_embedding_model = models.Transformer(model_path)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        super(SentenceTransformerWithAttention, self).__init__(modules=[word_embedding_model, pooling_model])
        self.attention_model = attention_model.to(device)
        self.pooling_model = pooling_model
        self._device = device

    def forward(self, features):
        for k in features:
            if isinstance(features[k], torch.Tensor):
                features[k] = features[k].to(self._device)

        output = self._first_module()(features)
        token_embeddings = output['token_embeddings'].to(self._device)
        pooled_output = self.pooling_model(output)
        attention_output = self.attention_model(token_embeddings)
        combined_output = attention_output + pooled_output['sentence_embedding']
        return {'sentence_embedding': combined_output}

    def encode(self, sentences, batch_size=8, **kwargs):
        features = self.tokenize(sentences)
        features = self.forward(features)
        return features['sentence_embedding']

    def save(self, path):
        # Save parent class modules
        super(SentenceTransformerWithAttention, self).save(path)
        # Save custom attention model
        self.attention_model.save(path)

    def load(self, path):
        # Load parent class modules
        super(SentenceTransformerWithAttention, self).load(path)
        # Load custom attention model
        self.attention_model.load(path)


# Instantiate custom model
model = SentenceTransformerWithAttention(model_path, attention_model, device)
model = model.to(device)

# Prepare training dataset
train_dataset, val_dataset = load_data('./all_train_v1.json')
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate)

# Define loss function
train_loss = losses.CosineSimilarityLoss(model=model)

# Define custom evaluator
evaluator = CustomEvaluator(val_loader, name='sts-eval')

# Start fine-tuning
model.fit(
    train_objectives=[(train_loader, train_loss)],
    evaluator=evaluator,
    epochs=3,
    warmup_steps=100,
    evaluation_steps=1000,
    output_path='./SocR_model'
)
