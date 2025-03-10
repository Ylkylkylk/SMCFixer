import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, InputExample, losses, util, evaluation, models
import os
import torch
import json
import re
import torch.nn as nn

json_file_path1 = './Knowledge_Base/Knowledge-Base-0.5_clean.json'
json_file_path2 = './Knowledge_Base/Knowledge-Base-0.6_clean.json'
json_file_path3 = './Knowledge_Base/Knowledge-Base-0.7_clean.json'
json_file_path4 = './Knowledge_Base/Knowledge-Base-0.8_clean.json'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
client = OpenAI(api_key='YOUR_API_KEY', base_url="BASE_URL")

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        attn_weights = self.attention(x)
        attn_weights = torch.softmax(attn_weights, dim=1)
        attn_output = torch.bmm(x.transpose(1, 2), attn_weights).squeeze(2)
        return attn_output

    def save(self, output_path):
        torch.save(self.state_dict(), os.path.join(output_path, 'self_attention.pt'))

    def load(self, input_path):
        self.load_state_dict(torch.load(os.path.join(input_path, 'self_attention.pt')))


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
        super(SentenceTransformerWithAttention, self).save(path)
        self.attention_model.save(path)

    def load(self, path):
        super(SentenceTransformerWithAttention, self).load(path)
        self.attention_model.load(path)

# Read JSON file and extract content
def load_content_from_json(*file_paths):
    content_list = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        content_list.extend(data)  # Directly extend the list with the data from the file
    return content_list

context_items = load_content_from_json(json_file_path1, json_file_path2, json_file_path3, json_file_path4)

def clean_query(query):
    query = re.sub(r'Error: ', '', query)
    query = re.sub(r'\n', ' ', query)
    query = re.sub(r'--> .*?:\d+:\d+:', '', query)
    return query.strip()

# Loading error messages and answers
def load_error_messages_and_answers(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        error_messages = [item['query'] for item in data]
        answers = [item['answer'] for item in data]
    return error_messages, answers
def extract_code_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)  # Load JSON data, assuming it's a list

    code_list = []

    for item in data:  # Iterate over each item in the list
        for change_key, change_value in item.items():  # Iterate over the keys like "Changes1", "Changes2"
            if "Uncompilable code" in change_value:
                for example_key, example_value in change_value["Uncompilable code"].items():
                    if "code" in example_value:
                        code_list.append(example_value["code"])

    return code_list

# Load embedding vector
def load_embeddings_from_json(file_path):
    with open(file_path, 'r') as file:
        embeddings_list = json.load(file)
    tensor_embeddings = [torch.tensor(embedding) for embedding in embeddings_list]
    return tensor_embeddings

context_embeddings = load_embeddings_from_json('./Knowledge_Base_embedding.json')

# Load pre-trained model
model_path = "./SocR_model"
word_embedding_model = models.Transformer(model_path)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
attention_model = SelfAttention(hidden_dim=word_embedding_model.get_word_embedding_dimension())

model = SentenceTransformerWithAttention(model_path, attention_model, device)
model = model.to(device)

# Retrieve the top-1 knowledge
def get_top1(error_message, context_embedding, error_code):
    context_embedding_matrix = np.vstack([emb.detach().numpy() for emb in context_embedding])
    query_text_new = clean_query(error_message)
    query_embedding = model.encode([query_text_new], convert_to_tensor=True)
    query_embedding = query_embedding.detach().cpu().numpy().reshape(1, -1)
    similarities = cosine_similarity(query_embedding, context_embedding_matrix).flatten()
    top_index = np.argmax(similarities)
    knowledge = context_items[top_index]
    result = f"""
[task description]
Please combine the following solidity knowledges, uncompilable code and error messages. Modify the uncompilable code so that it can be compiled correctly. Don't add comments in the code.

[solidity knowledges]
{knowledge}

[uncompile code]
{error_code}

[error messages]
{query_text_new}"""
    print(result)
    print("-----------------------------------------------------------------------------------------")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant who are good at solidity programming"},
            {"role": "user", "content": result},
        ],
        stream=False
    )
    text = response.choices[0].message.content
    print(text)
    return text

# Retrieve the top-3 knowledge
def get_top3(error_message, context_embedding, error_code):
    context_embedding_matrix = np.vstack([emb.detach().numpy() for emb in context_embedding])
    top_k = 3
    query_text_new = clean_query(error_message)
    query_embedding = model.encode([query_text_new], convert_to_tensor=True)
    query_embedding = query_embedding.detach().cpu().numpy().reshape(1, -1)
    similarities = cosine_similarity(query_embedding, context_embedding_matrix).flatten()
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    knowledge = " ".join([context_items[i] for i in top_indices])
    result = f"""
[task description]
Please combine the following solidity knowledges, uncompilable code and error messages. Modify the uncompilable code so that it can be compiled correctly. Don't add comments in the code.

[solidity knowledges]
{knowledge}

[uncompile code]
{error_code}

[error messages]
{query_text_new}"""
    print(result)
    print("-----------------------------------------------------------------------------------------")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant who are good at solidity programming"},
            {"role": "user", "content": result},
        ],
        stream=False
    )
    text = response.choices[0].message.content
    print(text)
    return text

# Retrieve the top-5 knowledge
def get_top5(error_message, context_embedding, error_code):
    context_embedding_matrix = np.vstack([emb.detach().numpy() for emb in context_embedding])
    top_k = 5
    query_text_new = clean_query(error_message)
    query_embedding = model.encode([query_text_new], convert_to_tensor=True)
    query_embedding = query_embedding.detach().cpu().numpy().reshape(1, -1)
    similarities = cosine_similarity(query_embedding, context_embedding_matrix).flatten()
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    knowledge = " ".join([context_items[i] for i in top_indices])
    result = f"""
[task description]
Please combine the following solidity knowledges, uncompilable code and error messages. Modify the uncompilable code so that it can be compiled correctly. Don't add comments in the code.

[solidity knowledges]
{knowledge}

[uncompile code]
{error_code}

[error messages]
{query_text_new}"""
    print(result)
    print("-----------------------------------------------------------------------------------------")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant who are good at solidity programming"},
            {"role": "user", "content": result},
        ],
        stream=False
    )
    text = response.choices[0].message.content
    print(text)
    return text