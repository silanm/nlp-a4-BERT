import streamlit as st
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity


# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Load weights
model.load_state_dict(torch.load("sbert_finetuned.pth", map_location=torch.device("cpu")))
model.eval()


# Mean pooling to get sentence embeddings
def mean_pool(token_embeds, attention_mask):
    # reshape attention_mask to cover 768-dimension embeddings
    in_mask = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
    # perform mean-pooling but exclude padding tokens (specified by in_mask)
    pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(in_mask.sum(1), min=1e-9)
    return pool


# Calculate similarity between two sentences
def calculate_similarity(model, tokenizer, sentence_a, sentence_b):
    # Tokenize and convert sentences to input IDs and attention masks
    inputs_a = tokenizer(sentence_a, return_tensors="pt", truncation=True, padding=True)
    inputs_b = tokenizer(sentence_b, return_tensors="pt", truncation=True, padding=True)

    # Move input IDs and attention masks to the active device
    inputs_ids_a = inputs_a["input_ids"]
    attention_a = inputs_a["attention_mask"]
    inputs_ids_b = inputs_b["input_ids"]
    attention_b = inputs_b["attention_mask"]

    # Extract token embeddings from BERT
    u = model(inputs_ids_a, attention_mask=attention_a)[0]  # all token embeddings A = batch_size, seq_len, hidden_dim
    v = model(inputs_ids_b, attention_mask=attention_b)[0]  # all token embeddings B = batch_size, seq_len, hidden_dim

    # Get the mean-pooled vectors
    u = mean_pool(u, attention_a).detach().cpu().numpy().reshape(-1)  # batch_size, hidden_dim
    v = mean_pool(v, attention_b).detach().cpu().numpy().reshape(-1)  # batch_size, hidden_dim

    # Calculate cosine similarity
    similarity_score = cosine_similarity(u.reshape(1, -1), v.reshape(1, -1))[0, 0]

    return similarity_score


# Streamlit app
st.title("🔎 Sentence Similarity")
st.caption("This app finds the similarity between two sentences using Sentence-BERT embeddings.")

# Input text boxes
text1 = st.text_area("Enter first sentence:")
text2 = st.text_area("Enter second sentence:")

# Button to find similarity
if st.button("Find Similarity"):
    if text1 and text2:
        similarity_score = calculate_similarity(model, tokenizer, text1, text2)
        st.write(f"Cosine Similarity Score: {similarity_score:.4f}")
        st.write(f"Label: {'Entailment' if similarity_score > 0.5 else ('Neutral' if similarity_score == 0.5 else 'Contradiction')}")
    else:
        st.write("Please enter both sentences to find similarity.")
