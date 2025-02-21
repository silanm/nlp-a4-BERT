# Custom Sentence Transformer for NLI

## 🎯 Objective

* Create a custom-trained sentence transformer model to predict Natural Language Inference (NLI).

## 🚀 Demo

(Coming soon... a cool GIF will go here. Promise! 🤞)

## 📚 Datasets

* Pre-training Dataset: yahoo_answers_topics
    * Why this dataset? It's large, diverse, and covers multiple topics, making it perfect for sentence-level understanding.
    * Details:
        * Total size: 1.4 million rows, but we kept it light by using only 10% (140k rows).
        * Used two columns: best_answer for text. topic for labels (10 classes).

* Fine-tuning Datasets: SNLI and MNLI
    * Why these datasets? Both are gold standards for NLI tasks, perfect for fine-tuning.
    * Details: Train set: 100k, Validation set: 10k, Test set: 10k

## 📝 Steps

1. Pre-train: Train a sentence transformer using yahoo_answers_topics.

2. Fine-tune: Use SNLI and MNLI datasets with saved pre-trained weights.

3. Evaluate: Generate a classification report.

4. Inference: Find the cosine similarity between two sentences.

5. Interface: Build a Streamlit app for user interaction.

## 📦 Result from Pre-training

* Saved Weights: XX MB

## 🎛️ Result from Fine-tuning

* Weights: XX MB
* Classification Report: (Coming soon... just like that GIF 🙃)
* Average Cosine Similarity is ___

### 10 Sample Sentence Pairs:

🤔 Why Cosine Similarity from 0 to 1 (Not -1)?

In real-world sentence embeddings, negative values are rare because the embeddings represent semantic spaces. Most vectors align positively, making the range from 0 (no similarity) to 1 (identical meaning) more practical and interpretable.

💡 "Because -1 isn't just negative, it's emotionally negative. We like to keep things positive!" 😉

