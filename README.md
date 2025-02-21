# Custom Sentence Transformer for NLI

## 🎯 Objective

* Create a custom-trained sentence transformer model to predict Natural Language Inference (NLI).

## 🚀 Demo


## 📝 Steps

1. Pre-train: Train a sentence transformer using yahoo_answers_topics.
2. Fine-tune: Use SNLI and MNLI datasets with saved pre-trained weights.
3. Evaluate: Generate a classification report.
4. Inference: Find the cosine similarity between two sentences.
5. Interface: Build a Streamlit app for user interaction.


## 📦 Pre-training

* Dataset: `yahoo_answers_topics`
   * [Hugging Face](https://huggingface.co/datasets/community-datasets/yahoo_answers_topics)
   * It's large, diverse, and covers multiple topics, making it perfect for sentence-level understanding.
   * Total size: `140k` rows out of `1.4m` rows
   * Used two columns: `best_answer` for text, and `topic` for labels (10 classes)
* Parameters
   *  Max padding length: `2000`; handle long samples
   *  Batch size: `2`; limited memory on GPU
   *  Number of epochs: `1000`

## 🎛️ Fine-tuning

* Datasets: `SNLI` and `MNLI`
    * Both are gold standards for NLI tasks, perfect for fine-tuning.
    * Train set: 100k, Validation set: 10k, Test set: 10k
* Training time: 
* Average Cosine Similarity: 
* Classification Report

   ```
                  precision    recall  f1-score   support
   
      entailment       0.34      0.35      0.35      3429
         neutral       0.32      0.49      0.39      3191
   contradiction       0.33      0.15      0.21      3380
   
        accuracy                           0.33     10000
       macro avg       0.33      0.33      0.31     10000
    weighted avg       0.33      0.33      0.31     10000
   ```

### 10 Sample Sentence Pairs:

> #### Why Cosine Similarity from 0 to 1 (Not -1)?
>
> In real-world sentence embeddings, negative values are rare because the embeddings represent semantic spaces. Most vectors align positively, making the range from 0 (no similarity) to 1 (identical meaning) more practical and interpretable.
