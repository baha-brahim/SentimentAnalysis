# Amazon Product Reviews Sentiment Analysis: Classical ML vs Deep Learning

A comprehensive comparison of Logistic Regression and DistilBERT for binary sentiment classification on Amazon product reviews.

## Project Overview
This project compares Logistic Regression (TF-IDF) and DistilBERT for classifying Amazon product reviews as positive or negative. It explores whether deep learning provides meaningful gains for simple sentiment tasks.

## Key Findings
- DistilBERT accuracy: 87.46% | Logistic Regression accuracy: 87.12% (just 0.34% difference)
- Classical ML remains competitive for binary sentiment classification
- DistilBERT is faster and smaller than BERT-base, with nearly the same performance
- Model choice depends on interpretability, resources, and use-case complexity

## Technical Details
- Dataset: Amazon Product Reviews (568,454 reviews)
- Preprocessing: spaCy (cleaning, lemmatization)
- Models:
  - Logistic Regression + TF-IDF (50K samples)
  - DistilBERT transformer (160K samples)
- Metrics: Accuracy, ROC-AUC, Precision, Recall, F1, Confusion Matrix
- Frameworks: scikit-learn, HuggingFace Transformers, spaCy

## Installation & Usage
```sh
git clone https://github.com/baha-brahim/SentimentAnalysis/
cd SentimentAnalysis
pip install -r requirements.txt
Run the notebook in your preferred environment
```
## Project Structure
- Data loading & clean-up
- Feature engineering
- Model 1: Logistic Regression (TF-IDF)
- Model 2: DistilBERT fine-tuning
- Comparative evaluation & insights

## Example Results

| Model                | Accuracy | ROC-AUC | Training Time | Model Size |
|----------------------|----------|---------|---------------|------------|
| Logistic Regression  | 87.12%   | 0.94    | ~2 min        | <1 MB      |
| DistilBERT           | 87.46%   | 0.945   | ~3.3 hours    | 268 MB     |

## When to Use Which Model?
- **Logistic Regression:** Fast, interpretable, ideal for binary/simple tasks or constrained environments
- **DistilBERT:** For nuanced/contextual tasks, multi-class, or when resources allow

## Author
Baha Eddine Ibrahim ([GitHub](https://github.com/baha-brahim))

## Acknowledgments
- Dataset: Amazon Product Reviews (Kaggle)
- BERT models: HuggingFace Transformers
- ML: scikit-learn
