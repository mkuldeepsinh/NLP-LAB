# Assignment 4: N-gram Language Models with Smoothing

## Overview
This assignment builds n-gram language models (unigram, bigram, trigram, and quadrigram) for Gujarati text with advanced smoothing techniques. The models learn probability distributions over word sequences and can predict how likely a sentence is in the language.

## What This Does

### 1. N-gram Model Training
Processes 500,000 sentences from the Gujarati corpus to build:
- **Unigram Model**: Single word probabilities
- **Bigram Model**: Two-word sequence probabilities
- **Trigram Model**: Three-word sequence probabilities
- **Quadrigram Model**: Four-word sequence probabilities

### 2. Smoothing Techniques
Implements three smoothing methods to handle unseen n-grams:

#### a. Add-one (Laplace) Smoothing
- Formula: `P(w₂|w₁) = (Count(w₁,w₂) + 1) / (Count(w₁) + V)`
- Adds 1 to all counts to avoid zero probabilities

#### b. Add-k Smoothing
- Formula: `P(w₂|w₁) = (Count(w₁,w₂) + k) / (Count(w₁) + k×V)`
- Generalizes Add-one with adjustable k parameter

#### c. Token Type Smoothing
- Normalizes by number of unique following words
- Formula: `Count(w₁,w₂) / (Count(w₁) + N(w₁))`

### 3. Visualization
Generates probability distribution plots for:
- Top 20 unigrams
- Top 20 bigrams
- Top 20 trigrams

## Dataset
- **Source**: ai4bharat/IndicCorpV2 (Gujarati)
- **Size**: 500,000 sentences processed
- **Vocabulary**: 459,103 unique tokens

## Key Results

### Most Common N-grams
**Unigrams**: છે (is), અને (and), આ (this)  
**Bigrams**: છે કે (is that), છે અને (is and)  
**Trigrams**: જણાવ્યું હતું કે (said that)

### Probability Example
For the bigram "છે કે":
- Add-one: 0.0382
- Add-k (k=0.1): 0.0710
- Token Type: 0.0724

## Generated Files
- `unigram_probabilities.png` - Unigram probability distribution
- `bigram_probabilities.png` - Bigram probability distribution
- `trigram_probabilities.png` - Trigram probability distribution

## Technical Highlights
- Uses NLTK's ngrams utility for efficient n-gram generation
- Implements custom probability calculation functions
- Handles Gujarati Unicode properly with font detection
- Streaming dataset processing for memory efficiency

## Running the Code
1. Open `main.ipynb` in Jupyter/Colab
2. Install: `datasets`, `nltk`, `matplotlib`, `seaborn`
3. Run cells to train models and generate visualizations
4. Experiment with different smoothing parameters

## Applications
These n-gram models can be used for:
- Language generation
- Spell checking
- Text prediction/autocomplete
- Machine translation evaluation

