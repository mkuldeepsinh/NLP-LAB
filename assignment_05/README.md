# Assignment 5: Advanced Language Modeling with Good-Turing & Deleted Interpolation

## Overview
This assignment implements sophisticated smoothing techniques for n-gram language models. It compares Good-Turing smoothing with Deleted Interpolation to create more robust probability estimates for language modeling tasks.

## What This Does

### 1. Data Preparation
- Loads and preprocesses 1 million Gujarati sentences
- Splits data into:
  - **Training**: 500,000 sentences (50%)
  - **Validation**: 250,000 sentences (25%)
  - **Testing**: 250,000 sentences (25%)

### 2. Good-Turing Smoothing
Implements the classic Good-Turing algorithm for all n-gram orders (1-4).

#### How It Works
- Calculates frequency-of-frequencies (Nc values)
- Adjusts counts using: `C* = (C+1) × N(C+1) / N(C)`
- Handles unseen n-grams with probability mass from singletons
- Prevents zero probabilities for any sequence

### 3. Deleted Interpolation
Combines multiple n-gram orders with learned weights.

#### Model Structure
- Interpolates: Unigram + Bigram + Trigram + Quadrigram
- Formula: `P(w₄|w₁,w₂,w₃) = λ₁P(w₄) + λ₂P(w₄|w₃) + λ₃P(w₄|w₂,w₃) + λ₄P(w₄|w₁,w₂,w₃)`
- Uses EM algorithm to find optimal λ weights
- Validates on held-out data

### 4. Model Evaluation
Compares all models using average log-probability on test set.

## Key Results

### Frequency Table Example (Quadrigram)
```
C (MLE)  |  Nc   |  C*
---------|-------|--------
100      |  45   |  98.25
50       |  123  |  49.87
10       |  2456 |  10.12
1        |  45000|  0.87
```

### Lambda Parameters (Typical Values)
- λ₁ (unigram): ~0.05
- λ₂ (bigram): ~0.15
- λ₃ (trigram): ~0.30
- λ₄ (quadgram): ~0.50

## Technical Implementation

### Good-Turing Model Features
- Handles vocabulary padding with `<s>` and `</s>` tokens
- Pre-calculates smoothed counts for efficiency
- Deals with edge cases (zero frequencies)
- Returns log probabilities to prevent underflow

### Deleted Interpolation Features
- Expectation-Maximization for parameter learning
- 10 iterations with progress tracking
- Add-1 smoothing for unigram backoff
- Efficient count storage with dictionaries

## Files
- `main.py` - Complete implementation with all models
- Training output includes detailed probability tables

## Running the Code
```bash
python main.py
```

The script will:
1. Load and split the Gujarati dataset
2. Train all four Good-Turing models
3. Display frequency tables
4. Train Deleted Interpolation model
5. Evaluate all models on test data
6. Print comparative results

## Performance Metrics
Models are evaluated using:
- **Average Log Probability**: Higher is better
- Measures how well the model predicts test sentences
- Accounts for sentence length variations

## Applications
These advanced models are useful for:
- Speech recognition
- Machine translation
- Text generation
- Grammar checking
- Predictive typing

