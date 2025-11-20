# Assignment 8: Byte Pair Encoding (BPE) & Naive Bayes Classification

## Overview
This assignment covers two important NLP techniques: subword tokenization using Byte Pair Encoding (BPE) and text classification using Naive Bayes with custom features.

## Part 1: Byte Pair Encoding (BPE)

### What Is BPE?
BPE is a data compression algorithm adapted for NLP tokenization. It learns subword units by iteratively merging the most frequent character pairs.

### How It Works
1. Start with character-level vocabulary
2. Add end-of-word token `</w>` to each word
3. Count frequency of adjacent symbol pairs
4. Merge most frequent pair into single token
5. Repeat for N iterations (default: 20)

### Example Training
```
Initial: "t h e </w>" → "th e </w>" → "the</w>"
```

After 20 merges, common subwords emerge:
- "the</w>", "ing</w>", "cat</w>"
- Handles rare words by breaking into learned parts

### Tokenization Algorithm
Uses greedy longest-match:
1. Find longest subword in vocabulary that matches prefix
2. Add to token list
3. Move to next position
4. Repeat until word is fully tokenized

### Example Output
```
Input: "The cat is chasing the dog quietly."

Tokenized:
'the'      → ['the</w>']
'cat'      → ['cat</w>']
'is'       → ['i', 's</w>']
'chasing'  → ['c', 'h', 'a', 's', 'ing</w>']
'dog'      → ['do', 'g</w>']
'quietly'  → ['q', 'u', 'i', 'e', 't', 'l', 'y', '</w>']
```

## Part 2: Naive Bayes Text Classification

### Task
Classify messages into three categories:
- **Inform**: Informational messages
- **Promo**: Promotional/marketing messages
- **Reminder**: Reminder/notification messages

### Preprocessing Steps
1. **URL Detection**: Replace URLs with `URL` token
2. **Number Normalization**: Replace all numbers with `NUMBER`
3. **Punctuation**: Replace punctuation with `PUNCT` token
4. **Lowercase**: Convert to lowercase

### Features Used

#### Binary Features
- `has_URL`: Presence of URL
- `has_NUMBER`: Presence of numbers
- `has_PUNCT`: Presence of punctuation

#### Bigram Features
- All adjacent word pairs
- Smoothed with Add-k (k=0.3)

### Model Formula
```
P(Class|Sentence) ∝ P(Class) × 
                    P(has_URL|Class) × 
                    P(has_NUMBER|Class) × 
                    P(has_PUNCT|Class) ×
                    ∏ P(bigram|Class)
```

### Classification Process
1. Calculate log probabilities for each class
2. Avoid underflow using log-space computation
3. Select class with highest log probability

## Training Data Examples
```
"Check out https://example.com for more info!" → Inform
"Order 3 items, get 1 free! Limited offer!!!" → Promo
"Meeting at 3pm, don't forget to bring files." → Reminder
```

## Model Output

### Class Priors
```
Inform:   0.3333
Promo:    0.3333
Reminder: 0.3333
```

### Feature Probabilities
Model learns conditional probabilities like:
- P(has_URL=1 | Inform) = high
- P(has_NUMBER=1 | Promo) = high
- P(has_PUNCT=1 | Promo) = very high (exclamation marks!)

## Example Prediction
```
Input: "You will get an exclusive offer in the meeting!"

Features:
- has_URL: 0
- has_NUMBER: 0
- has_PUNCT: 1
- Bigrams: [('you', 'will'), ('will', 'get'), ...]

Scores:
Inform:    -12.3456
Promo:     -10.1234  ← HIGHEST
Reminder:  -13.5678

Predicted: Promo
```

## Technical Details

### BPE Advantages
- Handles out-of-vocabulary words gracefully
- Balances vocabulary size vs. sequence length
- Learns meaningful subword units
- Used in GPT, RoBERTa, and other transformers

### Naive Bayes Advantages
- Fast training and prediction
- Works well with small datasets
- Probabilistic outputs
- Interpretable feature weights

## Running the Code
```bash
jupyter notebook main.ipynb
```

Both implementations are in the same notebook with verbose output for understanding each step.

## Applications
- **BPE**: Tokenization for neural models, handling morphologically rich languages
- **Naive Bayes**: Spam detection, sentiment analysis, document categorization

