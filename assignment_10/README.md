# Assignment 10: Hidden Markov Model (HMM) POS Tagging

## Overview
This assignment implements a Hidden Markov Model (HMM) with the Viterbi algorithm for Part-of-Speech (POS) tagging. The model learns statistical patterns from tagged training data to automatically assign POS tags to words in new sentences.

## What This Does

### 1. Hidden Markov Model
An HMM models sequences with two components:
- **Hidden States**: POS tags (noun, verb, adjective, etc.)
- **Observations**: Words in the sentence

The model assumes:
- Current tag depends only on previous tag (Markov property)
- Word depends only on its tag (emission independence)

### 2. Model Components

#### Initial Probabilities
`P(tag)` - Probability that a sentence starts with a given tag
```
P(NOUN) = 0.35
P(DET)  = 0.28
P(VERB) = 0.15
```

#### Transition Probabilities
`P(tag₂|tag₁)` - Probability of transitioning from one tag to another
```
P(VERB|NOUN) = 0.25
P(ADJ|DET)   = 0.40
```

#### Emission Probabilities
`P(word|tag)` - Probability of observing a word given its tag
```
P("run"|VERB) = 0.15
P("dog"|NOUN) = 0.08
```

### 3. Viterbi Algorithm
Finds the most likely sequence of tags for a sentence using dynamic programming.

#### How It Works
1. **Initialization**: Calculate probabilities for first word
   - `V[tag][0] = log(P(tag)) + log(P(word₀|tag))`

2. **Recursion**: For each subsequent word
   - `V[tag][t] = max(V[prev][t-1] + log(P(tag|prev)) + log(P(word_t|tag)))`
   - Store best previous tag in backpointer

3. **Backtracking**: Follow backpointers to reconstruct best path
   - Start from tag with highest probability at end
   - Walk backwards to build complete tag sequence

#### Why Log Probabilities?
- Prevents numerical underflow (multiplying many small probabilities)
- Converts products to sums for stability
- Standard practice in sequence modeling

## Training Process

### Data Format
Input file uses `word_tag` format:
```
The_DET cat_NOUN sat_VERB on_PREP the_DET mat_NOUN
```

### Learning Steps
1. **Parse Training Data**: Extract word-tag pairs from sentences
2. **Count Frequencies**:
   - Initial tag counts (sentence-starting tags)
   - Transition counts (tag-to-tag transitions)
   - Emission counts (word-tag pairs)
3. **Normalize to Probabilities**: Divide by totals to get probability distributions

### Smoothing
Uses add-epsilon smoothing (ε=1e-10) for unseen transitions/emissions to prevent zero probabilities.

## Model Architecture

```
Input Sentence: "The dog runs"

Viterbi Matrix:
         The      dog      runs
DET    -2.3     -8.5     -15.2
NOUN   -5.1     -3.2     -12.8
VERB   -8.7     -9.3     -5.6  ← Best path
ADJ    -6.4     -7.8     -14.3

Output Tags: DET → NOUN → VERB
```

## Implementation Features

### Class: HMM_POS_Tagger

#### Attributes
- `transition_prob`: Tag-to-tag probabilities
- `emission_prob`: Word-given-tag probabilities
- `initial_prob`: Sentence-starting tag probabilities
- `tags`: Set of all POS tags in training data
- `vocab`: Set of all words in training data

#### Methods
- `train(sentences)`: Learn probabilities from training data
- `viterbi(words)`: Find best tag sequence using Viterbi
- `predict(sentence)`: Tag a new sentence
- `evaluate(test_sentences)`: Calculate accuracy on test set

## Running the Code

### Requirements
```bash
pip install numpy
```

### Data Preparation
Ensure `pos_tagdata.txt` is in the same directory with format:
```
word1_TAG1 word2_TAG2 word3_TAG3
```

### Execution
```bash
python main.py
```

### Expected Output
```
Loading data...
Total sentences: 5000

Splitting data (80:20)...
Training sentences: 4000
Testing sentences: 1000

Training HMM model...
Number of unique tags: 12
Number of unique words: 8547
Tags: ['ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRON', 'PRT', 'VERB', '.', 'X']

Evaluating on test set...

Results:
Accuracy: 93.45%
Correct predictions: 9345/10000

Sample predictions:
Sentence 1:
Words: The quick brown fox jumps...
True tags: DET ADJ ADJ NOUN VERB...
Predicted: DET ADJ ADJ NOUN VERB...
```

## Performance Metrics

### Typical Accuracy
- **Training Accuracy**: 95-98%
- **Test Accuracy**: 90-95%
- Depends on corpus size and tag complexity

### Error Sources
1. **Unseen Words**: Words not in training vocabulary
2. **Ambiguous Words**: Words with multiple valid tags (e.g., "run" as NOUN or VERB)
3. **Rare Transitions**: Uncommon tag sequences
4. **Data Sparsity**: Insufficient training examples

## Advantages of HMM POS Tagging

✅ **Fast**: O(T²N) complexity where T=tags, N=words  
✅ **Interpretable**: Probabilities have clear meaning  
✅ **Memory Efficient**: Simple probability tables  
✅ **Proven**: Decades of successful applications  

## Limitations

⚠️ **Context**: Only considers previous tag (first-order Markov)  
⚠️ **OOV Words**: Struggles with out-of-vocabulary words  
⚠️ **Fixed Dependencies**: Can't model long-range dependencies  
⚠️ **Feature Poor**: Only uses word identity, no morphology/capitalization  

## Improvements & Extensions

### Better Smoothing
- Witten-Bell smoothing
- Kneser-Ney smoothing
- Interpolation with lower-order models

### Feature Engineering
- Suffix patterns (e.g., "-ing" → likely VERB)
- Capitalization (proper nouns)
- Word shape features

### Higher-Order Models
- Trigram HMM (consider two previous tags)
- Maximum Entropy Markov Models (MEMM)
- Conditional Random Fields (CRF)

### Modern Alternatives
- BiLSTM-CRF models
- Transformer-based taggers (BERT, RoBERTa)
- Fine-tuned language models

## Use Cases
- **Preprocessing**: For parsing, NER, information extraction
- **Grammar Checking**: Identify grammatical errors
- **Text-to-Speech**: Pronunciation depends on POS
- **Machine Translation**: Alignment and reordering
- **Information Retrieval**: Query understanding

## Files
- `main.py` - Complete HMM implementation with Viterbi
- `pos_tagdata.txt` - Training/testing data (required)

## Technical Notes

### Memory Complexity
- O(T × V) for emission probabilities
- O(T²) for transition probabilities
- Manageable even for large vocabularies

### Speed Optimization
- Uses numpy for vectorized operations
- Log-space computation prevents overflow
- Efficient backtracking with index array

### Numerical Stability
- All probabilities in log-space
- Epsilon smoothing prevents log(0)
- Integer backpointers save memory


