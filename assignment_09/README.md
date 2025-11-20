# Assignment 9: Subword Tokenization - BPE & WordPiece

## Overview
This assignment implements two popular subword tokenization algorithms from scratch: **Byte Pair Encoding (BPE)** and **WordPiece**. These algorithms are the foundation of modern language models like GPT (BPE) and BERT (WordPiece), enabling them to handle unlimited vocabularies while maintaining reasonable model sizes.

## What This Does

### The Subword Tokenization Problem
Traditional word-level tokenization faces challenges:
- **Large Vocabularies**: Millions of unique words increase memory usage
- **Out-of-Vocabulary (OOV)**: Unknown words become `<UNK>` tokens
- **Morphology**: Related words (walk, walking, walked) treated as completely separate
- **Rare Words**: Insufficient training data for uncommon words

Subword tokenization solves these by breaking words into meaningful pieces.

## Algorithms Implemented

### 1. Byte Pair Encoding (BPE)

#### Origins
- Originally a data compression algorithm (1994)
- Adapted for NLP by Sennrich et al. (2016)
- Used in GPT, GPT-2, RoBERTa, BART

#### How It Works

**Training Phase:**
1. Start with character-level vocabulary + `</w>` (end-of-word marker)
2. Count all adjacent symbol pairs in the corpus
3. Merge the most frequent pair into a new symbol
4. Repeat for N iterations or until vocab size reached
5. Store merge operations in order

**Example Training:**
```
Iteration 1: Merge ('l', 'o') → 'lo'
  "hello" → "hel" + "lo" + "</w>"

Iteration 2: Merge ('lo', '</w>') → 'lo</w>'
  "hel" + "lo" + "</w>" → "hel" + "lo</w>"

Iteration 3: Merge ('h', 'e') → 'he'
  "hel" + "lo</w>" → "he" + "l" + "lo</w>"
```

**Tokenization Phase:**
Apply learned merges in order to segment new words.

#### Advantages
✅ Handles OOV words gracefully (falls back to characters)  
✅ Captures common morphological patterns  
✅ Balances vocabulary size vs. sequence length  
✅ No word boundaries needed  

### 2. WordPiece

#### Origins
- Developed by Google for speech recognition (2012)
- Enhanced for BERT by Schuster & Nakajima (2012)
- Used in BERT, DistilBERT, ELECTRA

#### How It Works

**Key Difference from BPE:**
While BPE merges by raw frequency, WordPiece can use likelihood-based scoring:
```
score(pair) = freq(pair) / (freq(first) × freq(second))
```

This implementation uses frequency-based merging similar to BPE for simplicity.

**WordPiece Specifics:**
- Often uses `##` prefix for non-initial subwords (not in this implementation)
- Greedy longest-match-first during tokenization
- Slightly different vocabulary construction

#### Example
```
Input: "tokenization"

BPE might produce: ['token', 'ization', '</w>']
WordPiece might produce: ['token', '##ization']
```

## Implementation Details

### Core Functions

#### 1. Vocabulary Building
```python
def build_vocab(words):
    # Splits each word into characters + </w>
    # Returns: Counter of word tuples with frequencies
```

#### 2. Pair Counting
```python
def get_pair_counts(vocab):
    # Counts all adjacent symbol pairs
    # Returns: Counter of pairs with frequencies
```

#### 3. Vocabulary Merging
```python
def merge_vocab(vocab, pair):
    # Merges selected pair throughout vocabulary
    # Returns: Updated vocabulary + new merged symbol
```

#### 4. Training
```python
def train_bpe(words, merge_steps=1000, vocab_size=None):
    # Iteratively merges most frequent pairs
    # Returns: List of merges + final vocabulary
```

#### 5. Tokenization
```python
def apply_bpe(word, merges):
    # Applies learned merges to segment new word
    # Returns: List of subword tokens
```

### Key Parameters

#### merge_steps
- Number of merge iterations
- Default: 1000
- Larger → more subwords, smaller vocabulary
- Common values: 8K-40K for real models

#### vocab_size
- Target vocabulary size limit
- Default: 32000
- Controls when to stop merging
- Trade-off: vocabulary size vs. sequence length

## Code Structure

### Data Loading
```python
words = load_corpus("corpus.txt")
```
- Loads text from file
- Tokenizes into words
- Lowercases (optional)
- Cleans whitespace

### Training Both Models
```python
bpe_merges, bpe_vocab = train_bpe(words, merge_steps=32000, vocab_size=32000)
wp_merges, wp_vocab = train_wordpiece(words, vocab_size=32000, merge_steps=32000)
```

### Tokenizing New Text
```python
tokens = apply_bpe("unknown_word", bpe_merges)
```

## Example Output

### Sample Text
```
Input: "byte pair encoding and wordpiece tokenization"

BPE Output:
['byt', 'e', 'pair', 'encod', 'ing', 'an', 'd', 'wordpiec', 'e', 'tokenization</w>']

WordPiece Output:
['byt', 'e', 'pair', 'encod', 'ing', 'an', 'd', 'wordpiec', 'e', 'tokenization</w>']
```

### Observations
- Common words become single tokens: "pair"
- Morphological patterns captured: "encod" + "ing"
- Suffix identification: "ization"
- Rare words broken into subwords

## Running the Code

### Prerequisites
```python
# No external ML libraries needed!
# Only standard library:
import re
from collections import Counter
```

### Basic Usage
```bash
jupyter notebook main.ipynb
```

### With Custom Corpus
1. Create `corpus.txt` with your training text
2. Adjust `MERGES` and `VOCAB` parameters
3. Run all cells
4. Test with `apply_bpe()` or `apply_wordpiece()`

## Real-World Applications

### GPT Models (BPE)
- GPT-2: 50K BPE vocabulary
- GPT-3: Byte-level BPE variant
- Enables multilingual capabilities

### BERT Models (WordPiece)
- BERT-base: 30K WordPiece vocabulary
- Covers 100+ languages with shared vocab
- Better handling of morphology

### Modern Usage
- **Sentence Piece**: Unified BPE/Unigram framework
- **Hugging Face Tokenizers**: Fast Rust implementations
- **tiktoken**: OpenAI's efficient tokenizer

## Comparison: BPE vs WordPiece

| Aspect | BPE | WordPiece |
|--------|-----|-----------|
| **Merge Strategy** | Frequency-based | Likelihood-based (original) |
| **Speed** | Faster training | Slightly slower |
| **Quality** | Good for general text | Better for morphology |
| **Used In** | GPT, RoBERTa, BART | BERT, DistilBERT |
| **Notation** | `</w>` for word end | `##` for continuation |

## Advantages of Subword Tokenization

### 1. No OOV Problem
```
Unknown word: "unhappiness"
BPE: ['un', 'happy', 'ness', '</w>']
```

### 2. Vocabulary Efficiency
- 30K subwords cover millions of words
- Much smaller than word-level (100K-1M words)
- Larger than character-level (26-100 chars)

### 3. Morphological Awareness
```
"walk":    ['walk', '</w>']
"walked":  ['walk', 'ed', '</w>']
"walking": ['walk', 'ing', '</w>']
```

### 4. Multilingual Support
Single vocabulary across languages with shared scripts.

## Limitations

⚠️ **No Semantic Meaning**: Splits are statistical, not linguistic  
⚠️ **Language-Dependent**: Performance varies by language  
⚠️ **Corpus Bias**: Quality depends on training data  
⚠️ **Tokenization Artifacts**: Can split words awkwardly  

## Hyperparameter Tuning

### Vocabulary Size
- **Small (1K-5K)**: Long sequences, better generalization
- **Medium (10K-30K)**: Balanced performance
- **Large (50K-100K)**: Shorter sequences, more memory

### Merge Steps
- Directly affects final vocabulary size
- Usually set equal to vocab_size
- Early stopping if vocab_size reached

### Corpus Size
- Minimum: 1M tokens for basic quality
- Recommended: 10M+ tokens
- Best: 100M+ tokens for production models

## Extensions & Improvements

### Add Frequency Threshold
Filter rare pairs to improve quality:
```python
if pairs[best] < min_frequency:
    break
```

### Character Coverage
Ensure rare characters are preserved:
```python
base_vocab = set(unique_characters)
```

### Likelihood Scoring
True WordPiece uses:
```python
score = freq(AB) / (freq(A) * freq(B))
```

### Unicode Handling
Better support for non-ASCII:
```python
import unicodedata
text = unicodedata.normalize('NFKC', text)
```

## Files
- `main.ipynb` - Complete implementation with examples
- `corpus.txt` - Training corpus (optional, uses demo text if missing)

## Performance Considerations

### Time Complexity
- Training: O(N × M × V) where N=merges, M=corpus size, V=vocab size
- Tokenization: O(M × W) where M=merges, W=word length

### Space Complexity
- Vocabulary storage: O(V)
- Merge list: O(M)
- Efficient for production use

## Further Reading
- Sennrich et al. (2016): "Neural Machine Translation of Rare Words with Subword Units"
- Schuster & Nakajima (2012): "Japanese and Korean Voice Search"
- SentencePiece: Unified subword tokenization (Kudo & Richardson, 2018)

