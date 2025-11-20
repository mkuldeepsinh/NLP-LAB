# Assignment 7: Distributional Semantics & TF-IDF

## Overview
This assignment explores distributional semantics through two complementary approaches: Pointwise Mutual Information (PMI) for measuring word associations and TF-IDF for document similarity. It includes both intra-set and inter-set nearest neighbor searches.

## What This Does

### 1. Data Preparation
- Loads 9 million documents from IndicCorpV2 Gujarati corpus
- Cleans and tokenizes using Gujarati Unicode range (U+0A80-U+0AFF)
- Splits data into train/validation/test (80%/10%/10%)
- Generates n-gram frequency tables

### 2. Pointwise Mutual Information (PMI)
Measures how much more likely two words appear together vs. independently.

#### Formula
```
PMI(w₁, w₂) = log₂(P(w₁,w₂) / (P(w₁) × P(w₂)))
PPMI(w₁, w₂) = max(0, PMI(w₁, w₂))
```

#### What It Reveals
- High PMI → strong collocation (words that go together)
- Zero PMI → independent occurrence
- PPMI filters negative associations

### 3. TF-IDF Vectorization
Converts text to numerical vectors for similarity computation.

#### Components
- **TF (Term Frequency)**: How often a word appears in document
- **IDF (Inverse Document Frequency)**: Rarity of word across corpus
- **TF-IDF**: `TF × IDF` - highlights important, distinctive words

### 4. Nearest Neighbor Search

#### Intra-Set Search (Question 3)
- Finds most similar sentence **within the same set**
- Uses cosine similarity between TF-IDF vectors
- Validates that similar documents exist in validation/test sets

#### Inter-Set Search (Question 4 + Bonus)
- Finds most similar sentence **across different sets**
- Searches validation/test queries against training corpus
- Uses sklearn's NearestNeighbors for efficiency
- Demonstrates retrieval capabilities

## Generated Files

### N-gram Counts
- `unigram_counts.csv` - Word frequency table
- `bigram_counts.csv` - Word pair frequency table

### Text Splits
- `train_sentences.txt` - 80% of data for training
- `val_sentences.txt` - 10% for validation
- `test_sentences.txt` - 10% for testing

## Key Results

### Top PMI Pairs (Examples)
```
PPMI(હરિત, ક્રાંતિને): 19.6338
PPMI(બ્રિટની, સ્પીયર્સ): 19.6338
```
These pairs show strong collocations (named entities, compound terms).

### TF-IDF Statistics
- Vocabulary: ~11,500 unique terms (after preprocessing)
- Matrix dimensions: 20,000 × 11,511 (train)
- Sparse representation for memory efficiency

### Similarity Scores
- Range: 0 to 1 (cosine similarity)
- Higher scores indicate more similar content
- Typical intra-set similarities: 0.2 - 0.6

## Technical Implementation

### Preprocessing Pipeline
1. Remove non-Gujarati characters
2. Replace multiple spaces with single space
3. Tokenize by whitespace
4. Filter empty sentences

### Efficiency Optimizations
- Streaming dataset to handle large corpus
- Sparse matrix representation for TF-IDF
- Brute-force KNN with cosine metric
- Progress tracking every 5,000 documents

## Running the Code

### Full Pipeline
```bash
python main.py
```

### Notebook Version
```bash
jupyter notebook main.ipynb
```

## Use Cases
- **PMI**: Finding collocations, phrase detection, word associations
- **TF-IDF**: Document retrieval, similarity search, clustering
- **Nearest Neighbors**: Question answering, duplicate detection, recommendation

## Dependencies
```
nltk
pandas
sklearn
datasets
numpy
```

## Performance Notes
- Processing 9M documents takes significant time
- Consider reducing `NUM_DOCS_TO_TAKE` for faster experiments
- GPU acceleration not applicable (CPU-bound operations)
- Memory usage scales with vocabulary size

