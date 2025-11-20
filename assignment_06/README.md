# Assignment 6: Text Generation with N-gram Models

## Overview
This assignment implements text generation algorithms using pre-trained n-gram models. It compares greedy probabilistic sampling with beam search to generate coherent Gujarati sentences.

## What This Does

### 1. N-gram Model Loading
- Loads pre-trained trigram and quadrigram models from CSV files
- Models contain word sequence counts for probability estimation
- Uses Maximum Likelihood Estimation (MLE) for probability calculations

### 2. Generation Algorithms

#### Greedy Generation
- Samples next word based on probability distribution
- Uses `random.choices()` weighted by n-gram counts
- Faster but may not find optimal sequences
- Terminates at `</s>` or max length (15 words)

#### Beam Search
- Maintains top-k candidate sequences (beam_size=20)
- Explores multiple paths simultaneously
- Scores sequences by cumulative probability
- Finds higher-quality, more coherent sentences

### 3. Backoff Strategy
When context not found in n-gram model:
- Falls back to lower-order model
- Uses trigram as ultimate fallback
- Randomly samples from available last words
- Ensures generation never gets stuck

### 4. Batch Generation
Generates 100 sentences for each configuration:
- Greedy with trigrams
- Greedy with quadrigrams
- Beam search with trigrams
- Beam search with quadrigrams

## Generated Files

### CSV Outputs
- `greedy_3gram_100.csv` - 100 sentences from greedy trigram
- `greedy_4gram_100.csv` - 100 sentences from greedy quadrigram
- `beam_3gram_100.csv` - 100 sentences from beam trigram
- `beam_4gram_100.csv` - 100 sentences from beam quadrigram

### Model Files (Required)
- `trigrams.csv` - Pre-trained trigram counts
- `quadgrams.csv` - Pre-trained quadrigram counts

## Algorithm Comparison

### Greedy Sampling
**Pros:**
- Fast execution
- Natural variation in output
- Lower memory usage

**Cons:**
- May produce less coherent text
- Single path exploration
- Can get stuck in local optima

### Beam Search
**Pros:**
- Higher quality output
- Explores multiple hypotheses
- Better global coherence

**Cons:**
- Slower execution
- Higher memory requirements
- More deterministic (less variety)

## Technical Details

### Probability Calculation
```python
P(word|context) = Count(context, word) / Count(context)
```

### Context Handling
- Trigram: Uses last 2 words as context
- Quadrigram: Uses last 3 words as context
- Initializes with `<s>` tokens

### Beam Size Tuning
- Default: 20 sequences
- Larger beams = better quality, slower speed
- Smaller beams = faster, more variation

## Running the Code

### Notebook Version
```bash
jupyter notebook generator.ipynb
```

### Python Script
```bash
python main.py
```

## Files Included
- `main.ipynb` - Complete implementation notebook
- `generator.ipynb` - Specialized generation interface
- `main.py` - Standalone script
- `generated_sentences.txt` - Sample outputs

## Sample Output
```
<s> <s> આ વર્ષે ગુજરાત માં વરસાદ સારો રહ્યો છે </s>
<s> <s> શિક્ષણ ક્ષેત્રે નવા સુધારા કરવામાં આવશે </s>
```

## Use Cases
- Data augmentation for ML models
- Creative text generation
- Language model evaluation
- Corpus expansion
- Testing NLP pipelines

