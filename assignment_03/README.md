# Assignment 3: Frequency Analysis & Stopword Removal

## Overview
This assignment performs statistical analysis on a Gujarati text corpus, focusing on word frequency distributions and stopword identification. The goal is to understand which words appear most frequently and how removing high-frequency words affects the corpus.

## What This Does

### 1. Frequency Distribution Analysis
- Loads 10,000 documents from the IndicCorpV2 Gujarati corpus
- Tokenizes and counts word occurrences
- Identifies the top 100 most frequent words
- Visualizes frequency distributions using bar charts

### 2. Stopword Removal Experiments
Tests three different frequency thresholds to filter out common words:
- **Threshold 1**: Words appearing < 100 times
- **Threshold 2**: Words appearing < 500 times  
- **Threshold 3**: Words appearing < 200 times

Each threshold generates a new visualization showing how the vocabulary changes when high-frequency words are removed.

## Dataset
- **Source**: ai4bharat/IndicCorpV2 (Gujarati)
- **Sample Size**: 10,000 documents
- **Language**: Gujarati script

## Generated Outputs

The notebook creates several visualization files:
- `top_100_words.png` - Original top 100 words
- `top_100_threshold1.png` - After applying threshold 1
- `top_100_threshold2.png` - After applying threshold 2
- `top_100_threshold3.png` - After applying threshold 3

## Key Insights

Stopwords (common function words) typically include:
- છે (is/are)
- અને (and)
- આ (this)
- કે (that)
- માટે (for)

These words appear very frequently but carry less semantic meaning than content words.

## Technical Details
- Uses regex for tokenization
- Matplotlib for visualizations
- Attempts to use Gujarati fonts for proper display
- Frequency-based stopword identification approach

## Running the Code
1. Open `assignment_03.ipynb`
2. Install dependencies: `datasets`, `matplotlib`, `pandas`
3. Run all cells to generate frequency plots
4. Adjust thresholds to experiment with different cutoff values

## Notes
- Gujarati characters may not render correctly without proper font support
- The notebook attempts to find system fonts that support Gujarati
- Consider installing "Noto Sans Gujarati" for best results

