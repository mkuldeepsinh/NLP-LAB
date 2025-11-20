# Assignment 1: Text Tokenization for Gujarati Corpus

## Overview
This assignment focuses on building a tokenization system for the Gujarati language using the IndicCorp V2 dataset. The goal is to process raw Gujarati text and extract meaningful statistics about the corpus.

## What This Does

The notebook implements custom tokenizers for Gujarati text:
- **Sentence Tokenizer**: Splits text into individual sentences using regex patterns
- **Word Tokenizer**: Breaks sentences into individual words/tokens

After tokenization, the system calculates various corpus statistics including:
- Total number of sentences and words
- Average sentence length
- Average word length
- Type-Token Ratio (TTR) - a measure of vocabulary diversity

## Dataset
- **Source**: ai4bharat/IndicCorpV2 (Gujarati section)
- **Processing**: 100,000 sample documents from the streaming dataset
- **Language**: Gujarati (guj_Gujr script)

## Key Results
From the processed corpus:
- **141,403 sentences** extracted
- **4.4+ million words** tokenized
- **Average sentence length**: 31.36 words per sentence
- **Average word length**: 1.41 characters per word
- **Type/Token Ratio**: 0.01 (indicates vocabulary diversity)

## Running the Code
1. Open `assignment_01.ipynb` in Jupyter or Google Colab
2. Install required dependencies: `nltk`, `datasets`, `huggingface_hub`
3. Run cells sequentially to:
   - Load the Gujarati dataset
   - Apply regex-based tokenizers
   - Generate corpus statistics

## Technical Details
- Uses regex patterns for sentence and word boundary detection
- Handles Gujarati Unicode character range (U+0A80 to U+0AFF)
- Streaming dataset processing to handle large corpus efficiently

