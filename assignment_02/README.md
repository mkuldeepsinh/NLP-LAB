# Assignment 2: Finite Automata & Morphological Analysis

## Overview
This assignment explores two fundamental concepts in Natural Language Processing:
1. **Deterministic Finite Automaton (DFA)** for pattern recognition
2. **Morphological Analysis** for English noun forms

## Part 1: DFA Implementation

A simple finite state machine that validates strings containing only lowercase English letters (a-z).

### How It Works
The DFA has three states:
- **q0** (Start): Initial state
- **q1** (Accept): Final state for valid strings
- **q2** (Trap): Reject state for invalid inputs

### Accepted Examples
- "cat", "dog", "zebra", "helloworld" ✓

### Rejected Examples  
- "dog1", "1dog", "Dog_house", " cats" ✗

## Part 2: Noun Morphological Analyzer

Analyzes English nouns to identify their stem and grammatical number (singular/plural).

### Rules Implemented
- **"-ies" endings**: babies → baby+N+PL
- **"-es" endings**: watches → watch+N+PL (for words ending in ch, sh, s, x, z)
- **"-s" endings**: bags → bag+N+PL
- **Singular forms**: cat → cat+N+SG

### Output Format
`stem+N+{SG|PL}` where:
- `stem` = base form of the word
- `N` = noun tag
- `SG/PL` = singular/plural marker

## Part 3: Trie-Based Stemming

Implements prefix and suffix tries for automatic stemming discovery.

### Approach
- **Prefix Trie**: Finds common word beginnings
- **Suffix Trie**: Identifies common word endings
- **Scoring Function**: `frequency × number_of_children` to find optimal split points

### Dataset
Uses the Brown Corpus noun list (`brown_nouns.txt`) with 200,000+ words for training.

## Files
- `assignment_02.ipynb`: Main implementation notebook
- `1.py`: Standalone Python script version

## Running the Code
1. Ensure you have the Brown corpus nouns file
2. Run the notebook cells sequentially
3. Test with custom words to see stemming results

## Example Output
```
investigation = in + vestigation (Prefix)
investigation = investigatio + n (Suffix)

election = e + lection (Prefix)  
election = electio + n (Suffix)
```

