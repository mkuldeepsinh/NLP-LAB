# Natural Language Processing Lab

A comprehensive collection of NLP assignments covering fundamental to advanced concepts in computational linguistics and text processing.

## 📚 Assignment Overview

### Assignment 1: Text Tokenization for Gujarati
Custom regex-based tokenizers for Gujarati text with corpus statistics. Processes 100K+ documents from IndicCorpV2 dataset.

**Key Concepts:** Sentence segmentation, word tokenization, corpus analysis

### Assignment 2: Finite Automata & Morphology
DFA implementation for pattern recognition and morphological analysis of English nouns using trie-based stemming.

**Key Concepts:** Finite state machines, stemming, prefix/suffix tries

### Assignment 3: Frequency Analysis & Stopwords
Statistical analysis of word frequencies with threshold-based stopword removal and visualization for Gujarati corpus.

**Key Concepts:** Frequency distribution, stopword identification, data visualization

### Assignment 4: N-gram Language Models
Unigram through quadrigram models with three smoothing techniques (Add-one, Add-k, Token Type) for probability estimation.

**Key Concepts:** N-grams, MLE, smoothing techniques, probability distributions

### Assignment 5: Advanced Smoothing Techniques
Good-Turing smoothing and Deleted Interpolation with EM algorithm for robust language modeling.

**Key Concepts:** Good-Turing estimation, interpolation, frequency-of-frequencies, EM algorithm

### Assignment 6: Text Generation
Greedy sampling and beam search algorithms for generating coherent sentences using pre-trained n-gram models.

**Key Concepts:** Text generation, beam search, greedy algorithms, decoding strategies

### Assignment 7: Distributional Semantics
PMI/PPMI calculations and TF-IDF vectorization with nearest neighbor search for semantic similarity.

**Key Concepts:** Pointwise Mutual Information, TF-IDF, cosine similarity, document retrieval

### Assignment 8: BPE & Text Classification
Byte Pair Encoding tokenization and Naive Bayes classifier with custom features for message categorization.

**Key Concepts:** Subword tokenization, Naive Bayes, feature engineering, text classification

### Assignment 9: Subword Tokenization Algorithms
From-scratch implementations of BPE and WordPiece tokenization used in GPT and BERT models.

**Key Concepts:** Byte Pair Encoding, WordPiece, vocabulary construction, merge operations

### Assignment 10: HMM POS Tagging
Hidden Markov Model with Viterbi algorithm for automatic part-of-speech tagging with 90%+ accuracy.

**Key Concepts:** Hidden Markov Models, Viterbi algorithm, sequence labeling, POS tagging

## 🛠 Technologies Used

- **Python 3.x** - Primary programming language
- **NLTK** - Natural language toolkit
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation
- **Scikit-learn** - Machine learning utilities
- **Matplotlib/Seaborn** - Data visualization
- **Datasets (Hugging Face)** - Corpus loading

## 📂 Project Structure

```
NLP/
├── assignment_01/   # Gujarati tokenization
├── assignment_02/   # DFA & morphology
├── assignment_03/   # Frequency analysis
├── assignment_04/   # N-gram models
├── assignment_05/   # Advanced smoothing
├── assignment_06/   # Text generation
├── assignment_07/   # PMI & TF-IDF
├── assignment_08/   # BPE & Naive Bayes
├── assignment_09/   # Subword tokenization
└── assignment_10/   # HMM POS tagging
```

Each assignment folder contains:
- Implementation files (`.py` or `.ipynb`)
- Detailed `README.md` with explanations
- Generated outputs/visualizations (where applicable)

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy pandas nltk scikit-learn matplotlib seaborn datasets
```

### Running Assignments
Navigate to any assignment folder and follow its specific README instructions:
```bash
cd assignment_XX
python main.py
# or
jupyter notebook main.ipynb
```

## 📖 Learning Path

**Beginners** → Start with Assignments 1-4 (tokenization, basic models)  
**Intermediate** → Progress to Assignments 5-7 (advanced models, semantics)  
**Advanced** → Tackle Assignments 8-10 (modern algorithms, sequence models)

## 🎯 Key Takeaways

- **Text Preprocessing**: Tokenization, normalization, cleaning
- **Statistical Models**: N-grams, smoothing, probability estimation
- **Vector Representations**: TF-IDF, PMI, embeddings concepts
- **Sequence Modeling**: HMMs, Viterbi, dynamic programming
- **Modern Tokenization**: BPE, WordPiece (foundation for transformers)
- **Classification**: Naive Bayes, feature engineering
- **Text Generation**: Sampling strategies, beam search

## 📝 Notes

- Most assignments work with **Gujarati** corpus (IndicCorpV2 dataset)
- Implementations prioritize **clarity** over optimization
- Code follows **modular design** principles
- Each assignment is **self-contained** and runnable independently

---

**Course**: Natural Language Processing Lab  
**Focus**: Fundamental NLP concepts with practical implementations

