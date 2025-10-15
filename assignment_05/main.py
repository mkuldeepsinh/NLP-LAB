import numpy as np
import pandas as pd
from collections import Counter
from datasets import load_dataset

# --------------------------------------------------------------------------
# PART 1: DATA PREPARATION & SPLITTING
# --------------------------------------------------------------------------
print("--- Part 1: Loading and Splitting Data ---")

# Load the Gujarati dataset in streaming mode
try:
    streaming_dataset = load_dataset("ai4bharat/IndicCorpV2", "indiccorp_v2", streaming=True, split="guj_Gujr")
    print("Successfully loaded the ai4bharat/IndicCorpV2 Gujarati dataset.")
except Exception as e:
    print(f"Failed to load dataset. Error: {e}")
    exit()

# Since the dataset is streamed, we can't shuffle it directly.
# We'll iterate through it to create our splits.
# We'll take a fixed number of sentences for training to keep it manageable.
TRAIN_SIZE = 500000
VAL_SIZE = 250000
TEST_SIZE = 250000

print(f"Taking {TRAIN_SIZE} sentences for training, {VAL_SIZE} for validation, and {TEST_SIZE} for testing.")

def tokenize(text):
    """A simple tokenizer that splits by space and converts to lowercase."""
    return text.lower().split()

# Create the data splits by iterating through the stream
iterator = iter(streaming_dataset)
val_set = [tokenize(next(iterator)['text']) for _ in range(VAL_SIZE)]
test_set = [tokenize(next(iterator)['text']) for _ in range(TEST_SIZE)]
train_set = [tokenize(next(iterator)['text']) for _ in range(TRAIN_SIZE)]

print(f"Training set size: {len(train_set)} sentences")
print(f"Validation set size: {len(val_set)} sentences")
print(f"Test set size: {len(test_set)} sentences")
print("-" * 50)


# --------------------------------------------------------------------------
# PART 2 & 3: N-GRAM MODELS WITH GOOD-TURING SMOOTHING
# --------------------------------------------------------------------------

class NgramGoodTuringModel:
    
    def __init__(self, n, verbose=False):
        self.n = n
        self.verbose = verbose
        self.vocab = set()
        self.ngram_counts = Counter()
        self.freq_of_freqs = Counter()
        self.total_ngrams = 0
        self.smoothed_counts = {}

    def _pad_sentence(self, sentence):
        """Adds start and end padding to a sentence."""
        padding = ['<s>'] * (self.n - 1)
        return padding + sentence + ['</s>']

    def fit(self, sentences):
        """Trains the N-gram model on the training sentences."""
        if self.verbose:
            print(f"\n--- Training {self.n}-gram model ---")

        # Step 1: Count n-grams and build vocabulary
        all_ngrams = []
        for sentence in sentences:
            padded_sent = self._pad_sentence(sentence)
            # Add all words to vocabulary
            for word in padded_sent:
                self.vocab.add(word)
            # Generate and collect n-grams
            for i in range(len(padded_sent) - self.n + 1):
                ngram = tuple(padded_sent[i:i+self.n])
                all_ngrams.append(ngram)

        self.ngram_counts = Counter(all_ngrams)
        self.total_ngrams = sum(self.ngram_counts.values()) # This is N

        # Step 2: Calculate frequency of frequencies (Nc)
        for count in self.ngram_counts.values():
            self.freq_of_freqs[count] += 1

        # Step 3: Pre-calculate smoothed counts (C*) for seen n-grams
        n1 = self.freq_of_freqs.get(1, 0) # This is N1
        for c, nc in self.freq_of_freqs.items():
            nc_plus_1 = self.freq_of_freqs.get(c + 1, 0)
            if nc_plus_1 == 0:
                # Fall back to using the original count c if N(c+1) is 0.
                self.smoothed_counts[c] = c
            else:
                self.smoothed_counts[c] = (c + 1) * (nc_plus_1 / nc)

        # Step 4: Calculate probability for unseen n-grams
        # P_unseen = (N1/N) / (number of unseen n-grams)
        if n1 == 0:
            self.unseen_prob = 0
        else:
            V = len(self.vocab)
            if self.n == 1:
                num_unseen = V - len(self.ngram_counts)
            else:
                num_unseen = V**self.n - len(self.ngram_counts)

            if num_unseen > 0:
                self.unseen_prob = (n1 / self.total_ngrams) / num_unseen
            else:
                self.unseen_prob = 0

        if self.verbose:
            print(f"Vocabulary size (V): {len(self.vocab)}")
            print(f"Total n-grams observed (N): {self.total_ngrams}")
            print(f"Number of n-grams seen once (N1): {n1}")
            print(f"Probability for a single unseen {self.n}-gram: {self.unseen_prob:.10e}")

    def get_ngram_log_prob(self, ngram):
        """Calculates the log probability of a single n-gram."""
        c = self.ngram_counts.get(ngram, 0)
        if c == 0: # Unseen n-gram
            prob = self.unseen_prob
        else: # Seen n-gram
            c_star = self.smoothed_counts.get(c, c)
            prob = c_star / self.total_ngrams

        return np.log(prob) if prob > 0 else -np.inf

    def calculate_sentence_log_prob(self, sentence):
        """Computes the sentence probability using the smoothed model."""
        padded_sent = self._pad_sentence(sentence)
        total_log_prob = 0.0
        for i in range(len(padded_sent) - self.n + 1):
            ngram = tuple(padded_sent[i:i+self.n])
            total_log_prob += self.get_ngram_log_prob(ngram)
        return total_log_prob

    def show_freq_table(self, top_n=100):
        """Shows a table with top frequencies as requested."""
        print(f"\n--- Part 3: Frequency Table for {self.n}-gram Model (Top {top_n}) ---")
        data = []
        # Sort by count C (MLE) in descending order
        sorted_freqs = sorted(self.freq_of_freqs.items(), key=lambda x: x[0], reverse=True)

        for c, nc in sorted_freqs[:top_n]:
            c_star = self.smoothed_counts.get(c, c)
            data.append({"C (MLE)": c, "Nc": nc, "C*": f"{c_star:.4f}"})

        df = pd.DataFrame(data)
        print(df.to_string(index=False))
        print("-" * 50)

# Initialize and train all four N-gram models
models_gt = {}
for n in range(1, 5):
    model_name = f'{n}-gram'
    models_gt[model_name] = NgramGoodTuringModel(n=n, verbose=(n==4))
    models_gt[model_name].fit(train_set)

# Show the frequency table for the Quadrigram model
models_gt['4-gram'].show_freq_table()

# Evaluate the models on the test set
print("\n--- Evaluating Good-Turing Models on Test Set ---")
for name, model in models_gt.items():
    avg_log_prob = np.mean([model.calculate_sentence_log_prob(s) for s in test_set])
    print(f"Model: {name}, Average Log Probability: {avg_log_prob:.4f}")
print("-" * 50)


# --------------------------------------------------------------------------
# PART 4: DELETED INTERPOLATION SMOOTHING
# --------------------------------------------------------------------------
print("\n--- Part 4: Deleted Interpolation Smoothing for Quadrigram Model ---")

class DeletedInterpolationModel:
    """
    Implements deleted interpolated smoothing for the quadrigram model.
    """
    def __init__(self):
        self.counts = {n: Counter() for n in range(1, 5)}
        self.context_counts = {n: Counter() for n in range(1, 4)}
        self.lambdas = np.array([0.25] * 4) # Initialize lambdas
        self.vocab = set()

    def _pad_sentence(self, sentence):
        return ['<s>'] * 3 + sentence + ['</s>']

    def fit_counts(self, sentences):
        """Fit all the raw n-gram counts needed for MLE probabilities."""
        for sentence in sentences:
            padded_sent = self._pad_sentence(sentence)
            for word in padded_sent:
                self.vocab.add(word)

            for i in range(len(padded_sent)):
                if i >= 0: self.counts[1][(padded_sent[i],)] += 1
                if i >= 1:
                    self.counts[2][tuple(padded_sent[i-1:i+1])] += 1
                    self.context_counts[1][(padded_sent[i-1],)] += 1
                if i >= 2:
                    self.counts[3][tuple(padded_sent[i-2:i+1])] += 1
                    self.context_counts[2][tuple(padded_sent[i-2:i])] += 1
                if i >= 3:
                    self.counts[4][tuple(padded_sent[i-3:i+1])] += 1
                    self.context_counts[3][tuple(padded_sent[i-3:i])] += 1
        
        # For smoothed unigram probabilities
        self.counts[0] = {
            'total': sum(self.counts[1].values()),
            'vocab_size': len(self.vocab)
        }

    def _get_mle_probs(self, quadrigram):
        """Calculate raw MLE probabilities for a quadrigram and its backoffs."""
        w1, w2, w3, w4 = quadrigram[0], quadrigram[1], quadrigram[2], quadrigram[3]
        probs = np.zeros(4)

        # P(w4 | w1, w2, w3)
        c_quad_context = self.context_counts[3].get((w1, w2, w3), 0)
        probs[3] = self.counts[4].get(quadrigram, 0) / c_quad_context if c_quad_context > 0 else 0

        # P(w4 | w2, w3)
        c_tri_context = self.context_counts[2].get((w2, w3), 0)
        probs[2] = self.counts[3].get((w2, w3, w4), 0) / c_tri_context if c_tri_context > 0 else 0

        # P(w4 | w3)
        c_bi_context = self.context_counts[1].get((w3,), 0)
        probs[1] = self.counts[2].get((w3, w4), 0) / c_bi_context if c_bi_context > 0 else 0

        # P(w4) with Add-1 smoothing
        c_uni = self.counts[1].get((w4,), 0)
        probs[0] = (c_uni + 1) / (self.counts[0]['total'] + self.counts[0]['vocab_size'])
        return probs

    def find_best_lambdas(self, val_sentences, num_iterations=10):
        """Finds best lambdas using Expectation-Maximization on the validation set."""
        print("\nFinding best lambdas using EM algorithm...")
        val_quadrigrams = [
            tuple(padded[i-3:i+1])
            for sent in val_sentences
            for i, padded in enumerate([self._pad_sentence(sent)])
            if i >= 3
        ]

        for i in range(num_iterations):
            expected_counts = np.zeros(4)
            for quad in val_quadrigrams:
                mle_probs = self._get_mle_probs(quad)
                weighted_probs = self.lambdas * mle_probs
                total_prob = np.sum(weighted_probs)
                if total_prob > 0:
                    expected_counts += weighted_probs / total_prob
            
            self.lambdas = expected_counts / np.sum(expected_counts)
            if (i + 1) % 2 == 0:
                print(f"Iteration {i+1}, Lambdas: {[f'{l:.4f}' for l in self.lambdas]}")
        
        print("\n✅ Best lambda parameters found:")
        print(f"λ1 (unigram): {self.lambdas[0]:.4f}, λ2 (bigram): {self.lambdas[1]:.4f}, "
              f"λ3 (trigram): {self.lambdas[2]:.4f}, λ4 (quadgram): {self.lambdas[3]:.4f}")

    def get_interpolated_log_prob(self, sentence):
        """Calculates sentence log probability using the found lambdas."""
        padded_sent = self._pad_sentence(sentence)
        total_log_prob = 0.0
        for i in range(3, len(padded_sent)):
            quadrigram = tuple(padded_sent[i-3:i+1])
            mle_probs = self._get_mle_probs(quadrigram)
            interpolated_prob = np.dot(self.lambdas, mle_probs)
            total_log_prob += np.log(interpolated_prob) if interpolated_prob > 0 else -np.inf
        return total_log_prob

# Initialize and train the interpolation model
interp_model = DeletedInterpolationModel()
# 1. Get raw counts from the training data
interp_model.fit_counts(train_set)
# 2. Find optimal lambdas on the validation data
interp_model.find_best_lambdas(val_set)
# 3. Evaluate the final model on the test data
print("\n--- Evaluating Deleted Interpolation Model on Test Set ---")
avg_log_prob_interp = np.mean([interp_model.get_interpolated_log_prob(s) for s in test_set])
print(f"Model: Interpolated Quadrigram, Average Log Probability: {avg_log_prob_interp:.4f}")
print("-" * 50)