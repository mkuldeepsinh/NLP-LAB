import collections
import math
import heapq
from datasets import load_dataset
import sys
import codecs
import random

# --- Configuration ---
NUM_SENTENCES_TO_TRAIN = 50000
MAX_GENERATION_LENGTH = 20
VOCAB_MIN_FREQ = 3 
SMOOTHING_ALPHA = 0.01

# Ensure terminal can handle UTF-8 output
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())

class QuadrigramLanguageModel:
    """
    A class to implement a 4-gram language model with Additive Smoothing and backoff.
    """
    def __init__(self, min_freq=3):
        print("Initializing Quadrigram Language Model...")
        self.unigram_counts = collections.defaultdict(int)
        self.bigram_counts = collections.defaultdict(int)
        self.trigram_counts = collections.defaultdict(int)
        self.quadrigram_counts = collections.defaultdict(int)
        self.total_words = 0
        self.vocab = set()
        self.min_freq = min_freq
        self.trigram_follows = collections.defaultdict(set)

    def _preprocess_sentence(self, sentence):
        tokens = sentence.strip().split()
        processed_tokens = [word if word in self.vocab else '<UNK>' for word in tokens]
        return ['<s>'] * 3 + processed_tokens + ['</s>']

    def train(self, dataset_stream):
        print(f"Starting training on {NUM_SENTENCES_TO_TRAIN} sentences...")
        print("Pass 1: Building vocabulary...")
        temp_word_counts = collections.defaultdict(int)
        dataset_iterator = iter(dataset_stream)
        for i in range(NUM_SENTENCES_TO_TRAIN):
            try:
                item = next(dataset_iterator)
                tokens = item['text'].strip().split()
                for token in tokens:
                    temp_word_counts[token] += 1
                if (i + 1) % 5000 == 0:
                    print(f"  ...scanned {i+1}/{NUM_SENTENCES_TO_TRAIN} sentences for vocab.")
            except StopIteration:
                print(f"Warning: Dataset stream ended before reaching {NUM_SENTENCES_TO_TRAIN} sentences.")
                break
        
        self.vocab = {word for word, count in temp_word_counts.items() if count >= self.min_freq}
        self.vocab.add('<UNK>')
        self.vocab.add('<s>')
        self.vocab.add('</s>')
        print(f"Vocabulary size: {len(self.vocab)} words.")

        print("\nPass 2: Counting n-grams...")
        dataset_iterator = iter(dataset_stream)
        for i in range(NUM_SENTENCES_TO_TRAIN):
            try:
                item = next(dataset_iterator)
                tokens = self._preprocess_sentence(item['text'])
                if i == 0: self.total_words += self.unigram_counts['<s>']
                self.total_words += len(tokens) - 3

                for j in range(len(tokens) - 3):
                    w1, w2, w3, w4 = tokens[j], tokens[j+1], tokens[j+2], tokens[j+3]
                    self.unigram_counts[w4] += 1
                    self.bigram_counts[(w3, w4)] += 1
                    self.trigram_counts[(w2, w3, w4)] += 1
                    self.quadrigram_counts[(w1, w2, w3, w4)] += 1
                    self.trigram_follows[(w1, w2, w3)].add(w4)

                if (i + 1) % 5000 == 0:
                    print(f"  ...processed {i+1}/{NUM_SENTENCES_TO_TRAIN} sentences for counts.")
            except StopIteration:
                break
        print("Training complete.")

    def get_prob(self, word, context):
        V = len(self.vocab)
        alpha = SMOOTHING_ALPHA
        quad_context = tuple(context[-3:])
        quad_count = self.quadrigram_counts.get(quad_context + (word,), 0)
        tri_context_count = self.trigram_counts.get(quad_context, 0)
        if tri_context_count > 0:
            return (quad_count + alpha) / (tri_context_count + alpha * V)
        tri_context = tuple(context[-2:])
        tri_count = self.trigram_counts.get(tri_context + (word,), 0)
        bi_context_count = self.bigram_counts.get(tri_context, 0)
        if bi_context_count > 0:
            return (tri_count + alpha) / (bi_context_count + alpha * V)
        bi_context = tuple(context[-1:])
        bi_count = self.bigram_counts.get(bi_context + (word,), 0)
        uni_context_count = self.unigram_counts.get(bi_context[0], 0)
        if uni_context_count > 0:
            return (bi_count + alpha) / (uni_context_count + alpha * V)
        uni_count = self.unigram_counts.get(word, 0)
        return (uni_count + alpha) / (self.total_words + alpha * V)
    
    def generate(self, method, num_sentences=10, **kwargs):
        print(f"Generating {num_sentences} sentences with {method} search...")
        if method == 'greedy':
            return [self._generate_greedy() for _ in range(num_sentences)]
        elif method == 'beam':
            beam_size = kwargs.get('beam_size', 20)
            return [self._generate_beam(beam_size) for _ in range(num_sentences)]
        elif method == 'sampling':
            temperature = kwargs.get('temperature', 1.0)
            top_k = kwargs.get('top_k', 50)
            return [self._generate_sampling(temperature, top_k) for _ in range(num_sentences)]
        else:
            raise ValueError("Method must be 'greedy', 'beam', or 'sampling'")

    def _generate_greedy(self):
        context_list = ['<s>'] * 3
        sentence = []
        for _ in range(MAX_GENERATION_LENGTH):
            context_tuple = tuple(context_list)
            possible_next_words = self.trigram_follows.get(context_tuple, self.vocab)
            if not possible_next_words:
                 possible_next_words = self.vocab
            best_word = None
            max_prob = -1.0
            for word in possible_next_words:
                if word == '<s>': continue
                prob = self.get_prob(word, context_list)
                if prob > max_prob:
                    max_prob = prob
                    best_word = word
            if best_word == '</s>' or best_word is None: break
            sentence.append(best_word)
            context_list = context_list[1:] + [best_word]
        return ' '.join(sentence)

    def _generate_beam(self, beam_size):
        initial_context = ['<s>'] * 3
        beams = [(0.0, initial_context)]
        for _ in range(MAX_GENERATION_LENGTH):
            new_beams = []
            all_finished = True
            for log_prob, words in beams:
                if words[-1] == '</s>':
                    heapq.heappush(new_beams, (log_prob, words))
                    continue
                all_finished = False
                context_tuple = tuple(words[-3:])
                possible_next_words = self.trigram_follows.get(context_tuple, self.vocab)
                if not possible_next_words:
                    possible_next_words = self.vocab
                for word in possible_next_words:
                    if word == '<s>': continue
                    prob = self.get_prob(word, words)
                    if prob == 0: continue
                    new_log_prob = log_prob + math.log(prob)
                    new_words = words + [word]
                    heapq.heappush(new_beams, (new_log_prob, new_words))
            if all_finished: break
            beams = heapq.nlargest(beam_size, new_beams)
        best_log_prob, best_words = beams[0]
        return ' '.join(best_words[3:]).replace('</s>', '').strip()

    def _generate_sampling(self, temperature=1.0, top_k=50):
        context_list = ['<s>'] * 3
        sentence = []
        for _ in range(MAX_GENERATION_LENGTH):
            context_tuple = tuple(context_list)
            possible_next_words = self.trigram_follows.get(context_tuple, self.vocab)
            if not possible_next_words:
                 possible_next_words = self.vocab
            
            word_probs = {word: self.get_prob(word, context_list) for word in possible_next_words if word != '<s>'}

            if top_k > 0 and len(word_probs) > top_k:
                top_k_words = sorted(word_probs.items(), key=lambda item: item[1], reverse=True)[:top_k]
                word_probs = dict(top_k_words)
            
            if temperature != 1.0:
                for word in word_probs:
                    word_probs[word] = word_probs[word] ** (1.0 / temperature)

            total_prob = sum(word_probs.values())
            if total_prob == 0: break
            for word in word_probs:
                word_probs[word] /= total_prob
            
            words, probs = list(word_probs.keys()), list(word_probs.values())
            chosen_word = random.choices(words, weights=probs, k=1)[0]
            
            if chosen_word == '</s>': break
            sentence.append(chosen_word)
            context_list = context_list[1:] + [chosen_word]
        return ' '.join(sentence)

# --- Main Execution ---
if __name__ == "__main__":
    print("Loading Gujarati dataset stream...")
    try:
        streaming_dataset = load_dataset("ai4bharat/IndicCorpV2", "indiccorp_v2", streaming=True, split="guj_Gujr")
        print("Successfully loaded the ai4bharat/IndicCorpV2 Gujarati dataset.")
    except Exception as e:
        print(f"Failed to load dataset. Error: {e}")
        exit()

    model = QuadrigramLanguageModel(min_freq=VOCAB_MIN_FREQ)
    model.train(streaming_dataset)

    # --- GENERATE AND SAVE SENTENCES TO A FILE ---
    
    # 1. Greedy Search
    greedy_sentences = model.generate(method='greedy', num_sentences=100)
    
    # 2. Sampling Search
    sampling_sentences = model.generate(
        method='sampling', 
        num_sentences=100,
        temperature=1.0,
        top_k=50
    )
    
    # 3. Write all results to the output file
    output_filename = "generated_sentences.txt"
    print(f"\nWriting all 200 sentences to {output_filename}...")
    with open(output_filename, 'w', encoding='utf-8') as f:
        # Write Greedy Sentences
        f.write("="*50 + "\n")
        f.write("## 1. Greedy Search (Deterministic)\n")
        f.write("="*50 + "\n")
        for i, s in enumerate(greedy_sentences):
            f.write(f"{i+1}: {s}\n")
        
        f.write("\n\n") # Add space between sections

        # Write Sampling Sentences
        f.write("="*50 + "\n")
        f.write("## 2. Sampling Search (Stochastic, temp=1.0, top_k=50)\n")
        f.write("="*50 + "\n")
        for i, s in enumerate(sampling_sentences):
            f.write(f"{i+1}: {s}\n")

    print(f"Successfully saved output to {output_filename}")