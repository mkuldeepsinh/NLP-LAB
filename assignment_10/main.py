import numpy as np
from collections import defaultdict, Counter
import random

class HMM_POS_Tagger:
    def __init__(self):
        self.transition_prob = defaultdict(lambda: defaultdict(float))
        self.emission_prob = defaultdict(lambda: defaultdict(float))
        self.initial_prob = defaultdict(float)
        self.tags = set()
        self.vocab = set()
        
    def parse_sentence(self, sentence):
        words = []
        tags = []
        tokens = sentence.strip().split()
        
        for token in tokens:
            if '_' in token:
                parts = token.rsplit('_', 1)
                if len(parts) == 2:
                    word, tag = parts
                    words.append(word)
                    tags.append(tag)
        
        return words, tags
    
    def train(self, sentences):
        tag_counts = Counter()
        transition_counts = defaultdict(Counter)
        emission_counts = defaultdict(Counter)
        initial_counts = Counter()
        
        for sentence in sentences:
            words, tags = self.parse_sentence(sentence)
            
            if len(words) == 0:
                continue
                
            self.tags.update(tags)
            self.vocab.update(words)
            
            initial_counts[tags[0]] += 1
            
            for i in range(len(tags)):
                tag_counts[tags[i]] += 1
                emission_counts[tags[i]][words[i]] += 1
                
                if i > 0:
                    transition_counts[tags[i-1]][tags[i]] += 1
        
        total_initial = sum(initial_counts.values())
        for tag in initial_counts:
            self.initial_prob[tag] = initial_counts[tag] / total_initial
        
        for prev_tag in transition_counts:
            total = sum(transition_counts[prev_tag].values())
            for curr_tag in transition_counts[prev_tag]:
                self.transition_prob[prev_tag][curr_tag] = transition_counts[prev_tag][curr_tag] / total
        
        for tag in emission_counts:
            total = tag_counts[tag]
            for word in emission_counts[tag]:
                self.emission_prob[tag][word] = emission_counts[tag][word] / total
    
    def viterbi(self, words):
        n = len(words)
        tags_list = list(self.tags)
        
        viterbi_matrix = np.zeros((len(tags_list), n))
        backpointer = np.zeros((len(tags_list), n), dtype=int)
        
        for i, tag in enumerate(tags_list):
            init_prob = self.initial_prob.get(tag, 1e-10)
            emit_prob = self.emission_prob[tag].get(words[0], 1e-10)
            viterbi_matrix[i][0] = np.log(init_prob) + np.log(emit_prob)
        
        for t in range(1, n):
            for j, curr_tag in enumerate(tags_list):
                emit_prob = self.emission_prob[curr_tag].get(words[t], 1e-10)
                
                max_prob = float('-inf')
                max_idx = 0
                
                for i, prev_tag in enumerate(tags_list):
                    trans_prob = self.transition_prob[prev_tag].get(curr_tag, 1e-10)
                    prob = viterbi_matrix[i][t-1] + np.log(trans_prob) + np.log(emit_prob)
                    
                    if prob > max_prob:
                        max_prob = prob
                        max_idx = i
                
                viterbi_matrix[j][t] = max_prob
                backpointer[j][t] = max_idx
        
        best_path = [0] * n
        best_path[n-1] = int(np.argmax(viterbi_matrix[:, n-1]))
        
        for t in range(n-2, -1, -1):
            best_path[t] = backpointer[best_path[t+1]][t+1]
        
        return [tags_list[idx] for idx in best_path]
    
    def predict(self, sentence):
        words, _ = self.parse_sentence(sentence)
        
        if len(words) == 0:
            return []
        
        return self.viterbi(words)
    
    def evaluate(self, test_sentences):
        correct = 0
        total = 0
        
        for sentence in test_sentences:
            words, true_tags = self.parse_sentence(sentence)
            
            if len(words) == 0:
                continue
            
            predicted_tags = self.viterbi(words)
            
            for pred, true in zip(predicted_tags, true_tags):
                if pred == true:
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        return accuracy, correct, total


def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        sentences = f.readlines()
    return sentences


def split_data(sentences, train_ratio=0.8):
    random.seed(42)
    random.shuffle(sentences)
    
    split_idx = int(len(sentences) * train_ratio)
    train_data = sentences[:split_idx]
    test_data = sentences[split_idx:]
    
    return train_data, test_data


def main():
    print("Loading data...")
    sentences = load_data('pos_tagdata.txt')
    
    print(f"Total sentences: {len(sentences)}")
    
    print("\nSplitting data (80:20)...")
    train_data, test_data = split_data(sentences, train_ratio=0.8)
    print(f"Training sentences: {len(train_data)}")
    print(f"Testing sentences: {len(test_data)}")
    
    print("\nTraining HMM model...")
    hmm = HMM_POS_Tagger()
    hmm.train(train_data)
    
    print(f"Number of unique tags: {len(hmm.tags)}")
    print(f"Number of unique words: {len(hmm.vocab)}")
    print(f"Tags: {sorted(hmm.tags)}")
    
    print("\nEvaluating on test set...")
    accuracy, correct, total = hmm.evaluate(test_data)
    
    print(f"\nResults:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Correct predictions: {correct}/{total}")
    
    print("\nSample predictions:")
    for i in range(min(3, len(test_data))):
        words, true_tags = hmm.parse_sentence(test_data[i])
        if len(words) > 0:
            predicted_tags = hmm.viterbi(words)
            print(f"\nSentence {i+1}:")
            print(f"Words: {' '.join(words[:10])}...")
            print(f"True tags: {' '.join(true_tags[:10])}...")
            print(f"Predicted: {' '.join(predicted_tags[:10])}...")


if __name__ == "__main__":
    main()
