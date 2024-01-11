from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from scipy.special import softmax
import re
import pickle
import time
import nltk
from nltk.tokenize import sent_tokenize
# Load model and tokenizer
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess(text):
    # Preprocess text (replace usernames and links)
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def analyze_sentiment(text):
    # Analyze the sentiment of the input text
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    return scores
    # # Print the labels and scores
    # ranking = np.argsort(scores)
    # ranking = ranking[::-1]
    
    # for i in range(scores.shape[0]):
    #     l = model.config.id2label[ranking[i]]
    #     s = scores[ranking[i]]
    #     print(f"{i+1}) {l} {np.round(float(s), 4)}")
    
def split_sentences(text):
    nltk.download('punkt')
    sentences = sent_tokenize(text)
    sentences = [sentence.strip().replace("   "," ").replace("  ", " ").replace("\n", " ") for sentence in sentences]
    return sentences
    # # Regular expression to find sentences inside quotes

    # quote_pattern = r'"[^"]*"'

    # # List to store the final split sentences
    # sentences = []

    # # Temporary text where quotes are replaced with placeholders
    # temp_text = text
    # quote_placeholders = []
    # for i, quote in enumerate(re.findall(quote_pattern, text)):
    #     placeholder = f"__QUOTE{i}__"
    #     quote_placeholders.append((placeholder, quote))
    #     temp_text = temp_text.replace(quote, placeholder)

    # # Splitting the text without quotes into sentences
    # split_text = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!) +', temp_text)

    # # Replacing the placeholders with the original quotes
    # for sentence in split_text:
    #     for placeholder, quote in quote_placeholders:
    #         sentence = sentence.replace(placeholder, quote)
    #     sentence = sentence.strip()
    #     sentence = sentence.replace('  ', ' ')
    #     sentences.append(sentence)
    # return sentences

def remove_preface_and_chapters(text):
    text = text.replace("\n", " ")
    text = text.replace("  ", " ")
    pattern = r'\bCHAPTER\s+\w+\b'
    
    # Replace the matched patterns with an empty string
    cleaned_text = re.sub(pattern, '', text)
    cleaned_text = re.sub(r'THE PREFACE', '', cleaned_text)
    cleaned_text = re.sub(r'OSCAR WILDE', '', cleaned_text)
    return cleaned_text
# Example usage


def long_sentences(sentence):
    num_of_spaces = 200
    sentences = []
    curr_sentence = ''

    for word in sentence.split(' '):
        if curr_sentence.count(' ') < num_of_spaces:
            curr_sentence += ' ' + word
        elif curr_sentence.count(' ') >= num_of_spaces:
            sentences.append(curr_sentence)
            curr_sentence = ''
        else:
            raise Exception('Something went wrong')
    sentences = [sentence.strip() for sentence in sentences]
    return sentences
def eval_long_sentences(sentence):
    sentences = long_sentences(sentence)
    weights = [sentence.count(' ') for sentence in sentences]
    sentiments = [analyze_sentiment(sentence) for sentence in sentences]
    sentiments = np.array(sentiments)
    weighted_means = np.average(sentiments, axis=0, weights=weights)
    return weighted_means
def analyze_book(text):
    sentiments = []
    total = len(text)
    start_time = time.time()
    for index, sentence in enumerate(text):
        try:
            sentiment = analyze_sentiment(sentence)
            curr = (time.time() - start_time) * (total/(index+1) - 1)
            print(f"{index} / {total} with {curr} seconds left is negative:{sentiment[0]} | neutral:{sentiment[1]} | positive:{sentiment[2]}")
            sentiments.append(sentiment)

        except:
            
            sentiment = eval_long_sentences(sentence)
            
            curr = (time.time() - start_time) * (total/(index+1) - 1)
            print(f"{index} / {total} with {curr} seconds left is negative:{sentiment[0]} | neutral:{sentiment[1]} | positive:{sentiment[2]}")
            sentiments.append(sentiment)
    return sentiments

def save_book_to_pickle(text):
    with open('sentiments.pickle', 'wb') as handle:
        pickle.dump(text, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def weighted_avg_sent(sentences, sentiments):
    sentiments = np.array(sentiments)
    weights = [sentence.count(' ') for sentence in sentences]
    weighted_means = np.average(sentiments, axis=0, weights=weights)
    return weighted_means

def weight_func(leng):
    return leng**3/(leng**3 + 500)
def book_lowest(sentences, sentiments, count = 30):
    sentiments = np.array([weight_func(sentence.count(' ')) * sentiment for sentence, sentiment in zip(sentences, sentiments)])
    lowest_indexes = np.argsort(sentiments[:,0])[-count:]
    worst_sentences = [sentences[index] for index in lowest_indexes]
    return worst_sentences, lowest_indexes
def book_highest(sentences, sentiments, count = 30):
    sentiments = np.array([weight_func(sentence.count(' ')) * sentiment for sentence, sentiment in zip(sentences, sentiments)])
    highest_indexes = np.argsort(sentiments[:,2])[-count:]
    best_sentences = [sentences[index] for index in highest_indexes]
    return best_sentences, highest_indexes
def read_book_and_get_sentences(file_name):
    file = open(file_name, "r")
    file = file.read()
    file = remove_preface_and_chapters(file)
    sentences = split_sentences(file)
    return sentences

def take_list_and_sort_by_location(sentences, sentiments, indexes):
    sentiments = np.array(sentiments)
    sorted_indexes = sorted(indexes)
    sorted_subset_sentences = [sentences[index] for index in sorted_indexes]
    return sentences, sorted_indexes

def find_closest_sentences(sentence_1_ind, sentence_2_ind):
        # Initialize pointers and variables to store the minimum difference and pairs
    pairs = []
    used_indices = set()
    list1 = sentence_1_ind
    for value1 in sentence_1_ind:
        closest_value = None
        closest_diff = float('inf')
        closest_index = -1

        for i, value2 in enumerate(sentence_2_ind):
            if i not in used_indices:
                diff = abs(value1 - value2)
                if diff < closest_diff:
                    closest_diff = diff
                    closest_value = value2
                    closest_index = i

        # Mark the index as used and store the pair
        used_indices.add(closest_index)
        pairs.append((value1, closest_value))
    pairs = [pair for pair in pairs if abs(pair[0] - pair[1]) < 30]
    return pairs
def take_pairs_and_cross_ref(key, sentence_1, sentence_2):
    out = []
    for pair in key:
        out.append([sentence_1[pair[0]], sentence_2[pair[1]]])
    return out

def deconstructionist_prompt(sentence_1, sentence_2):
    base = f"Create a photorealistic image that represents a deconstruction of the binaries between these two ideas from *The Picture Dorian Gray* by Oscar Wilde:\n1. {sentence_1}\n2. {sentence_2}\nThink about breaking the binaries that exist between these two ideas."
    return base
def formalist_prompt(sentence_1, sentence_2):
    base = f"Create a photorealistic image that represents the binaries between these two ideas from *The Picture Dorian Gray* by Oscar Wilde: \n1. {sentence_1}\n2. {sentence_2}\nThink about reinforcing and representing the binaries that exist between these two ideas."
    return base


if __name__ == "__main__":
    
    file = "The Picture of Dorian Gray.txt"
    sentences = read_book_and_get_sentences(file)
    # sentiments =  analyze_book(sentences)
    # save_book_to_pickle(sentiments)
    analysis = pickle.load(open('sentiments.pickle', 'rb'))
    
    best, best_ind = book_highest(sentences, analysis, count = 50)
    worst, worst_ind = book_lowest(sentences, analysis, count = 50)
    
    best_1, best_1_ind = take_list_and_sort_by_location(sentences, best, best_ind)
    best_2, best_2_ind = take_list_and_sort_by_location(sentences, worst, worst_ind)
    pairs = find_closest_sentences(best_1_ind, best_2_ind)
    sentence_pairs = take_pairs_and_cross_ref(pairs, best_1, best_2)
    print("Best and Worst Sentences:")
    for sentence_ind, sentences in enumerate(sentence_pairs):
        # print(f"Best: {sentences[0]}\nWorst: {sentences[1]}", "\n")
        decon = deconstructionist_prompt(sentences[0], sentences[1])
        form = formalist_prompt(sentences[0], sentences[1])
        print(sentence_ind)
        print(decon, form, "\n", sep="\n\n")