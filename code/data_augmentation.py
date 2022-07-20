from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
import nltk
from nltk.corpus import stopwords
import os
import json
nltk.download('averaged_perceptron_tagger')

stop_words = set(stopwords.words('english'))
pos_dict = {'JJ': 'a', 'JJR': 'a', 'JJS': 'a', 'NN': 'n', 'NNP': 'n', 'NNPS': 'n', 'NNS': 'n', 'RB': 'r', 'RBR': 'r', 'RBS': 'r', 'VB': 'v', 'VBD': 'v', 'VBG': 'v', 'VBN': 'v', 'VBP': 'v', 'VBZ': 'v'}
WORD_TOPIC_TRANSLATION = dict()
POS_OF_WORD = dict()


glove_file = 'glove.6B.300d.txt'
word2vec_glove_file = "glove.6B.300d.word2vec.txt"
SOURCE_FILE = 'NYT_zg.txt'
TARGET_FILE = 'augmentation.txt'
LABEL = '50' # Label Number
from_class = "nationality" 
to_class = "location"

glove_model = KeyedVectors.load_word2vec_format(word2vec_glove_file)

    


def pos_list_of(word):
    global POS_OF_WORD
    if word not in POS_OF_WORD:
        POS_OF_WORD[word] = [ss.pos() for ss in wn.synsets(word)]
    return POS_OF_WORD[word]


def word_list_translation(word, from_class, to_class):
    global WORD_TOPIC_TRANSLATION
    word = word.lower()
    key = from_class+'-'+to_class
    if key not in WORD_TOPIC_TRANSLATION:
        WORD_TOPIC_TRANSLATION[key] = dict()
    if word not in WORD_TOPIC_TRANSLATION[key]:
        WORD_TOPIC_TRANSLATION[key][word] = [x[0] for x in glove_model.most_similar_cosmul(positive=[to_class, word], negative=[from_class], topn = 10)]
    return WORD_TOPIC_TRANSLATION[key][word]




s = open(os.getcwd()+ '/data/' + SOURCE_FILE)
while 1:
    line = s.readline()
    if not line:
        break
    line1 = eval(line)
    text = line1['text']
    #text = TreebankWordDetokenizer().detokenize(text)
    original_tokens = word_tokenize(text)
    pos_original_tokens = nltk.pos_tag(original_tokens)
    transferred_tokens = []
    replace_dict = dict()

    for token in pos_original_tokens:
        if token[0].lower() in stop_words or token[1] not in pos_dict or token[0].lower() not in glove_model.vocab:
            transferred_tokens.append(token[0])
        elif token[0].lower() in replace_dict:
            replacement = replace_dict[token[0].lower()]
            if token[0][0].lower() != token[0][0]:
                replacement = replacement[0].upper() + replacement[1:]
            transferred_tokens.append(replacement)

        else:
            candidates = word_list_translation(token[0].lower(), from_class, to_class)
            find_replacement = False
            for cand in candidates:
                if pos_dict[token[1]] in pos_list_of(cand) and cand not in replace_dict.values():
                    replacement = cand
                    replace_dict[token[0].lower()] = cand
                    if token[0][0].lower() != token[0][0]:
                        replacement = replacement[0].upper() + replacement[1:]
                    transferred_tokens.append(replacement)
                    find_replacement = True
                    break
            if not find_replacement:
                transferred_tokens.append(token[0])
    #new_sentence = TreebankWordDetokenizer().detokenize(transferred_tokens)        
    #print(new_sentence)
    details = {"token":transferred_tokens, "label":LABEL}
    #print(transferred_tokens)
    
    with open(os.getcwd()+ '/data/' + TARGET_FILE, 'a+') as convert_file:
        convert_file.write(json.dumps(details))
        convert_file.write('\n')
    