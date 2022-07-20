from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import math
import nltk
import torch
import torch.nn as nn
import torch.optim as optim
import spacy
import os
import pickle
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from EncoderCNN import EncoderCNN
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from nltk.tokenize.treebank import TreebankWordDetokenizer
from rake_nltk import Rake
from nltk.corpus import wordnet as wn
from gensim.summarization import keywords

spacy_nlp = spacy.load('en_core_web_sm')
if torch.cuda.is_available():
    spacy.prefer_gpu()
r = Rake()
word2vec_output_file = 'glove.6B.50d.word2vec.txt'
glove_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
#sentence_model = SentenceTransformer('paraphrase-mpnet-base-v2')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
with (open(os.getcwd()+"/data/Graphwiki_0.6_10.pickle", "rb")) as openfile:
    prompt = pickle.load(openfile)
    
MAX_LENGTH = 128
HIDDEN_SIZE = 768 #bert embedding
HIDDEN_DIM = 300 #hidden size after CNN
NEW_DIM = 300
batch_size = 4
num_workers = 0
learning_rate = 1e-2
WEIGHT_DECAY = 0
output = []
labels = []
EPOCH = 500
LABEL = 'wiki_labels.csv'
TRAIN_FILE = 'wiki_all_train.txt'
TEST_FILE = 'wiki_test_m15unseen.txt'

class EmbeddingBERT:
    def __init__(self, file, max_length, hidden_size):
        self.file = file
        self.max_length = max_length
        self.hidden_size = hidden_size

    def bert_text_preparation(text, tokenizer):
        marked_text = "[CLS] " + text + " [SEP]"
        tokenized_text = tokenizer.tokenize(marked_text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1]*len(indexed_tokens)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        return tokenized_text, tokens_tensor, segments_tensors

    def get_bert_embeddings(tokens_tensor, segments_tensors, model):

        # Gradient calculation id disabled
        # Model is in inference mode
        with torch.no_grad():
            outputs = model(tokens_tensor, segments_tensors)
            # Removing the first hidden state
            # The first state is the input state
            hidden_states = outputs[2][1:]

        # Getting embeddings from the final BERT layer
        token_embeddings = hidden_states[-1]
        # Collapsing the tensor into 1-dimension
        token_embeddings = torch.squeeze(token_embeddings, dim=0)
        # Converting torchtensors to lists
        list_token_embeddings = [token_embed.tolist() for token_embed in token_embeddings]

        return list_token_embeddings

    def get_embedding(file, max_length, hidden_size, flag):
        s = open(file,"r",encoding="utf-8")
        finaltext, finallabel = [], []
        while 1:
            line = s.readline()
            if not line:
                break
            try:
                line1 = eval(line)
            except:
                continue
            text = line1['token']#text for NYT
            text = TreebankWordDetokenizer().detokenize(text)#detokenize
            label = line1['label']
            
            tokenized_text, tokens_tensor, segments_tensors = EmbeddingBERT.bert_text_preparation(text, tokenizer)
            list_token_embeddings = EmbeddingBERT.get_bert_embeddings(tokens_tensor, segments_tensors, model)
            padded_array = np.zeros((max_length, hidden_size))
            
            if len(list_token_embeddings) > 100:
                list_token_embeddings = list_token_embeddings[0:MAX_LENGTH]
            shape = np.shape(list_token_embeddings)    
            padded_array[:shape[0],:shape[1]] = list_token_embeddings
            
            finaltext.append(padded_array)
            
            finallabel.append(float(label))
        return finaltext, finallabel

def get_acc(pred, y):
    y1 = y.cpu().tolist() #ground truth
    accuracy = accuracy_score(y1, pred)
    recall = recall_score(y1, pred, average='weighted')
    precision = precision_score(y1, pred, average='weighted')
    f1 = f1_score(y1, pred, average='weighted')
    return accuracy, recall, precision, f1

def get_prf(predicted_idx, gold_idx, i=-1, empty_label=None):
    gold_idx = gold_idx.cpu().tolist()
    accuracy = accuracy_score(gold_idx, predicted_idx)
    if i == -1:
        i = len(predicted_idx)

    complete_rel_set = set(gold_idx) - {empty_label}
    avg_prec = 0.0
    avg_rec = 0.0

    for r in complete_rel_set:
        r_indices = (predicted_idx[:i] == r)
        tp = len((predicted_idx[:i][r_indices] == gold_idx[:i][r_indices]).nonzero()[0])
        tp_fp = len(r_indices.nonzero()[0])
        tp_fn = len((gold_idx == r).nonzero()[0])
        prec = (tp / tp_fp) if tp_fp > 0 else 0
        rec = tp / tp_fn
        avg_prec += prec
        avg_rec += rec
    f1 = 0
    avg_prec = avg_prec / len(set(predicted_idx[:i]))
    avg_rec = avg_rec / len(complete_rel_set)
    if (avg_rec+avg_prec) > 0:
        f1 = 2.0 * avg_prec * avg_rec / (avg_prec + avg_rec)

    return accuracy, avg_rec, avg_prec, f1

def weights_calculation(output, labels):
    list_label_num = []
    list_ave_weight = []
    label_unique = set(labels)
   
    for label_data in label_unique:
        ave = np.zeros(shape=(1, NEW_DIM))
        total = 0
    
        for i in range(len(labels)):
            if labels[i] == label_data:
                total = total + 1
                ave = np.sum(([ave, output[i]]), axis = 0)
                
        list_label_num.append(label_data)
        list_ave_weight.append(ave/total)
        
    return list_label_num, list_ave_weight

def get_feature_embedding(file):
    s = open(file,"r",encoding="utf-8")
    finaltext = []
    while 1:
        line = s.readline()
        if not line:
            break
        try:
            line1 = eval(line)
        except:
            continue
        text = line1['token']#text for NYT, token for fewrel
        text = TreebankWordDetokenizer().detokenize(text)#detokenize
        keyword = keywords(text).split("\n")[0]
        
        keyword_embedding = oovcheck_t((keyword).lower())
        
        if wn.synsets(line1['h']) == []:
            head = "None"
        else:
            heads = wn.synsets(line1['h'])[0]
            if len(heads.hypernyms()) == 0:
                head = "None"
            for i in range(len(heads.hypernyms())):
                head = heads.hypernyms()[i].name().split('.')[0]

        head_embedding = oovcheck_t((head).lower())
        
        if wn.synsets(line1['t']) == []:
            tail = "None"
        else: 
            tails = wn.synsets(line1['t'])[0]
            if len(tails.hypernyms()) == 0:
                tail = "None"
            for i in range(len(tails.hypernyms())):
                tail = tails.hypernyms()[i].name().split('.')[0]
        tail_embedding = oovcheck_t((tail).lower())              
        #prompt = head + " is " + keyword + " of " + tail         
        finaltext.append((torch.cat((head_embedding, keyword_embedding, tail_embedding), 0)).detach().numpy())
    return finaltext

def get_test_embedding(model_u, loader):
    out_embedding = []
    ys = []
    for x, y in loader:
        ys.append(y.cuda())
       
        for i in range(batch_size):
            try:
                out_embedding.append(model_u((x.permute(0,2,1)).cuda()).cpu().detach().numpy()[i])
                
            except:
                continue
    y = torch.cat(ys, dim=0)
    '''
    for i in range(len(out_embedding)):
        out_embedding[i] = np.append(out_embedding[i], featest_embedding[y[i]-1])#-1
    '''
    return y, out_embedding


def get_prompt_embedding(label):
    head_embedding = oovcheck(hypernym1[label])
    labels = []
    weights = []
    for k in range(len(prompts_graph[label_description[label]])):
        if k >= 4:
            break
        label_embedding = oovcheck(prompts_graph[label_description[label]][k][0][0])
        weight = prompts_graph[label_description[label]][k][0][1]#[k][0][1]
        labels.append(label_embedding)
        weights.append(weight)
    
    labels = np.array(labels)
    weights = np.array(weights)
    sum_weights = np.sum(weights)
    weights = weights/float(sum_weights)
    key_embedding = np.zeros((len(weights), 50))
    for i in range(len(weights)):
        key_embedding[i] = labels[i] * weights[i]
    graph_embedding = np.sum((key_embedding),axis=0)
    tail_embedding = oovcheck(hypernym2[label])            
    prompt_embedding = np.concatenate((head_embedding, graph_embedding,tail_embedding),axis = None)            
    return prompt_embedding

def softmax(x):
    exp_x = x
    sm = exp_x/np.sum(exp_x, axis=-1, keepdims=True)
    return sm

def distance_calculation(truth_labels, test_embedding, list_label_nums, list_ave_weights):
    label_list = []
    for i in range(len(Y_test)):
        dist_list = []
        for j in range(len(list_ave_weights)):
            dist = np.linalg.norm(test_embedding[i] - list_ave_weights[j])
            dist1 = math.exp(-dist)
            dist_list.append(dist1)
   
        index = np.argmax(softmax(dist_list))
        label_list.append(list_label_nums[index])

    return get_acc(label_list, truth_labels)


def oovcheck(word):
    try:
        oov = glove_model[word]
    except:
        oov = np.random.rand(50)
    return oov

def oovcheck_t(word):
    try:
        oov = torch.from_numpy(glove_model[word])
    except:
        oov = torch.from_numpy(np.random.rand(50)).type(torch.FloatTensor)
    return oov

def get_model_acc(model_u, loader):
    ys = []
    y_preds = []
    for x, y in loader:
        ys.append(y.cuda())
        y_preds.append(torch.argmax(model_u((x.permute(0,2,1)).cuda()), dim=1))
    y = torch.cat(ys, dim=0)
    y_pred = torch.cat(y_preds, dim=0)
    accuracy = accuracy_score(y.cpu(), y_pred.cpu())
    recall = recall_score(y.cpu(), y_pred.cpu(), average='weighted')
    precision = precision_score(y.cpu(), y_pred.cpu(), average='weighted')
    f1 = f1_score(y.cpu(), y_pred.cpu(), average='weighted')
    return accuracy, recall, precision, f1

df = pd.read_csv(os.getcwd()+ '/labels/' + LABEL)
#labelsdf = df['ClassLabel'].to_numpy()
label_description = df['ClassLabel'].to_numpy()
hypernym1 = df['Hypernym1'].to_numpy()
hypernym2 = df['Hypernym2'].to_numpy()
text_description = df['ClassDescription'].to_numpy()
with (open(os.getcwd()+"/data/Graphwiki_0.6_10.pickle", "rb")) as openfile:
    prompts_graph = pickle.load(openfile)


X_train, Y_train = EmbeddingBERT.get_embedding(os.getcwd()+ '/data/' + TRAIN_FILE, MAX_LENGTH, HIDDEN_SIZE, True)
#np.save('wikiXembeddings.npy', X_train)
#np.save('wikiYembeddings.npy', Y_train)
#X_train = np.load('wikiXembeddings.npy')
#Y_train = np.load('wikiYembeddings.npy')
X_test, Y_test = EmbeddingBERT.get_embedding(os.getcwd()+ '/data/' + TEST_FILE, MAX_LENGTH, HIDDEN_SIZE, False)


X_train = torch.FloatTensor(X_train)
Y_train = torch.LongTensor(Y_train)
X_test = torch.FloatTensor(X_test)
Y_test = torch.LongTensor(Y_test)
train_dataset = TensorDataset(X_train, Y_train)
test_dataset = TensorDataset(X_test, Y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

model_encoder = EncoderCNN(MAX_LENGTH, HIDDEN_SIZE, HIDDEN_DIM)
if torch.cuda.is_available():
    model_encoder = model_encoder.cuda()
criterion = nn.CrossEntropyLoss()  #nn.CrossEntropyLoss()
optimizer = optim.SGD(model_encoder.parameters(), lr=learning_rate, weight_decay = WEIGHT_DECAY)

for e in range(EPOCH):

    loss_epoch = 0
    model_encoder.train()
    
    for x, y in train_loader:
        optimizer.zero_grad()
        y = y.view(-1,1).cuda()
        y_pred = model_encoder((x.permute(0,2,1)).cuda())        
   
        loss = criterion(y_pred, y.squeeze(1))
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()
        
        for i in range(batch_size):
              try:
                  output.append(y_pred[i].cpu().detach().numpy())
                  labels.append(y.cpu().squeeze(1)[i].item())
              except:
                  continue
        
    print(loss_epoch)

#torch.save(model.state_dict(), "Wiki_Augmentation.pth") 

#model.load_state_dict(torch.load('Wiki_Augmentation.pth'))
model_encoder.eval()
list_label_nums, list_ave_weights = weights_calculation(output, labels)
#featest_embedding = get_feature_embedding(os.getcwd()+ '/data/' + 'wiki_test_40unseen.txt')

truth_labels, test_embedding = get_test_embedding(model_encoder, test_loader)

print ("Testing Accuracy:", distance_calculation(truth_labels, test_embedding, list_label_nums, list_ave_weights))

#print ("Testing Accuracy:", get_model_acc(model_encoder, test_loader))
          