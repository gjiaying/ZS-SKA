import pickle, json, requests, csv, copy, os, re
import scipy
import numpy as np
import spacy
import collections
  
nlp = spacy.load('en_core_web_lg')


directory = os.listdir('./KG_VECTOR_NYT1')

for i in range(len(directory)):
    objects = []
    with (open(os.getcwd()+"/KG_VECTOR_NYT1/"+directory[i], "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break

    label = directory[i][18:][:-7]

    the_class = {} 
    super_class = {}
    description = {}
    hypernym1 = {}
    hypernym2 = {}
    full = {}
    count = []
    for k, v in objects[0].items():
       # print(sum(v))
        if sum(v[0:10]) > 0:
            the_class[k.split('/')[3]] = sum(v[0:10])
            #print(v)
        if sum(v[10:20]) > 0:
            super_class[k.split('/')[3]] = sum(v[10:20])

        if sum(v[20:30]) > 0:
            description[k.split('/')[3]] = sum(v[20:30])

        if sum(v[30:40]) > 0:
            hypernym1[k.split('/')[3]] = sum(v[30:40])

        if sum(v[40:50]) > 0:
            hypernym2[k.split('/')[3]] = sum(v[40:50])
        full[k.split('/')[3]] = sum(v)
        


    the_class = sorted(the_class.items(), key=lambda x:-x[1])
    super_class = sorted(super_class.items(), key=lambda x:-x[1])
    description = sorted(description.items(), key=lambda x:-x[1])
    hypernym1 = sorted(hypernym1.items(), key=lambda x:-x[1])
    hypernym2 = sorted(hypernym2.items(), key=lambda x:-x[1])
    full = sorted(full.items(), key=lambda x:-x[1])


    result = []
    result_dic = {}
    no_thredshold = {}
    for value in full:
        try:

            final = nlp(value[0]).similarity(nlp(label))
            if final > 0:
                result_dic[value] = final
            no_thredshold[value] = final    
        except:
            continue
    #result_dic = sorted(result_dic.items(),key=lambda x:-x[1])    
    pickle.dump(list(no_thredshold.items()), open(os.getcwd()+ '/data/KG_EM_NYT1/' + 'VECTORS_CLUSTER_1_' + label + ".pickle", "wb"))