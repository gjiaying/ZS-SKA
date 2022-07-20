import pickle, json, requests, csv, copy, os, re

THRESHOLD = 0.5
NUMBER = 10
directory = os.listdir('./data/KG_EM_NYT1')
result = {}
for i in range(len(directory)):
    
    objects = []
    with (open(os.getcwd()+"/data/KG_EM_NYT1/"+directory[i], "rb")) as openfile:
        objects.append(pickle.load(openfile))
        label = directory[i][18:][:-7]
        
        final = []
        
        for i in objects[0]:
            if i[1] > THRESHOLD:
                final.append(i)

        result[label] = final[0:NUMBER]
        
       # print(final[0:10][0][0])
    
#print(result)       
pickle.dump(result, open(os.getcwd()+"/data/"+"GraphNYT_"+str(THRESHOLD)+"_"+str(NUMBER)+".pickle", "wb"))
