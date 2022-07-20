# ZS-SKA
Before running the experiments, please make sure to put all data files directly in our data.zip folder (no sub-folders) in the folder name 'data'.
To run on wiki-zsl dataset, please run wiki_train.py
We follow all the parameter settings the same as in the parameter setting section in the paper appendix. If you are interested in changing the values of parameters, please change them directly in wiki_train.py

To run on FewRel dataset, please run FewRel.py
We follow all the parameter settings the same as in the parameter setting section in the paper appendix. If you are interested in changing the values of parameters, please change them directly in FewRel_train.py

To run on NYT dataset, please change all file parameters in FewRel.py to files of NYT dataset and then directly run FewRel.py.



If you are interested in each module performance, 
For data augmentation, please run data augmentation.py to get augmented sentences.
For knowledge graph, please run conceptNet.py to generate information that used in our prompts construction.
We also provide direct result, please see 'Graphwiki_0.6_10.pickle' and 'Graph_0.5_10.pickle' and 'GraphNYT_0.5_10.pickle' in the data.zip folder.



Due to the file size, we haven't provide intermediate files, so the training process will take more than 15 hours depending on your device. Using saved intermediate files, the training time can reduce to less than 10 hours.
