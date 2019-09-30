The notebooks are divided basing on their purpose, to obtain a proper and correct model which can replicate the experiments of the 
paper the notebook are to be consulted (and surely, runned) in the order showed below:

DATA PREPARATION
- (0.1) wikipedia_Abstracts: download and preprocess a bit the corpus 
- (0.2) import_elmo_embeddings: align ELMo's vectors with words in corpus, create the datasets
    - (0.2.1) composite_words: retrieving of the word phrases
    - (0.2.2) minimal_type: solving of polytipe words (words which are retrieved from more than one class)
                            the polytiping have to be erased to avoid a multilabel problem
BUILD THE NETWORK
- 1. network:
