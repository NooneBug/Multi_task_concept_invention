The notebooks are divided basing on their purpose, to obtain a proper and correct model which can replicate the experiments of the paper, the notebooks are to be consulted (and surely, runned) in the order showed below:

DATA PREPARATION
- (0.1) wikipedia_Abstracts: download and preprocess a bit the corpus
    - requires the corpus, you can download it like showed in the notebook or from [here](https://drive.google.com/file/d/1bWw0xbd6eWr0AHVMCQv9dH-VuztaGq5W/view?usp=sharing) 
- (0.2) import_elmo_embeddings: align ELMo's vectors with words in corpus, create the datasets
    - requires the elmo vectors, you can generate yourself with an ELMo's implementation or you can download mine from [here]: (https://drive.google.com/drive/folders/1Kpj5du0oDhB6HtLXDsHP6FYoqoUBJhUd?usp=sharing)
    - Vectors can be downloaded in bulk (54.5 GB) or 50+ zip can be downloaded, if you download the 50+ zip you have to recompose with `cat splitted* > elmo_vectors.zip`
    - (0.2.1) composite_words: retrieving of the word phrases
        - Requires the corpus and the elmo vectors
    - (0.2.2) minimal_type: solving of polytipe words (words which are retrieved from more than one class)
                            the polytiping have to be erased to avoid a multilabel problem

BUILD THE NETWORK
- 1 network:
    - requires the vectors of concepts (1.1, 1.2) and the datasets (1.3): 
        1.1 vectors of type2vec that you can find [here](https://drive.google.com/file/d/1S8VKBRI8ThE_lwmeoEts07Rc6N8rFLwB/view?usp=sharing),
        1.2 vectors of HyperE that you can generate with HyperE or you can find [here](https://drive.google.com/file/d/1rRYMXsSVHBcHNTfkIg-TTF_ah4ywb8pe/view?usp=sharing)
        1.3 datasets are available [here](https://drive.google.com/drive/folders/1Kpj5du0oDhB6HtLXDsHP6FYoqoUBJhUd?usp=sharing)
