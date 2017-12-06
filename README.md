# Sentimental Analysis on IMDB movie reviews -- Run Report

* UKY MA 721 course project 2
* Author: Jiho Noh (jiho.noh@uky.edu)

## Description
This project is to implement two different types of neural network for 
a document classification task: Convolutional Neural Network (CNN) and 
Recurrent Neural Network (RNN, here I used Gated Recurrent Units).

### Dataset
The [Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/) 
is for binary sentimental classification problem. It consists of 25,000 movie
reviews for training and another 25,000 for testing. Each dataset is evenly 
divided into two groups (pos and neg) based on the number of stars (out of 10)
that the reviewer assigned. 

#### document lengths
The average length of documents (in words) is 1,310. Depending on the 
tokenizer and the vocabulary of the word embeddings, the actual number of 
words that will be fed into the network will range between 700~900, since 
numeric literals, special characters, OOV (out of vocabulary) terms will not
be considered.  

#### vocabulary in word representations
The number of unique words identified by [FastTest Wikipedia](http://bit.ly/2Bs7zqh) word embeddings is 84,546. I've also tried with a different word representations, [GloVe](https://nlp.stanford.edu/projects/glove/), and there is no significant difference in results.

#### example review documents
*Negative Review*  
  
    This movie is terrible. Carlitos Way(1993) is a great film. Goodgfellas it 
    isn't but its one of the better crime films done. This movie should be 
    considered closer to THE STING Part2 or maybe speed Zone. Remember those gems! 
    The only reason this movie was made was to capitalize on the cult following of 
    the original. This movie lacked everything De Palma, Pacino and Penn worked so 
    hard on. There wasn't a likable character and that is the fault of everyone 
    responsible for making it. I hope RISE TO POWER wins every RAZZIE it possibly 
    can and maybe even invent some new categories to allow it be a record holder. 
    After I watched this S@*T FEST movie, I sat down and watched the original 
    Carlitos way to get th bad taste out of my mouth. After watching this I wish 
    Pachanga came and whacked me out of my misery.
    
*Positive Review*  

    This is a bizzare look at Al's "life", back when he still a hyper 
    20-something. The (real) home videos of Al as a kid are great, and the 
    commentary from his (real life) parents gives a nice glimpse of just how 
    Weird Al wound up as screwed up as he is. This video is a must own for any 
    devoted Al-coholic.
    

## Model Architecture

### Convolutional Neural Network (CNN)
The CNN model resembles the implementation of Kim's paper, [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882). Arguably, this is the most popular model for text classification using CNN. It has three different kernel sizes (3, 4, 5), with the depth of 128. That is, it applies convolutions on 3, 4, or 5 word representations. Max pooling follows right after that. 

![CNN model for Text Classification](http://bit.ly/2BqB7Vg)  
*Figure. CNN model for text classification (image from http://www.wildml.com/)*

Parameters are listed as below:


**Parameters of CNN**

*original*
```
CnnImdbSA (
  (encoder): Embedding(7643, 300, padding_idx=0), weights=((7643, 300),), parameters=2292900
  (convs1): ModuleList (
    (0): Conv2d(1, 128, kernel_size=(3, 300), stride=(1, 1))
    (1): Conv2d(1, 128, kernel_size=(4, 300), stride=(1, 1))
    (2): Conv2d(1, 128, kernel_size=(5, 300), stride=(1, 1))
  ), weights=((128, 1, 3, 300), (128,), (128, 1, 4, 300), (128,), (128, 1, 5, 300), (100,)), parameters=461184
  (dropout): Dropout (p = 0.5), weights=(), parameters=0
  (decoder): Linear (300 -> 2), weights=((2, 300), (2,)), parameters=602
)
```
*minimized version for best ratio*
```
CnnImdbSA (
  (encoder): Embedding(84546, 300, padding_idx=0), weights=((84546, 300),), parameters=25363800
  (convs1): ModuleList(
    (0): Conv2d (1, 32, kernel_size=(3, 300), stride=(1, 1))
    (1): Conv2d (1, 32, kernel_size=(5, 300), stride=(1, 1))
  ), weights=((32, 1, 3, 300), (32,), (32, 1, 5, 300), (32,)), parameters=76864
  (dropout): Dropout(p=0.5), weights=(), parameters=0
  (decoder): Linear(in_features=64, out_features=2), weights=((2, 64), (2,)), parameters=130
)
```
Total number of parameters (excluding the embedding layer): 461,786

### Recurrent Neural Network (RNN)
For the recurrent neural network, I've experimeted with Gate Recurrent Units (GRU) model. 
![Gate Recurrent Units](http://bit.ly/2Brzerj)  
*Figure. Gate Recurrent Units (image from http://colah.github.io/posts/2015-08-Understanding-LSTMs/)*

It has four learnable weight attributes: 

- weight\_input\_hidden (3 * hidden\_size x input\_size)
- weight\_hidden\_hidden (3 * hidden\_size x hidden\_size)
- bias\_input\_hidden (3 * hidden\_size)
- bias\_hidden\_hidden (3 * hidden\_size)

Parameters are listed as below:

**Parameters of RNN (GRU)**  

```
  GruImdbSA (
      (encoder: Embedding(84546, 300), weights=((84546, 300),), parameters=25363800
      (gru): GRU(300, 128), weights=((384, 300), (384, 128), (384,), (384,)), parameters=165120
      (decoder): Linear (128 -> 2), weights=((2, 128), (2,)), parameters=258
  )
```

Total number of parameters (excluding the embedding layer): 165,378

## Additional Work (and future work)
### NLP annotation tags
The model is trained in word-based tokens. Each token can be further characterized by additional attributes, such as annotation tags including POS (part of speech) and NER (named entity recognition). Providing additional features like these NLP tags will increase the efficiency of conveying the semantics of the constituent words. 
I have preprocessed the dataset with [spaCy](https://spacy.io/) tokenizer by which each token has been associated with the known POS and NER tags.

```
context: "i", "was", "looking", "forward", "to", "kathryn", "bigelow", "'s", "movie", "with", "great", "anticipation"
ner: "", "", "", "", "", "PERSON", "PERSON", "PERSON", "", "", "", ""
pos: "PRON", "VERB", "VERB", "ADV", "ADP", "PROPN", "PROPN", "PART", "NOUN", "ADP", "ADJ", "NOUN"
```
Considering the dimensions of word embeddings (mostly 300), adding these two 
feature vectors is worth to try. The entire 35 identified new features are 
listed as below:

```
[ {'pos=DET': 0, 'pos=PROPN': 1, 'pos=ADV': 2, 'pos=PUNCT': 3, 'pos=ADP': 4, 
   'pos=NOUN': 5, 'pos=VERB': 6, 'pos=ADJ': 7, 'pos=CCONJ': 8, 'pos=PRON': 9, 
   'pos=PART': 10, 'pos=SPACE': 11, 'pos=INTJ': 12, 'pos=NUM': 13, 'pos=X': 14, 
   'pos=SYM': 15, 'ner=': 16, 'ner=ORG': 17, 'ner=FAC': 18, 'ner=PERSON': 19, 
   'ner=ORDINAL': 20, 'ner=CARDINAL': 21, 'ner=DATE': 22, 'ner=LOC': 23, 
   'ner=GPE': 24, 'ner=WORK_OF_ART': 25, 'ner=NORP': 26, 'ner=TIME': 27, 
   'ner=PRODUCT': 28, 'ner=LANGUAGE': 29, 'ner=QUANTITY': 30, 'ner=MONEY': 31, 
   'ner=EVENT': 32, 'ner=LAW': 33, 'ner=PERCENT': 34} ]
```

I have not implemented the component utilizing this additional feature.

### Interactive Prediction
`/scripts/eval.py` is a script to predict sentiments of any arbitrary text 
review. It uses given pre-trained model with the best weight states and word 
dictionary. User can type a review and pass it into `process()`, then it will
 return the scores and the predicted label.

```
12/06/2017 10:06:22 PM: [ loading trained model from 1512597440-best.mdl ]
12/06/2017 10:06:22 PM: [ vocab_size: 84546, embedding_dim: 300 ]

Sentimental Analysis on IMDb Movie Reviews
>> ex = "This movie is full of entertaining."
>> process(ex)

>>> ex = """Sex,Drugs,Rock & Roll is without a doubt the worst product of 
Western Civilization. The monologues are both uninteresting and pointless In 
the rare monologue that captures the audience's attention it is quickly lost 
through overly long repetition and unnecessary additions (The Hells Angels at 
McDonalds comes to mind) I guess Bogosian's one man show needed some filler m
aterial to give a length that he thought justified the price of admission.
<br /><br />I would rather sleep with my aunt and be hung upside down and 
drained of my blood than see Sex,Drugs,Rock & Roll again."""

>>> process(ex)

Variable containing:
 0.3656 -1.9171
[torch.FloatTensor of size 1x2]

predicted label: negative
```

## Results

| <sub>network | <sub>capacity (#parameters) | <sub>batch\_size | <sub>optimizer | <sub>regularization | <sub>accuracy (vl/ts) | <sub>best ratio (acc. to computation) | <sub>best accuracy (vl/ts) |
|:--------:|:---------------------------------:|:----------:|:--------:|:--------------:|:--------:|:--------------------:|:-------------:|
| <sub>CNN | <sub>76,994 | <sub>10 | <sub>Adamax (lr=2e-3) | <sub>L2 (decay=0) | <sub>0.778 / 0.856 | <sub>0.278 (1 epoch) | <sub>1.000 / 0.999 (48 epochs) |
| <sub>GRU | <sub>70,402 | <sub>10 | <sub>Adamax (lr=2e-3) | <sub>L2 (decay=0) | <sub>0.811 / 0.873 | <sub>0.286 (1 epoch) | <sub>0.999 / 0.999 (26 epochs) |

![Loss/Accuracy Plot of CNN model](https://github.com/romanegloo/17-ma721_IMDB_CNN_GRU/blob/master/log/plot-cnn-1330.png?raw=true)  
*Figure. Loss/Accuracy Plots of CNN model*

![Loss/Accuracy Plot of RNN model](https://github.com/romanegloo/17-ma721_IMDB_CNN_GRU/blob/master/log/plot-rnn-021324.png?raw=true)  
*Figure. Loss/Accuracy Plots of RNN model*


