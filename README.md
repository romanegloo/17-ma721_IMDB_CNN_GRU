# Sentimental Analysis on IMDB movie reviews

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

Parameters are listed as below:


**Parameters of CNN**

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
Total number of parameters (excluding the embedding layer): 461,786

### Recurrent Neural Network (RNN)
For the recurrent neural network, I've experimeted with Gate Recurrent Units (GRU) model. It has four learnable weight attributes:
![Gate Recurrent Units](http://bit.ly/2Brzerj)

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

## Results

| network | capacity (#parameters) | batch\_size | optimizer | regularization | accuracy (vl/ts) | best ratio (acc. to computation) | best accuracy (vl/ts) |
|:--------:|:---------------------------------:|:----------:|:--------:|:--------------:|:--------:|:--------------------:|:-------------:|
| CNN | 461,786 | 10 | Adamax (lr=2e-3) | L2 (decay=0) | 0.867 / 0.889 | 0.224 (1 epoch) | 1.000 / 0.999 (48 epochs) |
| GRU | 165,378 | 10 | Adamax (lr=2e-3) | L2 (decay=0) | 0.867 / 0.889 | 0.224 (1 epoch) | 1.000 / 0.999 (48 epochs) |

![Loss/Accuracy Plot of CNN model](17-ma721_IMDB_CNN_GRU/log/plot-cnn-1330.png "Loss/Accuracy Plots of CNN model")

## Conclusion


