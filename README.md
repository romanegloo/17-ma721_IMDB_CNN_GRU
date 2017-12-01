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

#### vocabulary
The number of unique words identified by wikipedia word embeddings is 84,546. 

#### examples
*Example of a Negative Review*  
  
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
    
*Example of a Positive Review*  

    This is a bizzare look at Al's "life", back when he still a hyper 
    20-something. The (real) home videos of Al as a kid are great, and the 
    commentary from his (real life) parents gives a nice glimpse of just how 
    Weird Al wound up as screwed up as he is. This video is a must own for any 
    devoted Al-coholic.
    

## Model Architecture

### Convolutional Neural Network (CNN)

* **Parameters of CNN**

	```
	CnnImdbSA (
		(encoder): Embedding(7643, 300, padding_idx=0), weights=((7643, 300),), parameters=2292900
		(convs1): ModuleList (
    		(0): Conv2d(1, 100, kernel_size=(3, 300), stride=(1, 1))
			(1): Conv2d(1, 100, kernel_size=(4, 300), stride=(1, 1))
			(2): Conv2d(1, 100, kernel_size=(5, 300), stride=(1, 1))
		), weights=((100, 1, 3, 300), (100,), (100, 1, 4, 300), (100,), (100, 1, 5, 300), (100,)), parameters=360300
		(dropout): Dropout (p = 0.5), weights=(), parameters=0
		(decoder): Linear (300 -> 2), weights=((2, 300), (2,)), parameters=602
	)
	```
	Total number of parameters (excluding the embedding layer): 360,902
### Recurrent Neural Network (RNN)


* **Parameters of RNN (GRU)**  

  ```
  GruImdbSA (
      (encoder: Embedding(84546, 300), weights=((84546, 300),), parameters=25363800
      (gru): GRU(300, 128), weights=((384, 300), (384, 128), (384,), (384,)), parameters=165120
      (decoder): Linear (128 -> 2), weights=((2, 128), (2,)), parameters=258
  )
  ```

  Total number of parameters (excluding the embedding layer): 165,378

## Results

| network | capacity (epoch x parameters) | batch\_size | optimizer | regularization | accuracy (vl/ts) | ratio (acc. to computation) | best accuracy (vl/ts) |
|:--------:|:---------------------------------:|:----------:|:--------:|:--------------:|:--------:|:--------------------:|:-------------:|
| CNN | 3 x 165,378 | 64 | SGD (lr=2e-4) | L2 (decay=0.1) | 88.00 / 91 .60 | 26.39 | 98.92 / 99.20 (48 epochs) |

## Conclusion


