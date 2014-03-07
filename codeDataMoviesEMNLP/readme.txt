%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code for:
% Semi-Supervised Recursive Autoencoders for Predicting Sentiment Distributions
% Richard Socher, Jeffrey Pennington, Eric Huang, Andrew Y. Ng, and Christopher D. Manning
% Conference on Empirical Methods in Natural Language Processing (EMNLP 2011)
% See http://www.socher.org for more information or to ask questions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

This code is provided as is. It is free for academic, non-commercial purposes. 
For questions, please contact richard @ socher .org


This code can be used in two major ways:
1) To train a semi-supervised recursive autoencoder from random word vectors and without sentiment lexica on movie reviews.
2) To test using our best trained model on the first movie review fold.

The main file is trainTestRAE.m

To run it, just open matlab and enter trainTestRAE

In this file you can set the flag to switch between options 1) and 2) 
params.trainModel = 0;

The test code should give an accuracy of around acc_test = 0.7842.

If you have a multicore machine, the code will be able to use all cores and parallelize.


Please cite the following paper if you use the code:

@inproceedings{SocherEtAl2011:RAE,
author = {Richard Socher and Jeffrey Pennington and Eric H. Huang and Andrew Y. Ng and Christopher D. Manning},
title = {{Semi-Supervised Recursive Autoencoders for Predicting Sentiment Distributions}},
booktitle = {Proceedings of the 2011 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
year = 2011
}



This archive includes 2 external items for convenience:

- The movie review dataset from Bo Pang: sentence polarity dataset v1.0 
  available at http://www.cs.cornell.edu/people/pabo/movie-review-data/ 
  Introduced in Pang/Lee ACL 2005. Released July 2005. 

- minFunc, a general optimizer for Matlab from Mark Schmidt (2005) 
  available at http://www.di.ens.fr/~mschmidt/Software/minFunc.html

