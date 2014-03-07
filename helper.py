import numpy as np
import math


class parameters:
	def __init__(self, hiddenSize, visibleSize, cat_size, dictionary_length):
		self.r = math.sqrt(6)/math.sqrt(hiddenSize+visibleSize+1)
		self.W1 = np.random.randn(hiddenSize, visibleSize)*2*r-r
		self.W2 = np.random.randn(hiddenSize, visibleSize)*2*r-r
		self.W3 = np.random.randn(visibleSize, hiddenSize)*2*r-r
		self.W4 = np.random.randn(visibleSize, hiddenSize)*2*r-r
		self.We = 1e-3*(np.random.randn(hiddenSize, dictionary_length)*2*r-r)
		self.Wcat = np.random.randn(cat_size, hiddenSize)*2*r-r
		self.b1 = np.zeros(hiddenSize, 1)
		self.b2 = np.zeros(visibleSize, 1)
		self.b3 = np.zeros(visibleSize, 1)
		self.bcat = np.zeros(cat_size, 1)
	def updateparams():
		
	
		

def loaddata():




	
