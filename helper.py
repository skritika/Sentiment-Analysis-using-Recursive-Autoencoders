from def import*
import numpy as np
from scipy import io as sio
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
	#def updateparams():




class tree:
	def __init__(self, sl, hiddenSize, words):
		self.sl = sl
		self.hiddenSize = hiddenSize
		self.words = words 
		self.collapsed = range(1:sl+1)
		self.pp = np.zeros((2*sl-1,1))		
		self.nodeScores = np.zeros((2*sl-1,1))		
		self.kids = np.zeros((2*sl-1,2))		
		self.numkids = np.ones((2*sl-1,1))		
		self.y1c1 = np.zeros((hiddenSize,2*sl-1))		
		self.y2c2 = np.zeros((hiddenSize,2*sl-1))		
		self.freq = np.zeros((2*sl-1,1))		
		self.nodeFeatures = np.concatenate(words, np.zeros(hiddenSize,sl-1))
		self.nodeFeatures_unnorm = np.concatenate(words, np.zeros(hiddenSize,sl-1))
		self.delta1 = np.zeros((hiddenSize,2*sl-1))		
		self.delta2 = np.zeros((hiddenSize,2*sl-1))

	u1 = update1
	u2 = update2
	
	
def loaddata(path):

	vocab = sio.loadmat(path+'vocab.mat', squeeze_me=True, struct_as_record=False)['words'] #vocabulary used 
	print type(vocab)
	data_pos = sio.loadmat(path+'rt-polarity_pos_binarized.mat', squeeze_me=True, struct_as_record=False) 
	data_neg = sio.loadmat(path+'rt-polarity_neg_binarized.mat', squeeze_me=True, struct_as_record=False)
	
	#allSStr is array of all strring. Ex: allSStr[2] = [u'effective' u'but' u'*UNKNOWN*' u'*UNKNOWN*']
	allSStr = np.append(data_pos['allSStr'], data_neg['allSStr'])
	
	#allSNum is array of all string in which each word is mapped to index in vocab.
	#Ex: allSNum[2] = [1512   39    1    1]. This indexing starts with 1 -> so subtract 1
	allSNum = np.append(data_pos['allSNum']-1, data_neg['allSNum']-1)

	##manual editing###
	allSStr[5918] = [u'obvious'] #before editing - [array([], dtype=float64) array([], dtype=float64) u'obvious']
	allSNum[5918] = 4721
	allSStr[6549] = [u'horrible']
	allSNum[6549] = 20143;
	allSStr[9800] = [u'crummy'];
	allSNum[9800] = 241211;



	
