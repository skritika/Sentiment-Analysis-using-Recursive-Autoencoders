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
	
	def update1(self, freq, W1, W2, W3, W4, b1, b2, b3, alpha):
		for j in range(1,self.sl)
			words = self.words
			lens = words.shape[1]
			c1 = words[:,0:lens-1]
			c2 = words[:,1:lens]
			f1 = freq(0:lens-1)		
			f2 = freq(1:lens)
			hidden_out = np.tanh(np.dot(W1,c1)+np.dot(W2,c2)+np.tile(b1,lens-1))		
			hidden_out_norm = hidden_out/hidden_out.sum(axis=1)[:,np.newaxis]
			y1 = np.tanh(np.dot(W3,hidden_out_norm)+np.tile(b2,lens-1))			
			y2 = np.tanh(np.dot(W4,hidden_out_norm)+np.tile(b3,lens-1))			
			y1_norm = y1_norm/y1_norm.sum(axis=1)[:,np.newaxis]	
			y2_norm = y2_norm/y2_norm.sum(axis=1)[:,np.newaxis]	
			y1c1 = alpha*(y1-c1)
			y2c2 = alpha*(y2-c2)
			recons_error = sum(y1c1*(y1-c1)+y2c2*(y2-c2))*0.5
			re_min = min(recons_error)
			re_min_pos = amin(recons_error)
			self.node_y1c1[:,sl+j] = y1c1[:,re_min_pos]	
			self.node_y2c2[:,sl+j] = y2c2[:,re_min_pos]	
			self.delta1[:,sl+j]	= dot(np.square(np.sech(y1[:,re_min_pos])), y1c1[:,re_min_pos])		
			self.delta2[:,sl+j]	= dot(np.square(np.sech(y2[:,re_min_pos])), y2c2[:,re_min_pos])		
			
			delete(self.words, re_min_pos+1, 1)
			self.words[:,re_min_pos] = hidden_out_norm[:,re_min_pos]
			self.nodeFeatures[:,sl+j] = hidden_out_norm[:,re_min_pos]
			self.nodeFeatures_unnorm[:,sl+j] = hidden_out[:,re_min_pos]
			self.nodeScores[sl+j] = re_min
			self.pp[collapsed_sentence[re_min_pos]] = sl+j;
			self.pp[collapsed_sentence[re_min_pos+1]] = sl+j;
			self.kids[sl+j,:] = concat(collapsed_sentence(re_min_pos), collapsed_sentence(re_min_pos))
			self.numkids[sl+j,:] = self.numkids[self.kids[sl+j,1]] + self.numkids[self.kids[sl+j,2]]
			delete(self.freq,re_min_pos+1,0) 
			self.freq[re_min_pos] = dot(self.numkids[self.kids[sl+j,1]], f1[re_min_pos]) +  dot(self.numkids[self.kids[sl+j,2]], f2[re_min_pos])/self.numkids[sl+j,:]
			delete(collapsed_sentence,re_min_pos)
			collapsed_sentence[re_min_pos]=sl+j


	def update2():	


	
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



	
