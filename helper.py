#from def2 import *
from backprop import *
import scipy.io
import math

class parameters:
	def __init__(self, hiddenSize, visibleSize, cat_size, dictionary_length, alpha, beta):
		r = math.sqrt(6)/math.sqrt(hiddenSize+visibleSize+1)
		self.W1 = np.random.rand(hiddenSize, visibleSize)*2*r-r
		self.W2 = np.random.rand(hiddenSize, visibleSize)*2*r-r
		self.W3 = np.random.rand(visibleSize, hiddenSize)*2*r-r
		self.W4 = np.random.rand(visibleSize, hiddenSize)*2*r-r
		self.We = 1e-3*(np.random.rand(hiddenSize, dictionary_length)*2*r-r)
		self.Wcat = np.random.rand(cat_size, hiddenSize)*2*r-r
		self.b1 = np.zeros((hiddenSize, 1))
		self.b2 = np.zeros((visibleSize, 1))
		self.b3 = np.zeros((visibleSize, 1))
		self.bcat = np.zeros((cat_size, 1))
		self.hiddenSize = hiddenSize
		self.cat_size = cat_size
		self.dictionary_length = dictionary_length
		self.alpha = alpha
		self.beta = beta
	computeder = backprop
	predict = predict


def accuracy(test_sentences, labels, freq, p):
	n = len(test_sentences)
	correct = 0
	for i in range(len(test_sentences)):
		correct += 1*(p.predict(freq,test_sentences[i])==labels[i])
	return correct/float(n)
	
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


def load():
	data = scipy.io.loadmat('data.mat')
	data = data['snum']
	train = []
	for i in range(data.shape[0]):
		x = data[i]
		y = x[0]
		z = y.tolist()
		u = z[0]
		train.append(u)

	lbl = scipy.io.loadmat('labels.mat')
	lbl = lbl['lbl']
	lbl = lbl[0]
	labels = []
	for i in range(lbl.shape[0]):
		labels.append(lbl[i])	
	return (train, labels)







	
