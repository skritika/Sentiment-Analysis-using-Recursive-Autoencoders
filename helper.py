#from def2 import *
from backprop import *
import scipy.io
import math

def init_theta(d, num_cat, dict_length):
		r = math.sqrt(6)/math.sqrt(d+d+1)
		#We = 1e-3*(np.random.rand(hiddenSize, dictionary_length)*2*r-r)
		W = np.random.rand(1,4*d*d+d*num_cat+d*dict_length)*2*r-r #W1+W2+W3+W4+Wcat+We
		b = np.zeros((1,3*d+num_cat))
		return np.concatenate([W,b],axis=1)

def accuracy(W1, W2, W3, W4, Wcat, We, b1, b2, b3, bcat, alpha, beta, freq, test_sentences, labels, d, num_cat):
	n = len(test_sentences)
	correct = 0
	for i in range(len(test_sentences)):
		p = predict(W1, W2, W3, W4, Wcat, We, b1, b2, b3, bcat, alpha, beta, freq, test_sentences[i], d, num_cat)
		correct += 1*(p[0]==labels[i])
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
		y = x[0]-1
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







	
