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
	correct = correct*100
	return correct/float(n)

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







	
