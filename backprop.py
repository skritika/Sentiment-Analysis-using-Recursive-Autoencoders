from definitions import *

'''Backpropagation for derivative and cost computation'''
def backprop(self, training_data, training_labels, freq_original):
	W1, W2, W3, W4, Wcat, b1, b2, b3, bcat, We = self.W1, self.W2, self.W3, self.W4, self.Wcat, self.b1, self.b2, self.b3, self.bcat, self.We
	sW = (self.hiddenSize, self.hiddenSize)
	sb = (self.hiddenSize, 1)
	gW1, gW2, gW3, gW4, gWcat, gb1, gb2, gb3, gbcat, gWe = np.zeros(sW), np.zeros(sW), np.zeros(sW), np.zeros(sW), np.zeros((self.cat_size, self.hiddenSize)), np.zeros(sb), np.zeros(sb), np.zeros(sb), np.zeros((self.cat_size,1)), np.zeros((self.hiddenSize, self.dictionary_length))
	gWe_tot = np.zeros((self.hiddenSize, self.dictionary_length))	
	for i in range(len(training_data)):
		word_indices = training_data[i]
		true_label = training_labels[i]
		sl = len(word_indices)
		L = We[:,word_indices]
		gL = np.zeros((L.shape[0],L.shape[1]))		
		freq = [freq_original[k] for k in word_indices]
		tr = tree(sl, self.hiddenSize, self.cat_size, L)
		if sl>1 : 
			tr.forward(freq, W1, W2, W3, W4, Wcat, b1, b2, b3, bcat, self.alpha, self.beta, true_label)
			print tr.checkgradient(freq, 0.000000001, W1, W2, W3, W4, Wcat, b1, b2, b3, bcat, self.alpha, self.beta, true_label)
			for current in range(2*sl-2,sl-1,-1):
				kid1, kid2 = tr.kids[current,0], tr.kids[current,1]
				a1, a1_unnorm = tr.nodeFeatures[:,current][:,np.newaxis], tr.nodeFeatures_unnorm[:,current][:,np.newaxis]
				d1, d2 = tr.delta1[:,current][:,np.newaxis], tr.delta1[:,current][:,np.newaxis]
				pd = tr.parentdelta[:,current][:,np.newaxis]
				pp = tr.pp[current]
				if(current==(2*sl-2)):
					W = np.zeros((self.hiddenSize,self.hiddenSize))
				else:
					W=W2
					if(tr.kids[pp,0]==current):#left_child
						W = W1
				smd = tr.catdelta[:, current][:,np.newaxis]
				gbcat += smd
				h = np.dot(W3.T, d1) + np.dot(W4.T, d2) + np.dot(W.T, pd) + np.dot(Wcat.T, smd)
				parent_d = np.dot(f_prime(a1_unnorm), np.dot(W3.T, d1)+np.dot(W4.T, d2)+np.dot(W.T, pd)+np.dot(Wcat.T, smd))
				gWcat += np.dot(smd,a1.T)
				tr.parentdelta[:,kid1:kid1+1], tr.parentdelta[:,kid2:kid2+1] = parent_d, parent_d
				gb1, gb2, gb3 = gb1+parent_d, gb2+d1, gb3+d2
				gW1, gW2, gW3, gW4 = gW1 + np.dot(parent_d, tr.nodeFeatures[:,kid1][:,np.newaxis].T), gW2 + np.dot(parent_d, tr.nodeFeatures[:,kid2][:,np.newaxis].T), gW3 + np.dot(d1,a1.T), gW4 + np.dot(d2,a1.T)
			for j in range(sl-1,-1,-1):
				pp = tr.pp[j]
				W=W2
				if(tr.kids[pp,0]==j):#left_child
					W = W1
				gWcat += np.dot(tr.catdelta[:,j][:,np.newaxis],tr.nodeFeatures[:,j][:,np.newaxis].T) 
				gbcat += tr.catdelta[:,j]  
				gL[:,j] += np.dot(W.T,tr.parentdelta[:,j]) + np.dot(Wcat.T,tr.catdelta[:,j]) 	
				gWe[:,word_indices[j]] += gL[:,j]
				print gW1[1,1]
		gWe_tot += gWe
		
		

