from definitions import *

def predict(self, freq, test_sentence):
	sl = len(test_sentence)
	L = self.We[:,test_sentence]
	tr = tree(sl, self.hiddenSize, self.cat_size, L)
	tr.forward(freq, self.W1, self.W2, self.W3, self.W4, self.Wcat, self.b1, self.b2, self.b3, self.bcat, self.alpha, self.beta, 0)
	pred = fcat(np.dot(self.Wcat,tr.nodeFeatures[:,2*sl-2]))
	return 1*(pred>0.5)
			
'''Backpropagation for derivative and cost computation'''
def backprop(self, training_data, training_labels, freq_original):
	W1, W2, W3, W4, Wcat, b1, b2, b3, bcat, We = self.W1, self.W2, self.W3, self.W4, self.Wcat, self.b1, self.b2, self.b3, self.bcat, self.We
	sW = (self.hiddenSize, self.hiddenSize)
	sb = (self.hiddenSize, 1)
	gW1, gW2, gW3, gW4, gWcat, gb1, gb2, gb3, gbcat, gWe = np.zeros(sW), np.zeros(sW), np.zeros(sW), np.zeros(sW), np.zeros((self.cat_size, self.hiddenSize)), np.zeros(sb), np.zeros(sb), np.zeros(sb), np.zeros((self.cat_size,1)), np.zeros((self.hiddenSize, self.dictionary_length))
	gWe_tot = np.zeros((self.hiddenSize, self.dictionary_length))	
	cost_J = 0.0
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
			for current in range(2*sl-2,sl-1,-1):
				kid1, kid2 = tr.kids[current,0], tr.kids[current,1]
				a1, a1_unnorm = tr.nodeFeatures[:,current:current+1], tr.nodeFeatures_unnorm[:,current:current+1]
				d1, d2 = tr.delta1[:,current:current+1], tr.delta2[:,current:current+1]
				pd = tr.parentdelta[:,current:current+1]
				pp = tr.pp[current]
				if(current==(2*sl-2)):
					W = np.zeros((self.hiddenSize,self.hiddenSize))
					delt = np.zeros((self.hiddenSize, 1))
				else:
					W, delt = W2.copy(), tr.y2c2[:,pp:pp+1] 
					if(tr.kids[pp,0]==current):#left_child
						W, delt = W1.copy(), tr.y1c1[:,pp:pp+1]
				smd = tr.catdelta[:, current:current+1]
				gbcat += smd
				h = np.dot(W3.T, d1) + np.dot(W4.T, d2) + np.dot(W.T, pd) + np.dot(Wcat.T, smd) - delt
				parent_d = np.dot(fnorm_prime(a1_unnorm), h)
				gWcat += np.dot(smd,a1.T)
				tr.parentdelta[:,kid1:kid1+1], tr.parentdelta[:,kid2:kid2+1] = parent_d, parent_d
				gb1, gb2, gb3 = gb1+parent_d, gb2+d1, gb3+d2
				gW1, gW2, gW3, gW4 = gW1 + np.dot(parent_d, tr.nodeFeatures[:,kid1:kid1+1].T), gW2 + np.dot(parent_d, tr.nodeFeatures[:,kid2:kid2+1].T), gW3 + np.dot(d1,a1.T), gW4 + np.dot(d2,a1.T)
			for j in range(sl-1,-1,-1):
				pp = tr.pp[j]
				W, delt = W2.copy(), tr.y2c2[:,pp:pp+1] 
				if(tr.kids[pp,0]==j):#left_child
					W, delt = W1.copy(), tr.y1c1[:,pp:pp+1]
				gWcat += np.dot(tr.catdelta[:,j:j+1],tr.nodeFeatures[:,j:j+1].T) 
				gbcat += tr.catdelta[:,j]  
				gL[:,j:j+1] += np.dot(W.T,tr.parentdelta[:,j:j+1]) + np.dot(Wcat.T,tr.catdelta[:,j:j+1]) - delt 	
				gWe_tot[:,word_indices[j]] += gL[:,j]
			cost_J += sum(tr.nodeScores) + sum(tr.nodeScoresR)
			print gWe[0,2]
			print tr.checkgradient(word_indices, freq, 0.0000000000001, W1, W2, W3, W4, Wcat, We, b1, b2, b3, bcat, self.alpha, self.beta, true_label)
		gWe_tot += gWe
	F = np.ndarray.flatten
	D = np.dot
	#final grad computation
	grad_J = np.concatenate([F(gW1),F(gW2),F(gW3),F(gW4),F(gWcat),F(gb1),F(gb2),F(gb3),F(gbcat),F(gWe_tot)],axis=1)
	grad_reg = np.concatenate([F(W1),F(W2),F(W3),F(W4),F(Wcat),F(b1),F(b2),F(b3),F(We)],axis=1)
	#grad = grad_J/len(training_data) + .0004*grad_reg
	#print grad.shape
	#final cost computation		
	cost_reg = .0004*(D(F(W1),F(W1).T)+D(F(W2),F(W2).T)+D(F(W3),F(W3).T)+D(F(W4),F(W4).T)+D(F(Wcat),F(Wcat).T)+D(F(We),F(We).T))
	cost = cost_J/len(training_data) + cost_reg
	#return (grad, cost)
		
		

