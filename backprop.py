import numpy as np

def f(x):
	return 1/(1+np.exp(x))

def fcat(x):
	return 1/(1+np.exp(x))

def f_prime(x):
	return f(x)*(1-f(x))

def fcat_prime(x):
	return f(x)*(1-f(x))

class tree:
	def __init__(self, sl, hiddenSize, cat_size, words):
		self.sl = sl
		self.hiddenSize = hiddenSize
		self.words = words 
		self.collapsed = range(1,sl+1)
		self.pp = np.zeros((2*sl-1,1),dtype=int)		
		self.nodeScoresR = np.zeros((2*sl-1,1))		
		self.nodeScores = np.zeros((2*sl-1,1))		
		self.kids = np.zeros((2*sl-1,2))		
		self.numkids = np.ones((2*sl-1,1))		
		self.y1c1 = np.zeros((hiddenSize,2*sl-1))		
		self.y2c2 = np.zeros((hiddenSize,2*sl-1))		
		self.freq = np.zeros((2*sl-1,1))		
		self.nodeFeatures = np.concatenate([words, np.zeros((hiddenSize,sl-1))], axis=1)
		self.nodeFeatures_unnorm = np.concatenate([words, np.zeros((hiddenSize,sl-1))], axis=1)
		self.delta1 = np.zeros((hiddenSize,2*sl-1))		
		self.delta2 = np.zeros((hiddenSize,2*sl-1))
		self.parentdelta = np.zeros((hiddenSize,2*sl-1))
		self.catdelta = np.zeros((cat_size,2*sl-1))
		self.catdelta_out = np.zeros((self.hiddenSize,2*sl-1))
'''Tree building and forward propagation:
	Calculates features, scores for all the nodes in the tree. 
	For all the non-leaf nodes, calculates (yi-ci), delta_i and reconstruction error
'''
		def forward(self, freq, W1, W2, W3, W4, Wcat, b1, b2, b3, bcat, alpha, beta, sentence_label):
			sl = self.sl
	
		'''Builds tree and computes recontruction error for each node'''
			for j in range(0,sl-1):
				words, lens = self.words, self.words.shape[1]
				c1, f1, c2, f2 = words[:,0:lens-1], freq[0:lens-1], words[:,1:lens], freq[1:lens]
				hidden_out = f(np.dot(W1,c1)+np.dot(W2,c2)+np.tile(b1,lens-1))		
				hidden_out_norm = hidden_out/hidden_out.sum(axis=1)[:,np.newaxis]
				y1, y2 = f(np.dot(W3,hidden_out_norm)+np.tile(b2,lens-1)), f(np.dot(W4,hidden_out_norm)+np.tile(b3,lens-1))	
				y1_norm, y2_norm = y1/y1.sum(axis=1)[:,np.newaxis], y2/y2.sum(axis=1)[:,np.newaxis]	
				y1c1, y2c2 = alpha*(y1-c1), alpha*(y2-c2)
				recons_error = sum(y1c1*(y1-c1)+y2c2*(y2-c2))*0.5
				re_min, re_min_pos = np.min(recons_error), np.argmin(recons_error)
				self.y1c1[:,sl+j], self.y2c2[:,sl+j] = y1c1[:,re_min_pos], y2c2[:,re_min_pos]	
				self.delta1[:,sl+j], self.delta2[:,sl+j] = np.dot(f_prime(y1[:,re_min_pos]), y1c1[:,re_min_pos]), np.dot(f_prime(y2[:,re_min_pos]), y2c2[:,re_min_pos])			
				index_child1, index_child2 = self.collapsed[re_min_pos], self.collapsed[re_min_pos+1]
				np.delete(words, re_min_pos+1, 1)
				words[:,re_min_pos] = hidden_out_norm[:,re_min_pos]
				self.nodeFeatures[:,sl+j], self.nodeFeatures_unnorm[:,sl+j] = hidden_out_norm[:,re_min_pos], hidden_out[:,re_min_pos]
				self.nodeScoresR[sl+j] = re_min
				self.pp[index_child1], self.pp[index_child2] = sl+j, sl+j
				self.kids[sl+j,0], self.kids[sl+j,1] = index_child1, index_child2
				self.numkids[sl+j] = self.numkids[self.kids[sl+j,0]] + self.numkids[self.kids[sl+j,1]]
				np.delete(self.freq,re_min_pos+1,0) 
				self.freq[re_min_pos] = (np.dot(self.numkids[self.kids[sl+j,0]], f1[re_min_pos]) +  np.dot(self.numkids[self.kids[sl+j,1]], f2[re_min_pos]))/(self.numkids[sl+j,0]+self.numkids[sl+j,1])
				np.delete(self.collapsed,re_min_pos)
				self.collapsed[re_min_pos]=sl+j

		'''Classification error computation for each node'''
			out = fcat(np.dot(Wcat,self.words)+np.tile(bcat,lens))
			diff = np.tile(sentence_label,sl)-out
			lbl_sm = (1-alpha)*diff
			score = 0.5*lbl_sm*diff
			self.nodeScores[0:sl], self.catdelta[:,0:sl] = score.T, -(lbl_sm)*fcat_prime(out)
			for i in range(lens,2*lens-1):
				sm = f(np.dot(Wcat,self.nodeFeatures[:,i]) + bcat)
				lbl_sm = beta*(1-alpha)*(sentence_label-sm)
				self.catdelta[:,i] = -(lbl_sm)*f_prime(sm)
				J = 0.5*(np.dot(lbl_sm.T,(sentence_label-sm)))
				self.nodeScores[i] = J

'''Backpropagation for derivative and cost computation'''
def backprop(self, training_data, training_labels, freq_original):
	W1, W2, W3, W4, Wcat, b1, b2, b3, bcat, We = self.W1, self.W2, self.W3, self.W4, self.Wcat, self.b1, self.b2, self.b3, self.bcat, self.We
	sW = (self.hiddenSize, self.hiddenSize)
	sb = (self.hiddenSize, 1)
	gW1, gW2, gW3, gW4, gWcat, gb1, gb2, gb3, gbcat, gWe = np.zeros(sW), np.zeros(sW), np.zeros(sW), np.zeros(sW), np.zeros((self.cat_size, self.hiddenSize)), np.zeros(sb), np.zeros(sb), np.zeros(sb), np.zeros((self.cat_size,1)), np.zeros((self.hiddenSize, self.dictionary_length))
	gWe_tot = np.zeros((self.hiddenSize, self.dictionary_length))	
	for i in range(len(training_data)):
		data = training_data[i]
		true_label = training_labels[i]
		sl = len(data)
		words_emb = We[:,word_indices]
		gL = np.zeros((L.shape[0],L.shape[1]))		
		freq = [freq_original[k] for k in word_indices]
		tr = tree(sl, self.hiddenSize, self.cat_size, words_emb)
		if sl>1 : 
			tr.forward(freq, W1, W2, W3, W4, Wcat, b1, b2, b3, bcat, self.alpha, self.beta, true_label)
			for current in range(2*sl-2,sl+1,-1):
				kid1, kid2 = tr.kids[current,0], tr.kids[current,1]
				a1, a1_unnorm = tr.nodeFeatures[:,current], tr.nodeFeatures_unnorm[:,current]
				d1, d2 = tr.delta1[:,current], tr.delta1[:,current]
				pd = tr.parentdelta[:,current]	
				if(current==(2*sl-2)):
					W = np.zeros((self.hiddenSize,self.hiddenSize))
				else:
					W=W2
					if(tr.kids[pp,0]==current):#left_child
						W = W1
				smd = tr.catdelta[:, current]		
				gbcat += smd
				parent_d = np.dot(f_prime(a1_unnorm), np.dot(W3.T, d1)+np.dot(W4.T, d2)+np.dot(W.T, pd)+np.dot(Wcat.T, smd))
				gWcat += np.dot(smd,a1.T)
				tr.parentdelta[:,kid1], tr.parentdelta[:,kid2] = parent_d, parent_d
				gb1, gb2, gb3 = gb1+parent_d, gb2+d1, gb3+d2
				gW1, gW2, gW3, gW4 = gW1 + dot(parent_d, tr.nodeFeatures[:,kid1].T), gW2 + dot(parent_d, tr.nodeFeatures[:,kid2].T), gW3 + dot(d1,a1.T), gW4 + dot(d2,a1.T)
			for j in range(sl-1,-1,-1):
				pp = tr.pp[j]
				W=W2
				if(tr.kids[pp,0]==j):#left_child
					W = W1
				gWcat += np.dot(tr.catdelta[:,j][:,np.newaxis],tr.nodeFeatures[:,j][:,np.newaxis].T) 
				gbcat += tr.catdelta[:,j]  
				gL[:,j] += np.dot(W.T,tr.parentdelta[:,j]) + np.dot(Wcat.T,tr.catdelta[:,j]) 	
				gWe[:,word_indices[j]] += gL[:,j]
		gWe_tot += gWe
			


