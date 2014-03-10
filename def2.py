import numpy as np

def f(x):
	return 1/(1+np.exp(x))

def f_prime(x):
	return f(x)*(1-f(x))

def update1(self, freq, W1, W2, W3, W4, b1, b2, b3, alpha):
	'''Reconstruction error computation'''
	for j in range(1,self.sl):
		words, lens = self.words, words.shape[1]
		c1, c2 = words[:,0:lens-1], words[:,1:lens]
		f1, f2 = freq[0:lens-1], freq[1:lens]
		hidden_out = f(np.dot(W1,c1)+np.dot(W2,c2)+np.tile(b1,lens-1))		
		hidden_out_norm = hidden_out/hidden_out.sum(axis=1)[:,np.newaxis]
		y1, y2 = f(np.dot(W3,hidden_out_norm)+np.tile(b2,lens-1)), f(np.dot(W4,hidden_out_norm)+np.tile(b3,lens-1))	
		y1_norm, y2_norm = y1_norm/y1_norm.sum(axis=1)[:,np.newaxis], y2_norm/y2_norm.sum(axis=1)[:,np.newaxis]	
		y1c1, y2c2 = alpha*(y1-c1), alpha*(y2-c2)
		recons_error = sum(y1c1*(y1-c1)+y2c2*(y2-c2))*0.5
		re_min, re_min_pos = min(recons_error), amin(recons_error)
		self.node_y1c1[:,sl+j], self.node_y2c2[:,sl+j] = y1c1[:,re_min_pos], y2c2[:,re_min_pos]	
		self.delta1[:,sl+j]	= dot(f_prime(y1[:,re_min_pos]), y1c1[:,re_min_pos])		
		self.delta2[:,sl+j]	= dot(f_prime(y2[:,re_min_pos]), y2c2[:,re_min_pos])			
		index_child1, index_child2 = self.collapsed[re_min_pos], self.collapsed[re_min_pos+1]
		delete(words, re_min_pos+1, 1)
		words[:,re_min_pos] = hidden_out_norm[:,re_min_pos]
		self.nodeFeatures[:,sl+j], self.nodeFeatures_unnorm[:,sl+j] = hidden_out_norm[:,re_min_pos], hidden_out[:,re_min_pos]
		self.nodeScores[sl+j] = re_min
		self.pp[index_child1], self.pp[index_child2] = sl+j, sl+j
		self.kids[sl+j,:] = concat(index_child1, index_child2)
		self.numkids[sl+j,:] = self.numkids[self.kids[sl+j,1]] + self.numkids[self.kids[sl+j,2]]
		delete(self.freq,re_min_pos+1,0) 
		self.freq[re_min_pos] = (dot(self.numkids[self.kids[sl+j,1]], f1[re_min_pos]) +  dot(self.numkids[self.kids[sl+j,2]], f2[re_min_pos]))/self.numkids[sl+j,:]
		delete(self.collapsed,re_min_pos)
		self.collapsed[re_min_pos]=sl+j

def update2(self, allKids, W1, W2, Wcat, b1, bcat, alpha_cat, sentence_label, beta):	
	'''Classification error computation'''
	lens = self.sl
	nodeUnder = np.ones((2*lens-1,1))
	for i in range(lens+1,2*lens):
		kids = allKids[i,:]
		n1, n2 = nodeUnder[kids[1]], nodeUnder[kids[2]]		
		nodeUnder[i] = n1+n2
	num_cat = Wcat.shape[0]
	self.catdelta, self.catdelta_out = np.zeros(num_cat, 2*lens-1), np.zeros(self.hiddenSize, 2*lens-1)
	
	#I have no idea why but they are finding labels for single words here
	out = np.sig(dot(Wcat,self.words)+np.tile(bcat,lens))#check function, tanh is not exactly sigmoid
	#out is of size num_catxlens
	#The following line needs to be changed for multiple categories, it expects sentence_label to be of size num_cat
	diff = np.tile(sentence_label,lens,1)-out
	lbl_sm = (1-alpha_cat)*diff
	self.nodeScores[:,0:sl] = 0.5*lbl_sm*diff
	self.catdelta[:,0:sl] = -(lbl_sm)*sigmoid_prime(sm)
	for i in range(lens+1,2*lens):
		kids = allKids[i,:]
		c1, c2  = self.nodeFeatures[:,kids[1]], self.nodeFeatures[:,kids[2]]
		p, p_norm = np.tanh(dot(W1,c1)+dot(W2,c2)+b1), p/np.linalg.norm(p)
		sm = np.sigmoid(Wcat*p_norm + bcat)
		lbl_sm = beta*(1-alpha_cat)*(sentence_label-sm)
		self.catdelta[:,i] = -(lbl_sm)*sigmoid_prime(sm)
		J = 0.5*(dot(numpy.transpose(lbl_sm),(sentence_label-sm)))
		self.nodeFeatures[:,i], self.nodeFeatures_unnorm[:,i] = p_norm, p
		self.nodeScores[i] = J
		self.numkids = nodeUnder
	self.kids = allKids

class tree:
	def __init__(self, sl, hiddenSize, words):
		self.sl = sl
		self.hiddenSize = hiddenSize
		self.words = words 
		self.collapsed = range(1,sl+1)
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
		self.parentdelta = np.zeros((hiddenSize,2*sl-1))
		self.catdelta = np.zeros((hiddenSize,2*sl-1))
	update1 = update1
	update2 = update2


def backprop(self, flag_cat, training_data, training_labels, freq_original):#training_data is a list of lists
	'''Backpropagation'''
	W1, W2, W3, W4, Wcat, b1, b2, bcat, We = self.W1, self.W2, self.W3, self.W4, self.Wcat, self.b1, self.b2, self.bcat, self.We
	sW = (self.hiddenSize, self.hiddenSize)
	sb = (self.hiddenSize, 1)
	gW1, gW2, gW3, gW4, gWcat, gb1, gb2, gbcat, gWe = np.zeros(sW), np.zeros(sW), np.zeros(sW), np.zeros(sW), np.zeros((self.cat_size, self.hiddenSize)), np.zeros(sb), np.zeros(sb), np.zeros((self.cat_size,1)), np.zeros((self.hiddenSize, self.dictionary_length))
	
	for i in range(len(training_data)):
		data = training_data[i]
		true_label = training_labels[i]#not sure if this is the true_label
		sl = len(data)
	
		word_indices = data
		L = We[:,word_indices]#feature vectors for the words in sentences
		words_emb = L #incomplete
		gL = np.zeros((L.shape[0],L.shape[1]))		
		freq = [freq_original[k] for k in word_indices]
		tr = tree(sl, self.hiddenSize, words_emb)
		if sl>1 : 
			#Forward Propagation
			if flag_cat:
				tr.update2(allKids[i], W1, W2, Wcat, b1, bcat, alpha_cat, true_label, beta)
			else:
				tr.update1(freq, W1, W2, W3, W4, b1, b2, b3, alpha)
			#Backpropagation
			nodeFeatures, nodeFeatures_unnorm = tr.nodeFeatures, tr.nodeFeatures_unnorm
			for j in range(2*sl-2,sl+1,-1):
				current_parent = j
				kid1, kid2 = tr.kids[0], tr.kids[1]
				a1, a1_unnorm = tr.nodeFeatures[:,current_parent], tr.nodeFeatures_unnorm[:,current_parent]
				d1, d2 = tr.delta1[:,current_parent], tr.delta1[:,current_parent]
				
				W=W2
				if(tr.kids[pp,0]==current_parent):#left_child
					W = W1
 				if flag_cat:
					smd = tr.catdelta[:, current_parent]		
					gbcat += smd
					pp = tr.pp[current_parent]
					der = dot(W3.T, d1)+dot(W4.T, d2)+dot(W.T, pd)+dot(Wcat.T, smd)
					gWcat += dot(smd,a1.T)
				else:			
					der = dot(W3.T, d1)+dot(W4.T, d2)+dot(W.T, pd)
				parent_d = dot(f_prime(a1_unnorm), der)
					
				tr.parentdelta[:,kid1], tr.parentdelta[:,kid2] = parent_d, parent_d
				gb1, gb2, gb3 = gb1+parent_d, gb2+d1, gb3+d2
				gW1, gW2, gW3, gW4 = gW1 + dot(parent_d, tr.nodeFeatures[:,kid1]), gW2 + dot(parent_d, tr.nodeFeatures[:,kid2]), gW3 + dot(d1,a1.T), gW4 + dot(d2,a1.T)
			for j in range(sl-1,-1,-1):
				pp = tr.pp[j]
				W=W2
				if(tr.kids[pp,0]==j):#left_child
					W = W1
				if flag_cat:
					gWcat += dot(tr.nodeFeatures[catdelta[:,j]],tr.catdelta[:,j])  
					gbcat += tr.catdelta[:,j]  
					gL[:,j] += dot(W.T,tr.parent) + dot()	
				gWe[:,word_indices[j]] += gL[:,j]
			gWe_tot += gWe
				
	





