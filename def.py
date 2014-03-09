def update1(self, freq, W1, W2, W3, W4, b1, b2, b3, alpha):
	'''Reconstruction error computation'''
	for j in range(1,self.sl):
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
		
		delete(words, re_min_pos+1, 1)
		words[:,re_min_pos] = hidden_out_norm[:,re_min_pos]
		self.nodeFeatures[:,sl+j] = hidden_out_norm[:,re_min_pos]
		self.nodeFeatures_unnorm[:,sl+j] = hidden_out[:,re_min_pos]
		self.nodeScores[sl+j] = re_min
		self.pp[self.collapsed[re_min_pos]] = sl+j;
		self.pp[self.collapsed[re_min_pos+1]] = sl+j;
		self.kids[sl+j,:] = concat(self.collapsed[re_min_pos], self.collapsed[re_min_pos])
		self.numkids[sl+j,:] = self.numkids[self.kids[sl+j,1]] + self.numkids[self.kids[sl+j,2]]
		delete(self.freq,re_min_pos+1,0) 
		self.freq[re_min_pos] = (dot(self.numkids[self.kids[sl+j,1]], f1[re_min_pos]) +  dot(self.numkids[self.kids[sl+j,2]], f2[re_min_pos]))/self.numkids[sl+j,:]
		delete(self.collapsed,re_min_pos)
		self.collapsed[re_min_pos]=sl+j

def update2(self, allKids, Wcat, W1, W2, b1, bcat, alpha_cat, sentence_label, beta):	
	'''Classification error computation'''
	lens = self.sl
	nodeUnder = np.ones((2*lens-1,1))
	for i in range(lens+1,2*lens):
		kids = allKids[i,:]
		n1 = nodeUnder[kids[1]]		
		n2 = nodeUnder[kids[2]]		
		nodeUnder[i] = n1+n2
	num_cat = Wcat.shape[0]
	self.catdelta = np.zeros(num_cat, 2*lens-1)
	self.catdelta_out = np.zeros(self.hiddenSize, 2*lens-1)
	
	#I have no idea why but they are finding labels for single words here
	out = np.sig(dot(Wcat,self.words)+np.tile(bcat,lens))#check function, tanh is not exactly sigmoid
	#out is of size num_catxlens
	#The following line needs to be changed for multiple categories, it expects sentence_label to be of size num_cat
	diff = np.tile(sentence_label,lens,1)-out
	lbl_sm = (1-alpha_cat)*diff
	self.nodeScores[:,0:sl] = 0.5*lbl_sm*diff
	self.catdelta[:,0:sl] = -(lbl_sm)*sigmoid_prime(sm)
	for i in range(lens+1:2*lens):
		kids = allKids[i,:]
		c1  = self.nodeFeatures[:,kids[1]]
		c2  = self.nodeFeatures[:,kids[2]]
		p = np.tanh(dot(W1,c1)+dot(W2,c2)+b1)
		p_norm = p/np.linalg.norm(p)
		sm = np.sigmoid(Wcat*p_norm + bcat)
		lbl_sm = beta*(1-alpha_cat)*(sentence_label-sm)
		self.catdelta[:,i] = -(lbl_sm)*sigmoid_prime(sm)
		J = 0.5*(dot(numpy.transpose(lbl_sm),(sentence_label-sm)))
		self.nodeFeatures[:,i] = p_norm
		self.nodeFeatures_unnorm[:,i] = p
		self.nodeScores[i] = J
		self.numkids = nodeUnder







