import numpy as np

def f(x):
	f = np.tanh(x)
	return f

def fnorm(x):
	(n1, n2) = x.shape
	f = np.tanh(x)
	norm = np.linalg.norm(f,axis=0)*np.ones((n1,n2))
	fnorm = f/norm
	return fnorm	

def fcat(x):
	return 1/(1+np.exp(-x))

def f_prime(f):
	f_p = 1 - np.square(f)
	return f_p

def fnorm_prime(f_unnorm):
	f = f_unnorm
	f_p = 1 - np.square(f)
	diag = np.diagflat(f_p)
	norm = np.linalg.norm(f)
	fnorm_p = diag/norm - np.dot(diag, np.dot(f, f.T))/np.power(norm,3)
	return fnorm_p

def fcat_prime(x):
	#return fcat(x)*(1-fcat(x))
	return x*(1-x)

class tree:
	def __init__(self, sl, hiddenSize, cat_size, words):
		self.sl = sl
		self.hiddenSize = hiddenSize
		self.words = words 
		self.collapsed = range(0,sl)
		self.pp = np.zeros((2*sl-1,1),dtype=int)		
		self.nodeScoresR = np.zeros((2*sl-1,1))		
		self.nodeScores = np.zeros((2*sl-1,1))		
		self.kids = np.zeros((2*sl-1,2))		
		self.numkids = np.ones((2*sl-1,1))		
		self.y1c1 = np.zeros((hiddenSize,2*sl-1))		
		self.y2c2 = np.zeros((hiddenSize,2*sl-1))		
		self.freq = np.zeros((2*sl-1,1))		
		self.nodeFeatures = np.concatenate((words, np.zeros((hiddenSize,sl-1))), axis=1)
		self.nodeFeatures_unnorm = np.concatenate((words, np.zeros((hiddenSize,sl-1))), axis=1)
		self.delta1 = np.zeros((hiddenSize,2*sl-1))		
		self.delta2 = np.zeros((hiddenSize,2*sl-1))
		self.parentdelta = np.zeros((hiddenSize,2*sl-1))
		self.catdelta = np.zeros((cat_size,2*sl-1))
		self.catdelta_out = np.zeros((self.hiddenSize,2*sl-1))

	def forward(self, freq, W1, W2, W3, W4, Wcat, b1, b2, b3, bcat, alpha, beta, sentence_label):
		sl = self.sl
		D = np.dot
		'''Builds tree and computes recontruction error for each node'''
		words = self.words
		for j in range(0,sl-1):
			lens = words.shape[1]
			c1, f1, c2, f2 = words[:,0:lens-1], freq[0:lens-1], words[:,1:lens], freq[1:lens]	
			p = f(D(W1,c1)+D(W2,c2)+np.tile(b1,lens-1))		
			p_norm = p/(np.linalg.norm(p,axis=0)*np.ones(p.shape))
			y1, y2 = f(D(W3,p_norm)+np.tile(b2,lens-1)), f(D(W4,p_norm)+np.tile(b3,lens-1))	
			y1_norm, y2_norm = y1/(np.linalg.norm(y1,axis=0)*np.ones(y1.shape)), y2/(np.linalg.norm(y2,axis=0)*np.ones(y2.shape))	
			y1c1, y2c2 = alpha*(y1_norm-c1), alpha*(y2_norm-c2)
			recons_error = sum(y1c1*(y1_norm-c1)+y2c2*(y2_norm-c2))*0.5
			m, mp = np.min(recons_error), np.argmin(recons_error)
			self.y1c1[:,sl+j], self.y2c2[:,sl+j] = y1c1[:,mp], y2c2[:,mp]
			self.delta1[:,sl+j:sl+j+1], self.delta2[:,sl+j:sl+j+1] = D(fnorm_prime(y1[:,mp:mp+1]), y1c1[:,mp:mp+1]), D(fnorm_prime(y2[:,mp:mp+1]), y2c2[:,mp:mp+1])			
			index_child1, index_child2 = self.collapsed[mp], self.collapsed[mp+1]
			words = np.delete(words,mp+1,1)
			words[:,mp] = p_norm[:,mp]
			self.nodeFeatures[:,sl+j], self.nodeFeatures_unnorm[:,sl+j] = p_norm[:,mp], p[:,mp]
			self.nodeScoresR[sl+j] = m
			self.pp[index_child1], self.pp[index_child2] = sl+j, sl+j
			self.kids[sl+j,0], self.kids[sl+j,1] = index_child1, index_child2
			self.numkids[sl+j] = self.numkids[self.kids[sl+j,0]] + self.numkids[self.kids[sl+j,1]]
			self.freq = np.delete(self.freq,mp+1,0) 
			self.freq[mp] = (D(self.numkids[self.kids[sl+j,0]], f1[mp]) +  D(self.numkids[self.kids[sl+j,1]], f2[mp]))/self.numkids[sl+j]
			del self.collapsed[mp]
			self.collapsed[mp]=sl+j
		'''Classification error computation for each node'''
		out = fcat(D(Wcat,self.words)+np.tile(bcat,sl))
		diff = np.tile(sentence_label,sl)-out
		lbl_sm = (1-alpha)*diff
		score = 0.5*lbl_sm*diff
		self.nodeScores[0:sl], self.catdelta[:,0:sl] = score.T, -(lbl_sm)*fcat_prime(out)
		for i in range(sl,2*sl-1):
			sm = fcat(D(Wcat,self.nodeFeatures[:,i]) + bcat)
			lbl_sm = beta*(1-alpha)*(sentence_label-sm)
			self.catdelta[:,i] = -(lbl_sm)*fcat_prime(sm)
			J = 0.5*(D(lbl_sm.T,(sentence_label-sm)))
			self.nodeScores[i] = J
	
	def cost(self, words, W1, W2, W3, W4, Wcat, b1, b2, b3, bcat, alpha, beta, sentence_label):
		D = np.dot
		sl = self.sl
		nodeScoresR = np.zeros((2*sl-1,1))		
		nodeScores = np.zeros((2*sl-1,1))		
		nF = self.nodeFeatures.copy()
		nF[:,0:sl] = words
		for j in range(0,sl-1):
			k1, k2 = self.kids[sl+j,0], self.kids[sl+j,1]
			c1, c2 = nF[:,k1:k1+1], nF[:,k2:k2+1]
			nF[:,sl+j:sl+j+1] = fnorm(D(W1,c1)+D(W2,c2)+b1)
			y1, y2 = f(D(W3,nF[:,sl+j:sl+j+1])+b2), f(D(W4,nF[:,sl+j:sl+j+1])+b3)
			y1_norm, y2_norm = y1/(np.linalg.norm(y1,axis=0)*np.ones(y1.shape)), y2/(np.linalg.norm(y2,axis=0)*np.ones(y2.shape))	
			y1c1, y2c2 = alpha*(y1_norm-c1), alpha*(y2_norm-c2)
			nodeScoresR[sl+j] = sum(y1c1*(y1_norm-c1)+y2c2*(y2_norm-c2))*0.5
		out = fcat(D(Wcat,words)+np.tile(bcat,sl))
		diff = np.tile(sentence_label,sl)-out
		lbl_sm = (1-alpha)*diff
		score = 0.5*lbl_sm*diff
		nodeScores[0:sl] = score.T
		for i in range(sl,2*sl-1):
			sm = fcat(D(Wcat,nF[:,i]) + bcat)
			lbl_sm = beta*(1-alpha)*(sentence_label-sm)
			nodeScores[i] = 0.5*(D(lbl_sm.T,(sentence_label-sm)))
		error = (sum(nodeScoresR) + sum(nodeScores))
		return error

	def checkgradient(self, sentence, freq, eps, W1, W2, W3, W4, Wcat, We, b1, b2, b3, bcat, alpha, beta, sl):
		w = We[:,sentence]
		wa, wb = w.copy(), w.copy()
		W1a, W1b = W1.copy(), W1.copy()
		W2a, W2b = W2.copy(), W2.copy()
		W3a, W3b = W3.copy(), W3.copy()
		W4a, W4b = W4.copy(), W4.copy()
		Wcata, Wcatb = Wcat.copy(), Wcat.copy()
		#W2a[3,3], W2b[3,3] = W2[3,3] + eps, W2[3,3] - eps
		#W1a[3,3], W1b[3,3] = W1[3,3] + eps, W1[3,3] - eps
		#W3a[3,4], W3b[3,4] = W3[3,4] + eps, W3[3,4] - eps
		#W4a[2,4], W4b[2,4] = W4[2,4] + eps, W4[2,4] - eps
		#Wcata[0,0], Wcatb[0,0] = Wcat[0,0] + eps, Wcat[0,0] - eps
		wa[0,2], wb[0,2] = w[0,2] + eps, w[0,2] - eps	
		j1 = self.cost(wa,W1a,W2a,W3a,W4a,Wcata,b1,b2,b3,bcat,alpha,beta,sl)
		j2 = self.cost(wb,W1b,W2b,W3b,W4b,Wcatb,b1,b2,b3,bcat,alpha,beta,sl)
		return (j1-j2)/(2*eps)


