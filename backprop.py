from definitions import *
from functools import partial

def predict(W1, W2, W3, W4, Wcat, We, b1, b2, b3, bcat, alpha, beta, freq, test_sentence, d, num_cat):
	sl = len(test_sentence)
	L = We[:,test_sentence]
	tr = tree(sl, d, num_cat, L)
	tr.forward(freq, W1, W2, W3, W4, Wcat, b1, b2, b3, bcat, alpha, beta, 0)
	pred = fcat(np.dot(Wcat,tr.nodeFeatures[:,2*sl-2])+bcat)
	return 1*(pred>0.5)
			
def getW(t, d, num_cat, dict_length):
	if t.shape[0] is not 1:
		theta = t[np.newaxis,:]
	else:
		theta = t
	sW = (d, d)
	sb = (d, 1)
	s = d*d
	s2 = num_cat*d
	s3 = dict_length*d
	W1, W2, W3, W4, Wcat, We = theta[0,0:s].reshape(sW), theta[0,s:2*s].reshape(sW), theta[0,2*s:3*s].reshape(sW), theta[0,3*s:4*s].reshape(sW), theta[0,4*s:4*s+s2].reshape((num_cat,d)), theta[0,4*s+s2:4*s+s2+s3].reshape((d,dict_length))
	s4 = 4*s+s2+s3 
	b1, b2, b3, bcat = theta[0,s4:s4+d].reshape(sb), theta[0,s4+d:s4+2*d].reshape(sb), theta[0,s4+2*d:s4+3*d].reshape(sb), theta[0,s4+3*d:s4+3*d+num_cat].reshape((num_cat,1))
	return (W1,W2,W3,W4,Wcat,We,b1,b2,b3,bcat)


'''Backpropagation for derivative and cost computation'''
def backprop(x, training_data, training_labels, freq_original, d, num_cat, dict_length, alpha, beta, theta):
	sW = (d, d)
	sb = (d, 1)
	(W1,W2,W3,W4,Wcat,We,b1,b2,b3,bcat) = getW(theta, d, num_cat, dict_length)
	gW1, gW2, gW3, gW4, gWcat, gb1, gb2, gb3, gbcat, gWe = np.zeros(sW), np.zeros(sW), np.zeros(sW), np.zeros(sW), np.zeros((num_cat, d)), np.zeros(sb), np.zeros(sb), np.zeros(sb), np.zeros((num_cat,1)), np.zeros((d, dict_length))
	cost_J = 0.0
	for i in range(len(training_data)):
		word_indices = training_data[i]
		true_label = training_labels[i]
		sl = len(word_indices)
		L = We[:,word_indices]
		gL = np.zeros((L.shape[0],L.shape[1]))		
		freq = [freq_original[k] for k in word_indices]
		tr = tree(sl, d, num_cat, L)
		if sl>1 : 
			tr.forward(freq, W1, W2, W3, W4, Wcat, b1, b2, b3, bcat, alpha, beta, true_label)
			for current in range(2*sl-2,sl-1,-1):
				kid1, kid2 = tr.kids[current,0], tr.kids[current,1]
				a1, a1_unnorm = tr.nodeFeatures[:,current:current+1], tr.nodeFeatures_unnorm[:,current:current+1]
				d1, d2 = tr.delta1[:,current:current+1], tr.delta2[:,current:current+1]
				pd = tr.parentdelta[:,current:current+1]
				pp = tr.pp[current]
				if(current==(2*sl-2)):
					W = np.zeros((d,d))
					delt = np.zeros((d, 1))
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
				gWe[:,word_indices[j]] += gL[:,j]
			cost_J += sum(tr.nodeScores) + sum(tr.nodeScoresR)
			actual = gW1[0,2]
			tr.checkgradient(actual, word_indices, freq, 0.0000000000001, W1, W2, W3, W4, Wcat, We, b1, b2, b3, bcat, alpha, beta, true_label)
			#exit()
	F = np.ndarray.flatten
	D = np.dot
	#final grad computation
	grad_J = np.concatenate([F(gW1),F(gW2),F(gW3),F(gW4),F(gWcat),F(gWe),F(gb1),F(gb2),F(gb3),F(gbcat)],axis=1)
	grad_reg = np.concatenate([F(W1),F(W2),F(W3),F(W4),F(Wcat),F(We),np.zeros(d),np.zeros(d),np.zeros(d),np.zeros(num_cat)],axis=1)
	grad = grad_J/len(training_data) + .0004*grad_reg
	#final cost computation		
	cost_reg = .0002*(D(F(W1),F(W1).T)+D(F(W2),F(W2).T)+D(F(W3),F(W3).T)+D(F(W4),F(W4).T)+D(F(Wcat),F(Wcat).T)+D(F(We),F(We).T))
	cost = cost_J/len(training_data) + cost_reg
	if(x==1):
		return grad
	else:
		return cost[0]
	

	

