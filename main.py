from helper import *
import scipy.io
from scipy.optimize import fmin_l_bfgs_b
#loaddata("codeDataMoviesEMNLP/data/rt-polaritydata/")
#vocab = 10

#freq = [1/float(14043)]*14043
#p = parameters(hiddenSize,hiddenSize,1,10,0.2,0.5)
#p.updateparams(0, data, labels, freq)
#p.updateparams(1, data, labels, freq)
#(x, y) = p.computeder(train2, labels2, freq)
#print x.shape
#print y
#print accuracy(train2, labels2, freq, p)
#train2 = [[1,2,3], [3,4,5], [1,3,9], [2,4,2]]
#labels2 = [0, 0, 1, 1]



d = 20
(t, l) = load()
s = range(len(t))
print len(s)
np.random.shuffle(s)
s = s[0:300]
train = [t[i] for i in s]
labels = [l[i] for i in s]

print len(train)
print len(labels)
num_cat = 1
dict_length = 14043
alpha = 0.2
beta = 0.5
initv = init_theta(d,num_cat,dict_length)
print initv.shape
freq = [1/float(14043)]*14043
fgrad = partial(backprop, 1, train, labels, freq, d, num_cat, dict_length, alpha, beta)
fcost = partial(backprop, 0, train, labels, freq, d, num_cat, dict_length, alpha, beta)


#backprop(1, train, labels, freq, d, num_cat, dict_length, alpha, beta, initv)
theta_min = fmin_l_bfgs_b(fcost, initv, fprime = fgrad, args=(), maxiter=100, disp=1)[0] 
(W1,W2,W3,W4,Wcat,We,b1,b2,b3,bcat) = getW(theta_min, d, num_cat, dict_length)
#test = train
#test_labels = labels
#print accuracy(W1, W2, W3, W4, Wcat, We, b1, b2, b3, bcat, alpha, beta, freq, test, test_labels, d, num_cat)

