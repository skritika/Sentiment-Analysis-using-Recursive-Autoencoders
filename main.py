from helper import *
import scipy.io
from scipy.optimize import fmin_l_bfgs_b

d = 20
(t, l) = load()
s = range(len(t))
np.random.shuffle(s)
data = [t[i] for i in s]
labels = [l[i] for i in s]

train_data = data[0:8]
train_labels = labels[0:8]
test_data = data[8:10]
test_labels = labels[8:10]


num_cat = 1
dict_length = 14043
alpha = 0.2
beta = 0.5
initv = init_theta(d,num_cat,dict_length)
freq = [1/float(14043)]*14043
fgrad = partial(backprop, 1, train_data, train_labels, freq, d, num_cat, dict_length, alpha, beta)
fcost = partial(backprop, 0, train_data, train_labels, freq, d, num_cat, dict_length, alpha, beta)
backprop(1, train_data, train_labels, freq, d, num_cat, dict_length, alpha, beta, initv)
theta_min = fmin_l_bfgs_b(fcost, initv, fprime = fgrad, args=(), maxiter=100, disp=1)[0] 
(W1,W2,W3,W4,Wcat,We,b1,b2,b3,bcat) = getW(theta_min, d, num_cat, dict_length)
print "Accuracy on the test set is", accuracy(W1, W2, W3, W4, Wcat, We, b1, b2, b3, bcat, alpha, beta, freq, test_data, test_labels, d, num_cat)

