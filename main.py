from helper import *
import scipy.io
#loaddata("codeDataMoviesEMNLP/data/rt-polaritydata/")
#vocab = 10

(train, labels) = load()
#freq = [1/float(14043)]*14043
freq = [0.1]*10
train2 = [[1,2,3], [3,4,5], [1,3,9], [2,4,2]]
labels2 = [0, 0, 1, 1]
hiddenSize = 5
#p = parameters(hiddenSize,hiddenSize,1,10,0.2,0.5)
#p.updateparams(0, data, labels, freq)
#p.updateparams(1, data, labels, freq)
#(x, y) = p.computeder(train2, labels2, freq)
#print x.shape
#print y
#print accuracy(train2, labels2, freq, p)
initv = init_theta(5,1,10)
print initv.shape
backprop(initv, train2, labels2, freq, 5, 1, 10, 0.2, 0.5)


