from helper import *

#loaddata("codeDataMoviesEMNLP/data/rt-polaritydata/")
#vocab = 10
freq = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
data = [[1,2,3], [3,4,5], [1,3,9], [2,4,2]]
labels = [0, 0, 1, 1]
hiddenSize = 50
p = parameters(hiddenSize,hiddenSize,1,10,0.2,0.5)
#p.updateparams(0, data, labels, freq)
#p.updateparams(1, data, labels, freq)
p.updateparams(data, labels, freq)


