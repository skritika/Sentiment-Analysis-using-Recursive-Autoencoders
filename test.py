import numpy as np

def f(x):
	(n1, n2) = x.shape
	f = np.tanh(x)
	norm = np.linalg.norm(f,axis=0)*np.ones((n1,n2))
	fnorm = f/norm
	#return f
	return fnorm	

def f_prime(x):
	f = np.tanh(x)
	f_p = 1 - np.square(f)
	diag = np.diagflat(f_p)
	norm = np.linalg.norm(f)
	fnorm_p = diag/norm - np.dot(diag, np.dot(f,f.T))/np.power(norm,3)
	return fnorm_p

a = np.ones((4,1))
a[1,0], a[2,0], a[3,0] = 2, 3, 4
print np.linalg.norm(f(a))
