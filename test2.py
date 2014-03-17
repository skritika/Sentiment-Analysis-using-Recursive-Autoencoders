def f(n):
	def ret1(n):
		return n*n
	def ret2(n):
		return n+n
	return (ret1, ret2)



(f1, f2) = f(4)
print f1(n)
print f2(n)


