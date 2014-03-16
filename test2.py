def f(n):
	x = n*n
	y = n+n
	def ret1():
		return x
	def ret2():
		return y
	return (ret1, ret2)



(f1, f2) = f(4)
print f1()
print f2()


