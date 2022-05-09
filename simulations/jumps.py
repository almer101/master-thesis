from numpy.random import normal
from scipy.stats import norm

class NormalJump:
	def __init__(self, mean, std):
		self.mean = mean
		self.std = std
		self.min_size = -0.99999

	def generate(self):
		jump = normal(loc=self.mean, scale=self.std)
		if jump < self.min_size:
			jump = self.min_size
		return jump

	def pdf(self, x):
		return norm.pdf(x, self.mean, self.std)

	def cdf(self, x):
		return norm.cdf(x, self.mean, self.std)

	def expected_value(self):
		return self.mean


class ConstantJump:
	def __init__(self, jump_size):
		self.jump_size = jump_size

	def generate(self):
		return self.jump_size

	def cdf(self, x):
		return 1.0 if x == jump_size else 0.0

	def expected_value(self):
		return self.jump_size