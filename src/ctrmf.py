import numpy as np

num_recipes = 45389

class CTRMF():
	def __init__(self, R, r, mu, bu, br, p, latentDim=1000):
		self.F = np.zeros((latentDim, 1668))
		self.R = R
		self.r = r
		self.mu = mu
		self.bu = bu
		self.br = br
		self.p = p
		self.e = np.zeros(num_recipes)
		self.br = np.zeros(num_recipes)
		self.bu = 0

	def gradient_descent(self):
		"""
		Perform gradient descent
		"""
		for r in range(num_recipes):
			self.e[r] = (self.r[r] - self.mu[r] - self.bu - self.br[r] - self.p * self.F * self.R[r])
		self.F = self.F + self.lr * sum([self.e[r] * self.R[r] * self.p for r in range(self.num_recipes)])
		for r in range(num_recipes):
			self.br[r] = self.br[r] + self.lr(self.e[r] - self.lamb * self.br[r])
		self.p = self.p + self.lr * sum([self.e[r] * self.F * self.R[r] - self.lamb * self.p for r in range(self.num_recipes)])
		self.bu = self.bu + self.lr * (self.e - self.lamb * self.bu)

