import warnings
warnings.filterwarnings("error")
import numpy as np
import scipy.integrate as integrate
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from numpy.random import normal, standard_normal, exponential
from asset_pricing import normal_jump, poisson_arrival_times
from tqdm import tqdm

def payoff(t=0, T=1.0, n=1000, r=0.02, lmbda=4, mu=0.05, sigma=0.5, x0=100, K=100.0, jump_size=None):
	arrivals = poisson_arrival_times(lmbda=lmbda)
	next_arrival_index = 0

	sharpe = (mu - r) / sigma

	dt = T/n
	path = np.zeros(n)
	path[0] = x0
	for i in range(1,n):
		diffusion_diff = (r - lmbda * 0) * path[i-1] * dt + sigma * path[i-1] * (normal(scale=np.sqrt(dt)) + sharpe * dt)
		path[i] = path[i-1] + diffusion_diff
		if next_arrival_index < len(arrivals) and arrivals[next_arrival_index] <= dt * i:
			path[i] += path[i-1] * jump_size()
			next_arrival_index += 1
	
	xT = path[-1]
	#call
	payoff = max(0.0, xT - K)
	return payoff


def option_price(t=0, T=1.0, n=1000, r=0.02, lmbda=4, mu=0.05, sigma=0.5, x0=100, K=100, jump_size=None):
	if x0 <= 0: raise ValueError('Starting price of an asset must be positive')
	if jump_size is None: raise ValueError('Jump size function has to be provided')

	arrivals = poisson_arrival_times(lmbda=lmbda)
	next_arrival_index = 0

	sharpe = (mu - r) / sigma

	dt = T/n
	path = np.zeros(n)
	path[0] = x0
	for i in range(1,n):
		diffusion_diff = (r - lmbda * 0) * path[i-1] * dt + sigma * path[i-1] * (normal(scale=np.sqrt(dt)) + sharpe * dt)
		path[i] = path[i-1] + diffusion_diff
		if next_arrival_index < len(arrivals) and arrivals[next_arrival_index] <= dt * i:
			path[i] += path[i-1] * jump_size()
			next_arrival_index += 1
	
	return path, arrivals

def call_option_price_bs(S, K, r, T, sigma):
	d1 = (np.log(S / K) + (r + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))
	d2 = d1 - sigma * np.sqrt(T)
	return norm.cdf(d1)*S - norm.cdf(d2)*K*np.exp(-r*T)


def calculate_option_price(S, K, r, T, sigma, lmbda, jump_size, n_path_simulations=500):
	if lmbda == 0:
		return call_option_price_bs(S=S, K=K, r=r, T=T, sigma=sigma)

	n_iter = optimal_n_jumps(lmbda)
	prices = []

	for n in range(n_iter):
		c_sum = 0.0
		for i in range(n_path_simulations):
			product = 1.0
			for j in range(n):
				product *= 1 + jump_size()

			E_jump_size = 0 # TODO: change this
			try:
				c = call_option_price_bs(S = S*np.exp(-lmbda * E_jump_size * T) * product, K = K, r = r, T = T, sigma = sigma)
			except RuntimeWarning:
				print(f'S = ')
				print(f'S_new = {S*np.exp(-lmbda * E_jump_size * T) * product}')
				print(f'product = {product}')
				print()
			c_sum += c

		element = (c_sum / n_path_simulations) * np.exp(-lmbda*T) * (lmbda * T)**n / (np.math.factorial(n))
		prices.append(element)

	return np.sum(prices)


def calculate_option_price_path(asset_price, K, r, T, sigma, lmbda, jump_size, n_path_simulations=200):
	option_prices = []
	ts = np.linspace(0, T, len(asset_price))

	for i in tqdm(range(len(ts))):
		t = ts[i]
		S = asset_price[i]
		tau = T - t
		if tau == 0: # we are at time T the price is equal to the payoff which is (S - K)+
			option_prices.append(max(0, S - K))
		else:
			# c = call_option_price_bs(S, K, r, tau, sigma)
			c = calculate_option_price(S, K, r, tau, sigma, lmbda=lmbda, jump_size=jump_size, n_path_simulations=n_path_simulations)
			option_prices.append(c)

	return option_prices


def approx_integrate(function, a, b, n=50):
	dx = (b - a) / n
	sum_ = 0.0
	for i in tqdm(range(n)):
		ksi = a + dx*i + dx / 2.0
		sum_ += function(ksi) * dx
	return sum_


def optimal_n_jumps(lmbda, threshold=1e-02):
	optimal_n_jumps = lmbda
	prob = np.exp(-lmbda * 1.0) * (lmbda * 1.0)**optimal_n_jumps / np.math.factorial(optimal_n_jumps)
	while prob > threshold:
		optimal_n_jumps += 1
		prob = np.exp(-lmbda * 1.0) * (lmbda * 1.0)**optimal_n_jumps / np.math.factorial(optimal_n_jumps)
	return optimal_n_jumps


if __name__ == "__main__":
	spot_prices = np.linspace(1, 140, 30)
	risk_free_rate = 0.02
	lmbdas = [0,1,4]
	mu = 0.05
	sigma = 0.5
	K = 100
	lmbda = 4
	S = 100
	T = 1.0


	a = -3.345
	b = 5.34
	result = integrate.quad(lambda x: np.sin(x), a, b)
	my_result = approx_integrate(lambda x: np.sin(x), a, b, n=50)

	print(result)
	print(my_result)
	print(abs(result[0] - my_result) / result[0])

	# exit()

	# print("Start")
	# c_t = calculate_option_price(S, K, risk_free_rate, T, sigma, lmbda, normal_jump)

	# result = approx_integrate(lambda u: u * (calculate_option_price(S * (1+u), K, risk_free_rate, T, sigma, lmbda, normal_jump, n_path_simulations=200) - c_t) * norm.pdf(u/sigma), -0.99999, 5, n=25)
	# print(result)
	# print(result / S)
	# exit()
	# # print(result)
	# ns = [200, 300, 500, 1000, 2000]
	# prices = []
	# for n in tqdm(ns):
	# 	price = calculate_option_price(S, K, risk_free_rate, T, sigma, lmbda, normal_jump, n_path_simulations=n)
	# 	prices.append(price)
	
	# plt.plot(ns, prices)
	# plt.show()
	# exit()

	for lmbda in lmbdas:
		call_prices = []
		for spot in tqdm(spot_prices):
			price = calculate_option_price(S=spot, K=K, r=risk_free_rate, T=1.0, sigma=sigma, lmbda=lmbda, jump_size=normal_jump)
			call_prices.append(price)

		plt.plot(spot_prices, call_prices, label=f'lambda={lmbda}')	
	
	plt.legend()
	plt.show()

	exit()
	
	for lmbda in lmbdas:
		call_prices = []
		for spot in tqdm(spot_prices):
			payoffs = []
			for i in range(4000):
				p = payoff(r=risk_free_rate, lmbda=lmbda, mu=0.05, sigma=0.5, x0=spot, K=100, jump_size=normal_jump)
				payoffs.append(p)

			payoffs = np.array(payoffs)
			expected = payoffs.mean()
			price = expected * np.exp(-risk_free_rate * 1.0)
			call_prices.append(price)

		plt.plot(spot_prices, call_prices, label=f'lambda={lmbda}')

		# df = pd.DataFrame(columns=['x', 'y'])
		# df['x'] = spot_prices
		# df['y'] = call_prices
		# df.to_csv(f'../data/option_price_lambda_{lmbda}.dat', sep=' ', header=False, index=False)

	plt.legend()
	plt.show()

