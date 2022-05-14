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
import jumps

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


def calculate_option_price(S, K, r, T, sigma, lmbda, jump, n_path_simulations=1000):
	if lmbda == 0:
		return call_option_price_bs(S=S, K=K, r=r, T=T, sigma=sigma)

	n_iter = optimal_n_jumps(lmbda)
	prices = []

	for n in range(n_iter):
		c_sum = 0.0
		for i in range(n_path_simulations):
			product = 1.0
			for j in range(n):
				product *= 1 + jump.generate()

			c = call_option_price_bs(S = S*np.exp(-lmbda * jump.expected_value() * T) * product, K = K, r = r, T = T, sigma = sigma)
			c_sum += c

		element = (c_sum / n_path_simulations) * np.exp(-lmbda*T) * (lmbda * T)**n / (np.math.factorial(n))
		prices.append(element)

	return np.sum(prices)


def calculate_option_price_path(asset_price, K, r, T, sigma, lmbda=0, jump=None, n_path_simulations=200):
	option_prices = []
	ts = np.linspace(0, T, len(asset_price))

	# for i in tqdm(range(len(ts))):
	for i in range(len(ts)):
		t = ts[i]
		S = asset_price[i]
		tau = T - t
		if tau == 0: # we are at time T the price is equal to the payoff which is (S - K)+
			option_prices.append(max(0, S - K))
		else:
			# c = call_option_price_bs(S, K, r, tau, sigma)
			c = calculate_option_price(S, K, r, tau, sigma, lmbda=lmbda, jump=jump, n_path_simulations=n_path_simulations)
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
	
	risk_free_rate = 0.02
	lmbdas = [0,1,2,4,8]
	mu = 0.05
	sigma = 0.5
	K = 100
	lmbda = 4
	S = 100
	T = 1.0

	df = pd.read_csv('../data/jump_diffusion_process.dat', sep=' ')
	print(df)

	call_price_path = calculate_option_price_path(asset_price=df['x'], K=100, r=0.02, T=1.0, sigma=0.4, lmbda=4, jump=jumps.NormalJump(mean=0, std=0.2), n_path_simulations=200)

	df['call_price'] = call_price_path
	plt.plot(df['t'], df['x'])
	plt.plot(df['t'], call_price_path)

	df.to_csv('../data/jump_diffusion_process.dat', index = False, sep=' ')

	plt.grid()
	plt.show()

	exit()

	#################################  SURFACE !!! #################################
	# normal_jump = jumps.NormalJump(mean=0.05, std=0.3)
	# mu = 0.05
	# sigma = 0.5
	# K = 95
	# lmbda = 4
	# T=1.0
	# spot_prices = np.linspace(1, 140, 20)
	# taus = np.linspace(0, T, 20)[::-1]
	# # tau, spot_price, call_price
	# rows = []
	# for i in range(len(taus)):
	# 	tau = taus[i]
	# 	print(f'({i+1}/{len(taus)})')
	# 	for s in tqdm(spot_prices):
	# 		if tau == 0.0:
	# 			rows.append((tau, s, max(0, s - K)))
	# 		else:
	# 			price = calculate_option_price(S=s, K=K, r=risk_free_rate, T=tau, sigma=sigma, lmbda=lmbda, jump=normal_jump)
	# 			rows.append((tau, s, price))
	# df = pd.DataFrame(np.vstack(rows), columns=['time_to_maturity', 'spot_price', 'call_price'])
	# df.to_csv('call_price_surface.csv')

	# exit()

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
	
	df = pd.DataFrame(columns=['spot_price'] + list(map(lambda l: f'lambda_{l}', lmbdas)))
	normal_jump = jumps.NormalJump(mean=0.05, std=0.3)
	spot_prices = np.linspace(1, 140, 30)
	df['spot_price'] = spot_prices
	for i in range(len(lmbdas)):
		lmbda = lmbdas[i]
		print(f'({i+1}/{len(lmbdas)})')
		call_prices = []
		for spot in tqdm(spot_prices):
			price = calculate_option_price(S=spot, K=K, r=risk_free_rate, T=1.0, sigma=sigma, lmbda=lmbda, jump=normal_jump)
			call_prices.append(price)

		df[f'lambda_{lmbda}'] = call_prices
		plt.plot(spot_prices, call_prices, label=f'lambda={lmbda}')	
	
	# df.to_csv('call_prices_for_different_lambdas.csv')
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

