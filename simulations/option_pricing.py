import numpy as np
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


if __name__ == "__main__":
	spot_prices = np.linspace(1, 140, 20)
	lmbdas = [1,4]
	
	for lmbda in lmbdas:
		call_prices = []
		for spot in tqdm(spot_prices):
			payoffs = []
			for i in range(1000):
				p = payoff(lmbda=1, x0=spot, K=100, jump_size=normal_jump)
				payoffs.append(p)

			payoffs = np.array(payoffs)
			expected = payoffs.mean()
			price = expected * np.exp(-0.02 * 1.0)
			call_prices.append(price)

		plt.plot(spot_prices, call_prices, label=f'lambda={lmbda}')
	
	plt.legend()
	plt.show()

