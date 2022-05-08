import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from asset_pricing import *
from option_pricing import call_option_price_bs
import asset_pricing
import option_pricing
from tqdm import tqdm


def test_optimal_n_simulations_for_jumps():
	n = 252
	T = 1.0
	K = 100
	r = 0.02
	mu = 0.05
	lmbda = 4
	sigma = 0.5

	n_sims = [50, 100, 200, 300, 500, 1000]
	S = 60
	std_devs = []
	means = []
	for n_sim in tqdm(n_sims):
		prices = []
		for i in range(30):
			price = option_pricing.calculate_option_price(S=S, K=K, r=r, T=T, sigma=sigma, lmbda=lmbda, jump_size=normal_jump, n_path_simulations=n_sim)
			prices.append(price)
		std_devs.append(np.std(prices))
		means.append(np.mean(prices))

	print(n_sims)
	print(means)
	print(std_devs)

	df = pd.DataFrame(columns=['N_simulations', 'Mean price', 'Price Std dev'])
	df['N_simulations'] = n_sims
	df['Mean price'] = means
	df['Price Std dev'] = std_devs

	df.to_csv(f"n_simulations_test_S={S}.csv")

if __name__ == "__main__":
	# df = pd.read_csv('jump_diffusion_call_n1000_K100.csv')
	df = pd.read_csv('jump_diffusion_call_K100.csv')
	plt.plot(df['t'], df['x'], label='asset price')
	plt.plot(df['t'], df['f(x)'], label = 'call option price')

	plt.grid()
	plt.legend()
	plt.show()
	
	n = 1000
	n_rebalancings = 24

	dn = int(round(n / n_rebalancings))
	print(dn)
	print(dn * (n_rebalancings - 1))
	
	for i in range(0, n, dn):
		print(i)

