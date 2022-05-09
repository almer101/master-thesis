import numpy as np
import pandas as pd
from scipy.stats import norm
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from asset_pricing import *
from option_pricing import call_option_price_bs
import asset_pricing
import option_pricing
from tqdm import tqdm

def delta_hedge_exact(S, K, r, sigma, T, pi_f = -1):
	d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
	df_dx = norm.cdf(d1)
	return - pi_f * df_dx

def delta_hedge_approx():
	pass

def simulate_hedging(ts, asset_price, riskless_asset, option_price, df_dx, K, r, sigma, T, pi_f = -1, initial_portfolio_value = 0, n_rebalancings = None):
	if n_rebalancings is None:
		n_rebalancings = len(ts)
	dn = int(round(len(ts) / n_rebalancings))

	t, S, f, b = ts[0], asset_price[0], option_price[0], riskless_asset[0]
	option_shares = [pi_f]
	pi_x = delta_hedge_exact(S, K, r, sigma, T, pi_f=pi_f)
	stock_shares = [pi_x]
	pi_r = -1.0 / b * (-initial_portfolio_value + pi_x * S + pi_f * f)
	riskless_shares = [pi_r]

	portfolio_value = [S * pi_x + b * pi_r +  f * pi_f]

	for i in range(0, len(ts)):
		if i == 0: continue

		t, S, f, b = ts[i], asset_price[i], option_price[i], riskless_asset[i]
		portfolio_value.append(S * stock_shares[-1] + b * riskless_shares[-1] + f * option_shares[-1])

		if i % dn != 0:
			# not a rebalancing time, just repeat the last weights
			stock_shares.append(stock_shares[-1])
			option_shares.append(option_shares[-1])
			riskless_shares.append(riskless_shares[-1])
			continue
		
		tau = T - t
		if tau == 0.0:
			stock_shares.append(stock_shares[-1])
			riskless_shares.append(riskless_shares[-1])
			option_shares.append(option_shares[-1])
			break

		# calculate the optimal position
		pi_x = df_dx[i]
		pi_x = delta_hedge_exact(S, K, r, sigma, tau, pi_f=pi_f)
		pi_r = -1.0 / b * (- portfolio_value[-1] + pi_x * S + pi_f * f)

		stock_shares.append(pi_x)
		riskless_shares.append(pi_r)
		option_shares.append(pi_f)

	return portfolio_value, stock_shares, option_shares, riskless_shares


def simulate_quadratic_hedging(ts, asset_price, riskless_asset, option_price, df_dx, K, r, sigma, T, lmbda, pi_f = -1, initial_portfolio_value = 0, n_rebalancings=None):
	if n_rebalancings is None:
		n_rebalancings = len(ts)
	dn = int(round(len(ts) / n_rebalancings))

	# TODO: change with the proper class for jumps
	expected_jump_size, _ = integrate.quad(lambda u: u * norm.pdf(u/0.2), -10, 10)

	t, S, f, b = ts[0], asset_price[0], option_price[0], riskless_asset[0]
	option_shares = [pi_f]
	integral = result = option_pricing.approx_integrate(lambda u: u * (option_pricing.calculate_option_price(S * (1+u), K, r, T, sigma, lmbda, normal_jump, n_path_simulations=200) - f) * norm.pdf(u/0.2), -0.99999, 4, n=25)
	sensitivity = delta_hedge_exact(S, K, r, sigma, T, pi_f=pi_f)
	pi_x = (sigma**2 * sensitivity + 1.0/S * integral) / (sigma ** 2 + expected_jump_size)
	stock_shares = [pi_x]
	pi_r = -1.0 / b * (-initial_portfolio_value + pi_x * S + pi_f * f)
	riskless_shares = [pi_r]

	portfolio_value = [S * pi_x + b * pi_r +  f * pi_f]

	for i in range(0, len(ts)):
		if i == 0: continue

		t, S, f, b = ts[i], asset_price[i], option_price[i], riskless_asset[i]
		portfolio_value.append(S * stock_shares[-1] + b * riskless_shares[-1] + f * option_shares[-1])

		if i % dn != 0:
			# not a rebalancing time, just repeat the last weights
			stock_shares.append(stock_shares[-1])
			option_shares.append(option_shares[-1])
			riskless_shares.append(riskless_shares[-1])
			continue
		
		tau = T - t
		if tau == 0.0:
			stock_shares.append(stock_shares[-1])
			riskless_shares.append(riskless_shares[-1])
			option_shares.append(option_shares[-1])
			break

		# calculate the optimal position
		integral = result = option_pricing.approx_integrate(lambda u: u * (option_pricing.calculate_option_price(S * (1+u), K, r, T - ts[i+1], sigma, lmbda, normal_jump, n_path_simulations=200) - f) * norm.pdf(u/0.2), -0.99999, 4, n=25)
		sensitivity = df_dx[i]
		sensitivity = delta_hedge_exact(S, K, r, sigma, tau, pi_f=pi_f)
		phi = (sigma**2 * sensitivity + 1.0/S * integral) / (sigma ** 2 + expected_jump_size)

		pi_x = phi
		pi_r = -1.0 / b * (- portfolio_value[-1] + pi_x * S + pi_f * f)

		stock_shares.append(pi_x)
		riskless_shares.append(pi_r)
		option_shares.append(pi_f)

	return portfolio_value, stock_shares, option_shares, riskless_shares


if __name__ == "__main__":
	n = 252
	T = 1.0
	K = 100
	r = 0.02
	mu = 0.05
	lmbda = 4
	sigma = 0.5

	ts = np.linspace(0.0, T, n)
	# asset_price = geometric_bm_path(T=T, n=n, mu=mu, sigma=sigma, x0=100)
	asset_price, arrivals = jump_diffusion_process(T=T, n=n, lmbda=lmbda, mu=mu, sigma=sigma, x0=100, jump_size=asset_pricing.normal_jump)
	
	# print(arrivals)
	# plt.plot(ts, asset_price)
	# plt.show()
	# df = pd.DataFrame(columns=['t', 'x'])
	# df['t'] = ts
	# df['x'] = asset_price
	# df.to_csv('jump_diffusion.csv')

	df = pd.read_csv('jump_diffusion_call_lmbda8_n1000.csv')
	ts = df['t'].values
	asset_price = df['x'].values
	call_price = df['f(x)'].values

	riskless_asset = list(map(lambda t: 1.0 * np.exp(r * t / T), ts))

	df_dx = [None]
	for i in range(1, len(asset_price)):
		df_dx.append((call_price[i] - call_price[i-1]) / (asset_price[i] - asset_price[i-1]))

	ts = ts[1:]
	asset_price = asset_price[1:]
	call_price = call_price[1:]
	df_dx = df_dx[1:]

	# for i in tqdm(range(len(ts))):
	# 	t = ts[i]
	# 	S = asset_price[i]
	# 	tau = T - t
	# 	if tau == 0: # we are at time T the price is equal to the payoff which is (S - K)+
	# 		call_price.append(max(0, S - K))
	# 	else:
	# 		# c = call_option_price_bs(S, K, r, tau, sigma)
	# 		c = option_pricing.calculate_option_price(S, K, r, tau, sigma, lmbda=lmbda, jump_size=asset_pricing.normal_jump, n_path_simulations=200)
	# 		call_price.append(c)

	# df = pd.DataFrame(columns=['t', 'x'])
	# df['t'] = ts
	# df['x'] = asset_price
	# df.to_csv('jump_diffusion.csv')
	portfolio_value, stock_shares, option_shares, riskless_shares = simulate_hedging(ts, asset_price, riskless_asset, call_price, df_dx, K, r, sigma, T, pi_f=-1, initial_portfolio_value=50, n_rebalancings=3) # short one option

	# plt.plot(ts, portfolio_value)
	# plt.plot(ts, stock_shares, label='stock shares')
	# plt.plot(ts, option_shares, label='option shares')
	# plt.plot(ts, riskless_shares, label='riskless shares')
	plt.plot(ts, asset_price, label='asset price')
	plt.plot(ts, portfolio_value, label='portfolio_value')
	# plt.plot(ts, call_price, label='call price')
	plt.show()
