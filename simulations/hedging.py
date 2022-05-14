import numpy as np
from functools import reduce
import pandas as pd
from scipy.stats import norm
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from asset_pricing import *
from option_pricing import *
from tqdm import tqdm
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.interpolate import interp1d

def delta_hedge_exact(S, K, r, sigma, T, pi_f = -1):
	d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
	df_dx = norm.cdf(d1)
	return - pi_f * df_dx


def simulate_hedging(ts, asset_price, option_price, K, r, sigma, T, pi_f = -1, initial_portfolio_value = 0, n_rebalancings = None):
	if n_rebalancings is None:
		n_rebalancings = len(ts)
	dn = int(round(len(ts) / n_rebalancings))

	riskless_asset = list(map(lambda t: 1.0 * np.exp(r * t / T), ts))

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
		pi_x = delta_hedge_exact(S, K, r, sigma, tau, pi_f=pi_f)
		pi_r = -1.0 / b * (- portfolio_value[-1] + pi_x * S + pi_f * f)

		stock_shares.append(pi_x)
		riskless_shares.append(pi_r)
		option_shares.append(pi_f)

	return portfolio_value, stock_shares, option_shares, riskless_shares


def simulate_quadratic_hedging(ts, asset_price, option_price, K, r, sigma, T, lmbda, pi_f = -1, initial_portfolio_value = 0, n_rebalancings=None):
	if n_rebalancings is None:
		n_rebalancings = len(ts)
	dn = int(round(len(ts) / n_rebalancings))

	riskless_asset = list(map(lambda t: 1.0 * np.exp(r * t / T), ts))

	jump = jumps.NormalJump(mean=0, std=0.2)

	t, S, f, b = ts[0], asset_price[0], option_price[0], riskless_asset[0]
	option_shares = [pi_f]

	integral = option_pricing.approx_integrate(lambda u: u * (option_pricing.calculate_option_price(S * (1+u), K, r, T - ts[1], sigma, lmbda, normal_jump, n_path_simulations=200) - f) * jump.pdf(u), -0.99999, 4, n=25)
	sensitivity = delta_hedge_exact(S, K, r, sigma, T, pi_f=pi_f)
	pi_x = (sigma**2 * sensitivity + 1.0/S * integral) / (sigma ** 2 + jump.expected_value())
	stock_shares = [pi_x]

	pi_r = -1.0 / b * (-initial_portfolio_value + pi_x * S + pi_f * f)
	riskless_shares = [pi_r]

	portfolio_value = [S * pi_x + b * pi_r +  f * pi_f]

	for i in range(0, len(ts)):
		if i == 0: continue

		t, S, f, b = ts[i], asset_price[i], option_price[i], riskless_asset[i]
		portfolio_value.append(S * stock_shares[-1] + b * riskless_shares[-1] + f * option_shares[-1])

		tau = T - t
		if i % dn != 0 or tau == 0.0:
			# not a rebalancing time, just repeat the last weights
			stock_shares.append(stock_shares[-1])
			option_shares.append(option_shares[-1])
			riskless_shares.append(riskless_shares[-1])
			continue

		# calculate the optimal position
		integral = option_pricing.approx_integrate(lambda u: u * (option_pricing.calculate_option_price(S * (1+u), K, r, T - ts[i], sigma, lmbda, normal_jump, n_path_simulations=200) - f) * norm.pdf(u/0.2), -0.99999, 4, n=25)
		sensitivity = delta_hedge_exact(S, K, r, sigma, tau, pi_f=pi_f)
		phi = (sigma**2 * sensitivity + 1.0/S * integral) / (sigma ** 2 + jump.expected_value())

		pi_x = phi
		pi_r = -1.0 / b * (- portfolio_value[-1] + pi_x * S + pi_f * f)

		stock_shares.append(pi_x)
		riskless_shares.append(pi_r)
		option_shares.append(pi_f)

	return portfolio_value, stock_shares, option_shares, riskless_shares

def get_pdf(cdf):
	pdf_x, pdf_y = [], []
	for i in range(1, len(cdf.x)):
		pdf_x.append(0.5 * (cdf.x[i-1] + cdf.x[i]))
		pdf_y.append((cdf.y[i] - cdf.y[i-1]) / (cdf.x[i] - cdf.x[i-1]))
	return pdf_x, pdf_y

def show_gbm_hedging_result():
	T = 1.0
	n = 1000
	mu = 0.05
	sigma = 0.5
	r = 0.02
	K = 120

	# ts = np.linspace(0.0, T, n)
	# asset_price = geometric_bm_path(T=T, n=n, mu=mu, sigma=sigma, x0=100)
	# call_price = calculate_option_price_path(asset_price, K=K, r=r, T=T, sigma=sigma)

	# plt.plot(ts, asset_price)
	# plt.plot(ts, call_price)
	# plt.grid()
	# plt.show()

	# a = input('save(y/n)?')
	# if a.strip().lower() == 'y':
	# 	df = pd.DataFrame(columns=['t', 'x', 'call_price'])
	# 	df['t'] = ts
	# 	df['x'] = asset_price
	# 	df['call_price'] = call_price
	# 	df.to_csv('gbm_prices_path.csv', index = False)

	df = pd.read_csv('gbm_prices_path.csv')

	for n_rebalancings in [1,2,5,10]:
		portfolio_value, stock_shares, option_shares, riskless_shares = simulate_hedging(df['t'], df['x'], option_price=df['call_price'], K=K, r=r, sigma=sigma, T=T, pi_f = -1, initial_portfolio_value = 50, n_rebalancings = n_rebalancings)
		df[f'portfolio_value_n{n_rebalancings}'] = portfolio_value

	# plt.plot(df['t'], df['x'])
	# plt.plot(df['t'], df['call_price'])
	# plt.plot(df['t'], portfolio_value)
	# plt.grid()
	# plt.show()

	df.to_csv('../data/gbm_hedging_results.dat', index=False, sep=' ')

	plt.plot(df['t'], stock_shares, label='stock shares')
	plt.plot(df['t'], option_shares, label='option shares')
	# plt.plot(df['t'], riskless_shares, label='riskless asset shares')
	plt.grid()
	plt.legend()
	plt.show()

def gbm_hedging():
	T = 1.0
	n = 1000
	mu = 0.05
	sigma = 0.5
	r = 0.02
	K = 120

	ns_rebalancings = [1, 2, 4, 10, 50, 100, 500]
	ts = np.linspace(0.0, T, n)
	df = pd.DataFrame(columns = list(map(lambda e: f'portfolio_{e}', ns_rebalancings)))

	results = {}

	for i in range(len(ns_rebalancings)):
		print(f"({i+1}/{len(ns_rebalancings)})")
		n_rebalancings = ns_rebalancings[i]

		results[n_rebalancings] = []
		for i in tqdm(range(200)):
			asset_price = geometric_bm_path(T=T, n=n, mu=mu, sigma=sigma, x0=100)
			call_price = calculate_option_price_path(asset_price, K=K, r=r, T=T, sigma=sigma)

			portfolio_value, _, _, _ = simulate_hedging(ts, asset_price, option_price=call_price, K=K, r=r, sigma=sigma, T=T, pi_f = -1, initial_portfolio_value = 50, n_rebalancings = n_rebalancings)
			results[n_rebalancings].append(portfolio_value[-1])

		df[f'portfolio_{n_rebalancings}'] = results[n_rebalancings]

	df.to_csv('gbm_portfolio_values.csv', index=False)


def analyze_hedging_results():
	ns_rebalancings = [1, 2, 4, 10, 50, 100, 500]
	df = pd.read_csv('gbm_portfolio_values.csv')

	pdfs = {}
	results = {}

	for n_rebalancings in ns_rebalancings:
		n_bins = 10
		count_c, bins_c = np.histogram(df[f'portfolio_{n_rebalancings}'], bins=n_bins)
		pdf = count_c/ (np.sum(count_c) * np.diff(bins_c)) 
		dxc = np.diff(bins_c)[0]
		xc = bins_c[0:-1] + 0.5*dxc
		# plt.hist(df[f'portfolio_{n_rebalancings}'], bins=n_bins, density=True)
		plt.plot(xc, pdf)

		pdfs[n_rebalancings] = (xc, pdf)


	plt.grid()
	plt.legend()
	plt.show()

	# save the pdfs to df and csv
	columns = reduce(lambda value, e: value + e, map(lambda e: [f'x_{e}', f'pdf_{e}'], ns_rebalancings), [])
	df = pd.DataFrame(columns=columns)
	for n_rebalancings in ns_rebalancings:
		xs, pdf = pdfs[n_rebalancings]
		df[f'x_{n_rebalancings}'] = xs
		df[f'pdf_{n_rebalancings}'] = pdf

	df.to_csv('../data/gbm_hedging_distributions.dat', index=False, sep=' ')

if __name__ == "__main__":
	
	# gbm_hedging()
	# show_gbm_hedging_result()
	analyze_hedging_results()
	exit()

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
	portfolio_value, stock_shares, option_shares, riskless_shares = simulate_hedging(ts, asset_price, riskless_asset, call_price, K, r, sigma, T, pi_f=-1, initial_portfolio_value=50, n_rebalancings=3) # short one option

	# plt.plot(ts, portfolio_value)
	# plt.plot(ts, stock_shares, label='stock shares')
	# plt.plot(ts, option_shares, label='option shares')
	# plt.plot(ts, riskless_shares, label='riskless shares')
	plt.plot(ts, asset_price, label='asset price')
	plt.plot(ts, portfolio_value, label='portfolio_value')
	# plt.plot(ts, call_price, label='call price')
	plt.show()
