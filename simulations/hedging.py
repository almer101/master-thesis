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
from scipy.stats import ttest_rel, ttest_1samp

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


def simulate_quadratic_hedging(ts, asset_price, option_price, K, r, sigma, T, pi_f = -1, initial_portfolio_value = 0, n_rebalancings=None):	
	if n_rebalancings is None:
		n_rebalancings = len(ts)
	dn = int(round(len(ts) / n_rebalancings))

	riskless_asset = list(map(lambda t: 1.0 * np.exp(r * t / T), ts))

	jump = jumps.NormalJump(mean=0, std=0.2)

	t, S, f, b = ts[0], asset_price[0], option_price[0], riskless_asset[0]
	option_shares = [pi_f]

	c_no_jump = call_option_price_bs(S, K, r, T, sigma)
	integral = integrate.quad(lambda u: u * jump.pdf(u) * (call_option_price_bs(S * (1+u), K, r, T, sigma) - c_no_jump), -0.99999, 5)[0]
	squared_expectation = integrate.quad(lambda u: u**2 * jump.pdf(u), -0.99999, 5)[0]
	sensitivity = delta_hedge_exact(S, K, r, sigma, T, pi_f=pi_f)

	pi_x = (sigma**2 * sensitivity + 1.0/S * integral) / (sigma ** 2 + squared_expectation)
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
		c_no_jump = call_option_price_bs(S, K, r, tau, sigma)
		integral = integrate.quad(lambda u: u * jump.pdf(u) * (call_option_price_bs(S * (1+(max(u, -0.99999))), K, r, tau, sigma) - c_no_jump), -4, 4)[0]
		squared_expectation = integrate.quad(lambda u: u**2 * jump.pdf(u), -0.99999, 4)[0]

		sensitivity = delta_hedge_exact(S, K, r, sigma, tau, pi_f=pi_f)
		phi_x = (sigma**2 * sensitivity + 1.0/S * integral) / (sigma ** 2 + squared_expectation)
		# phi_x = delta_hedge_exact(S, K, r, sigma, tau, pi_f=pi_f)

		pi_x = phi_x
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


def compare_quadratic_and_delta():
	df = pd.read_csv('../data/jump_diffusion_process.dat', sep=' ')
	print(df)

	portfolio_value, stock_shares, option_shares, riskless_shares = simulate_hedging(df['t'], df['x'], df['call_price'], K=100, r=0.02, sigma=0.4, T=1.0, pi_f = -1, initial_portfolio_value = 50, n_rebalancings = 100)
	portfolio_value1, stock_shares1, option_shares1, riskless_shares1 = simulate_quadratic_hedging(df['t'], df['x'], df['call_price'], K=100, r=0.02, sigma=0.4, T=1.0, pi_f = -1, initial_portfolio_value = 50, n_rebalancings=100)

	plt.plot(df['t'], portfolio_value1, label='quadratic')
	plt.plot(df['t'], portfolio_value, label='delta')
	plt.grid()
	plt.legend()
	plt.show()
	
	jump = jumps.NormalJump(mean=0, std=0.2)
	
	S = 100
	K = 100
	r = 0.02
	tau = 1.0
	sigma = 0.4
	c_no_jump = call_option_price_bs(S, K, r, tau, sigma)

	# call_option_price_bs(S, K, r, T, sigma)
	integral = integrate.quad(lambda u: u * jump.pdf(u) * (call_option_price_bs(S * (1+u), K, r, tau, sigma) - c_no_jump), -0.99999, 5)[0]
	squared_expectation = integrate.quad(lambda u: u**2 * jump.pdf(u), -0.99999, 5)[0]
	
	# integral = integrate(lambda u: u * (option_pricing.calculate_option_price(S * (1+u), K, r, T - ts[i], sigma, lmbda, normal_jump, n_path_simulations=200) - f) * norm.pdf(u/0.2), -0.99999, 4, n=25)
	sensitivity = delta_hedge_exact(S, K, r, sigma, tau, pi_f=-1)
	phi = (sigma**2 * sensitivity + 1.0/S * integral) / (sigma ** 2 + squared_expectation)

	print(phi)
	new_df = pd.DataFrame(columns=['t', 'delta', 'quadratic'])
	new_df['t'] = df['t']
	new_df['delta'] = portfolio_value
	new_df['quadratic'] = portfolio_value1

	new_df.to_csv('../data/one_process_hedging_comparison.csv', index=False, sep=' ')


def generate_jump_diffusion_paths():
	n_samples = 30

	T = 1.0
	lmbda = 4
	mu = 0.05
	sigma = 0.5
	K = 100
	x0 = 100
	n = 1000
	r = 0.02
	normal_jump = jumps.NormalJump(mean=0.05, std=0.22)

	columns = reduce(lambda value, e: value + e, map(lambda e: [f'x_{e+1}', f'call_price_{e+1}'], range(n_samples)), [])
	columns = ['t'] + columns
	df = pd.DataFrame(columns=['t'])

	ts = np.linspace(0, T, n)
	df['t'] = ts

	for i in range(20, n_samples):
		# N(mean=0.08, std=0.35)
		print(f'({i+1}/{n_samples})')
		# generate path
		x, jump_arrivals = jump_diffusion_process(T=T, n=n, lmbda=lmbda, mu=mu, sigma=sigma, x0=x0, jump=normal_jump)
		df[f'x_{i+1}'] = x
		# plt.plot(ts, x, label=f'process{i+1}')

		# calculate call path
		call_price_path = calculate_option_price_path(asset_price=x, K=K, r=r, T=T, sigma=sigma, lmbda=lmbda, jump=normal_jump, n_path_simulations=200)
		df[f'call_price_{i+1}'] = call_price_path

		df.to_csv(f'iterations/jump_diffusion_paths_it{i+1}.csv', index=False)
		print(f"Saved iteration {i+1}")
		# plt.plot(ts, x, label=f'x_{i+1}')
		# plt.plot(ts, call_price_path, label=f'call_price_{i+1}')


def merge_files():
	df1 = pd.read_csv('iterations/jump_diffusion_paths_it10.csv')
	df2 = pd.read_csv('iterations/jump_diffusion_paths_it20.csv')
	df3 = pd.read_csv('iterations/jump_diffusion_paths_it30.csv')

	df = pd.DataFrame()
	df['t'] = df1['t']

	for d in [df1, df2, df3]:
		for col in d.columns[1:]:
			df[col] = d[col]

	print(df.columns)
	print(df)

	df.to_csv('jump_diffusion_paths_combined.csv', index=False)


def compare_hedging_methods():
	n_rebalancings = 100

	T = 1.0
	lmbda = 4
	mu = 0.05
	sigma = 0.5
	K = 100
	x0 = 100
	n = 1000
	r = 0.02
	normal_jump = jumps.NormalJump(mean=0.05, std=0.22)

	df = pd.read_csv('jump_diffusion_paths_combined.csv')
	ts = df['t']

	results = pd.DataFrame()
	results['t'] = df['t']

	for i in tqdm(range(30)): # because we have 30 simulations
		portfolio_value, stock_shares, option_shares, riskless_shares = simulate_hedging(ts, df[f'x_{i+1}'], df[f'call_price_{i+1}'], K=K, r=r, sigma=sigma, T=T, pi_f = -1, initial_portfolio_value = 50, n_rebalancings = n_rebalancings)
		portfolio_value1, stock_shares1, option_shares1, riskless_shares1 = simulate_quadratic_hedging(ts, df[f'x_{i+1}'], df[f'call_price_{i+1}'], K=K, r=r, sigma=sigma, T=T, pi_f = -1, initial_portfolio_value = 50, n_rebalancings=n_rebalancings)

		results[f'delta_{i+1}'] = portfolio_value
		results[f'quadratic_{i+1}'] = portfolio_value1

	results.to_csv('hedging_results.csv', index=False)


def test_hypothesis():
	initial_portfolio_value = 50
	r = 0.02
	maturity_riskless = initial_portfolio_value * np.e**r

	df = pd.read_csv('hedging_results.csv')
	last_row = df.iloc[len(df) - 1].values[1:]

	delta_hedge = last_row[0:len(last_row):2]
	quadratic_hedge = last_row[1:len(last_row):2]

	df_ttest = pd.DataFrame(columns=['delta', 'quadratic', 'difference'])

	df_ttest['delta'] = delta_hedge
	df_ttest['quadratic'] = quadratic_hedge

	df_ttest['delta'] = np.abs(df_ttest['delta'] - maturity_riskless)
	df_ttest['quadratic'] = np.abs(df_ttest['quadratic'] - maturity_riskless)

	df_ttest['diff'] = df_ttest['delta'] - df_ttest['quadratic']
	print(ttest_1samp(df_ttest['diff'], 0, alternative='greater'))

	result = ttest_rel(df_ttest['delta'], df_ttest['quadratic'], alternative='greater')
	print(result)


def create_latex_table():
	initial_portfolio_value = 50
	r = 0.02
	maturity_riskless = initial_portfolio_value * np.e**r

	df = pd.read_csv('hedging_results.csv')
	last_row = df.iloc[len(df) - 1].values[1:]

	delta_hedge = last_row[0:len(last_row):2]
	quadratic_hedge = last_row[1:len(last_row):2]

	s = "\\begin{center}\n\\begin{table}\n\\centering\n\\begin{tabular}{c c c c c c}\n\\hline\nn & $\\Delta$-hedge($u$) & Quadratic hedge ($v$) & $d_1=|u - Ve^{rT}|$ & $d_2=|v - Ve^{rT}|$ & d = d1 - d2\\\\\n\\hline \\hline\n"

	for i in range(len(delta_hedge)):
	    s+=f"{i+1} & {round(delta_hedge[i], 4)} & {round(quadratic_hedge[i], 4)} & {round(abs(delta_hedge[i] - maturity_riskless), 4)} & {round(abs(quadratic_hedge[i] - maturity_riskless), 4)} & {round(abs(delta_hedge[i] - maturity_riskless) - abs(quadratic_hedge[i] - maturity_riskless), 4)} \\\\ \n\\hline \n"
	    
	s+="\\end{tabular}\n\\caption{Portfolio value at time of maturity $T$ for $2$ hedging methods where number of rebalancings was $n=100$}\n\\end{table}\n\\end{center}\n"
	print(s)

	print(ttest_1samp(abs(np.array(delta_hedge) - maturity_riskless) - abs(np.array(quadratic_hedge) - maturity_riskless), 0, alternative='greater'))

if __name__ == "__main__":
	
	# gbm_hedging()
	# show_gbm_hedging_result()
	# analyze_hedging_results()
	# compare_quadratic_and_delta()
	# generate_jump_diffusion_paths()
	# merge_files()
	# compare_hedging_methods()
	test_hypothesis()
	# create_latex_table()
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
