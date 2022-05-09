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
import hedging
import jumps


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


def plot_call_price_surface():
	file = 'call_price_surface.csv'
	df = pd.read_csv(file)

	ax = plt.axes(projection ='3d')
 
	# defining all 3 axes
	z = df['call_price']
	x = np.sort(np.unique(df['time_to_maturity']))
	y = np.sort(np.unique(df['spot_price']))
	  
	X, Y = np.meshgrid(x, y)
	Z = np.zeros(X.shape)
	print(X)
	print(Y)

	for i in range(len(X)):
		for j in range(len(X[i])):
			flag1 = df['time_to_maturity'] == X[i][j]
			flag2 = df['spot_price'] == Y[i][j]
			Z[i][j] = df[flag1 & flag2]['call_price'].values[0]
	 
	print(Z)
	ax = plt.axes(projection ='3d')
	ax.plot_wireframe(X, Y, Z, color ='green')
	ax.set_title('wireframe geeks for geeks')
	plt.show()

def generate_latex_coordinates():
	file = 'call_price_surface.csv'
	df = pd.read_csv(file)

	code = """\\begin{tikzpicture}\n\\begin{axis}[title=Call price surface, colormap/cool]\n\\addplot3[mesh,]\n coordinates {\n"""
	x = np.sort(np.unique(df['time_to_maturity']))
	y = np.sort(np.unique(df['spot_price']))
	  
	X, Y = np.meshgrid(x, y)
	Z = np.zeros(X.shape)

	for i in range(len(X)):
		for j in range(len(X[i])):
			flag1 = df['time_to_maturity'] == X[i][j]
			flag2 = df['spot_price'] == Y[i][j]
			price =  df[flag1 & flag2]['call_price'].values[0]
			code += f"({X[i][j]}, {Y[i][j]}, {price}) "
		code += '\n\n'
	
	code += "};\n\\end{axis}\n\\end{tikzpicture}"

	print(code)

if __name__ == "__main__":
	generate_latex_coordinates()
	exit()
	n = 300
	T = 1.0
	K = 100
	r = 0.02
	mu = 0.05
	lmbda = 4
	sigma = 0.5


	normal_jump = jumps.NormalJump(mean=0.05, std=0.3)
	lmbda = 4
	ts = np.linspace(0, T, n)
	asset_price, arrivals = jump_diffusion_process(T=T, n=n, lmbda=lmbda, mu=mu, sigma=sigma, x0=100, jump=normal_jump)
	print(arrivals)
	# call_price = option_pricing.calculate_option_price_path(asset_price, K=100, r=r, T=T, sigma=sigma, lmbda=lmbda, jump=normal_jump, n_path_simulations=200)
	
	plt.plot(ts, asset_price)
	# plt.plot(ts, call_price)
	plt.grid()
	plt.show()

	exit()
	# plt.plot(ts, asset_price)
	# plt.plot(ts, call_price)
	# plt.show()

	# df = pd.DataFrame(columns=['t', 'x', 'f(x)'])
	# df['t'] = ts
	# df['x'] = asset_price
	# df['f(x)'] = call_price
	# df.to_csv('jump_diffusion_call_lmbda8_n1000.csv')

	# df = pd.read_csv('jump_diffusion_call_n1000_K100.csv')
	df = pd.read_csv('jump_diffusion_call_K100.csv')

	lmbda = 8
	df = pd.read_csv('jump_diffusion_call_lmbda8_n1000.csv')
	# plt.plot(df['t'], df['x'], label='asset price')
	# plt.plot(df['t'], df['f(x)'], label = 'call option price')
	# plt.grid()
	# plt.legend()
	# plt.show()
	
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

	# plt.plot(ts, df_dx)
	# plt.show()
	# exit()

	# TODO: JUMP size class !!!! so you can have on one place the mean, sigma etc.
	n_rebalancings = 10
	portfolio_value, stock_shares, option_shares, riskless_shares = hedging.simulate_hedging(ts, asset_price, riskless_asset, call_price, df_dx, K, r, sigma, T, pi_f=-1, initial_portfolio_value=50, n_rebalancings=n_rebalancings) # short one option
	portfolio_value_quad, stock_shares_quad, option_shares_quad, riskless_shares_quad = hedging.simulate_quadratic_hedging(ts, asset_price, riskless_asset, call_price, df_dx, K, r, sigma, T, lmbda, pi_f=-1, initial_portfolio_value=50, n_rebalancings=n_rebalancings) # short one option

	plt.plot(ts, asset_price, label='asset price')
	plt.plot(ts, portfolio_value, label='delta hedged portfolio')
	plt.plot(ts, portfolio_value_quad, label='quadratic hedged portfolio')

	plt.legend()
	plt.grid()
	plt.show()
