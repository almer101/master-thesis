import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal, standard_normal, exponential

def bm_path(n=252, sigma=0.5):
	# scale is std dev
	path = np.zeros(n)
	for i in range(1, n):
		path[i] = path[i-1] + sigma * standard_normal()
	return path

def wiener_process(T=1, n=252): 
	dt = T/n
	return bm_path(n=n, sigma=np.sqrt(dt))

def geometric_bm_path(T=1.0, n=252, mu=0.05, sigma=0.5, x0=100):
	if x0 <= 0: raise ValueError('Starting price of an asset must be positive')
	dt = T/n
	path = np.zeros(n)
	path[0] = x0
	for i in range(1,n):
		path[i] = path[i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * normal(scale=np.sqrt(dt)))
	return path

def poisson_arrival_times(lmbda):
	arrivals = []
	cumulative_time = 0.0
	while cumulative_time < 1.0:
		e = exponential(scale = 1 / lmbda)
		cumulative_time += e
		if cumulative_time > 1.0:
			break
		arrivals.append(cumulative_time)
	return np.array(arrivals)

def static_jump_1():
	return 1

def normal_jump():
	return normal(loc=0, scale=0.2)

def jump_process_path(lmbda = 4, jump_size=None): # jump size should be callable
	arrivals = poisson_arrival_times(lmbda=lmbda)
	n = len(arrivals)
	xs = [0.0]
	for i in range(n):
		xs.append(arrivals[i])
		xs.append(arrivals[i])
	xs.append(1.0) # appending T=1.0
	ys = [0,0]
	process_value = 0.0
	for i in range(n):
		process_value += jump_size()
		ys.append(process_value)
		ys.append(process_value)
	return xs, ys

def poisson_process_path(lmbda = 4):
	return jump_process_path(lmbda=lmbda, jump_size=static_jump_1)

def jump_diffusion_process(T=1.0, n=1000, lmbda=4, mu=0.05, sigma=0.5, x0=100, jump_size=None):
	if x0 <= 0: raise ValueError('Starting price of an asset must be positive')
	if jump_size is None: raise ValueError('Jump size function has to be provided')

	arrivals = poisson_arrival_times(lmbda=lmbda)
	next_arrival_index = 0

	dt = T/n
	path = np.zeros(n)
	path[0] = x0
	for i in range(1,n):
		if next_arrival_index < len(arrivals) and arrivals[next_arrival_index] <= dt * i:
			path[i] = path[i-1] * (1 + jump_size())
			next_arrival_index += 1
		else:
			path[i] = path[i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * normal(scale=np.sqrt(dt)))
	return path, arrivals

if __name__ == "__main__":
	# std_devs = [0.1, 0.5, 1.1]
	# for std_dev in std_devs:
	# 	plt.plot(bm_path(scale=std_dev), label=f'sigma={std_dev}')
	# plt.legend()
	# plt.show()

	# mus = [0.01, 0.02, 0.12]
	# for mu in mus:
	# 	plt.plot(geometric_bm_path(n=252, mu=mu, sigma=0.2), label=f'mu={mu}')
	# plt.legend()
	# plt.show()
	
	n = 1000
	x,y = poisson_process_path(lmbda=4)
	x,y = jump_process_path(lmbda=6, jump_size=normal_jump)
	# plt.plot(x,y)
	
	path, arrivals = jump_diffusion_process(T=1.0, n=n, lmbda=4, mu=0.05, sigma=0.4, x0=100, jump_size=normal_jump)
	for a in arrivals:
		plt.axvline(x=a, c='green', linestyle='--', alpha=0.5, linewidth=0.9)
	plt.plot(np.linspace(0,1,n), path, c='black', alpha=0.8)
	plt.show()
