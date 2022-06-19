# Option Pricing and Hedging under Jump-diffusion model

---
**NOTE**

This README provides a rough overview of what was achieved in the scope of this thesis.   
The full text in English can be found [here](thesis.pdf)

---

Stochastic processes have a very wide application in the field of finance. The main reason for that is that they can be used to model an asset price as a process where uncertainty is present. Behavior of such processes can be observed to potentially draw some conclusions from them. 

The goal of this thesis is to utilize stochastic processes and apply them to tackle some of the problems from mathematical finance. Firstly, I will start with defining necessary mathematical and financial concepts which will be used throughout this thesis. After the foundation is set, starting with the Brownian motion, step-by-step we will arrive at the jump-diffusion process which will be used as a process of the price of an asset. We will then introduce the idea of a financial derivative, in our case an option, and we will tackle the problem of determining a fair price of the option. Lastly, we will introduce the idea of risk, how to quantify it and how to hedge the risk of our position in the asset or the option. It is my hope that, this thesis will not only provide you with a good overview of pricing and hedging of financial instruments, but also give you the understanding of such the underlying processes are useful to model the scenarios that occur every day in the market.

## Pricing an Asset
A process $X_t$ satisfying the dynamics
$$ dX_t = \mu_t X_t dt + \sigma_t X_t dW_t $$
is called a **Geometric Brownian motion**. Here we make one step further and incorporate jumps in this process. Jumps arrive as a **Poisson process**, which means that times between jumps are independent and identically distributed exponential random variables. When the jump arrives, the size of the jump is drawn from the normal distribution $\mathcal{N}(\mu_j, \sigma^2_j)$. The resulting process is called a **Jump-diffusion process**. A sample trajectory of such a process can be seen in the following figure:

<p align="center">
    <img width="550" alt="Screenshot 2022-06-19 at 11 23 06" src="https://user-images.githubusercontent.com/40769239/174474299-c2e7e3d4-e654-47a2-8ae4-6760b6c2e4e2.png">
</p>

## Pricing an Option
After the price of an asset has been defined we try to determine a price of a **European call option**. Since now we are not anymore in the _simple_ Black-Scholes world, by utilizing only the delta hedge we are unable to hedge the jump risk. We go around that by emplying the **Capital Asset Pricing Model (CAPM)** and finally we are able to come to the conclusion that the premium requested by the writer of the option is the discounted expected payoff of the option. For the case of the European call option we derive the following formula for the option price:
$$ f(t) = \sum_{n=0}^{\infty}\mathbb{E}\left[c_{BS}\left(X(t)e^{-k\lambda(T-t)}\left(\prod_{i=1}^{n}1+U_i\right), t\right)\right]e^{-\lambda(T-t)}\frac{[\lambda(T-t)]^n}{n!} $$
With simulations we show the behaviour of the option price as the option moves closer to maturity of one year ($T=1$):

<p align="center">
  <img width="550" alt="Screenshot 2022-06-19 at 11 43 32" src="https://user-images.githubusercontent.com/40769239/174475128-80314256-2877-46a1-9262-d47b854f2094.png">
</p>

Another interesting thing to highlight is the shape of the European call option price surface dependent on the `spot price` and `time left to maturity`. We expect a somewhat downward sloping surface as we approach time of maturity:

<p align="center">
  <img width="550" alt="Screenshot 2022-06-19 at 11 47 28" src="https://user-images.githubusercontent.com/40769239/174475257-8b29afa1-fbc1-4558-a1b2-cb3b8dd270d8.png">
</p>

## Hedging
In this last chapter of the thesis we first examine the effect of the famous delta hedge when the underlying asset is governed by a regular geometric Brownian motion. Just to recap, the delta hedge offsets the risk of the diffusion part of the stochastic process. In case we have shorted one call option, the amount of the underlying we would have to hold at any time $t$ in order to perfectly hedge our short position is: $$\frac{\partial C}{\partial X} = N(d_1)$$ where $N$ is the unit normal CDF and $d_1$ is defined as: $$d_1 = \frac{\ln{\frac{X_t}{K}} + \left(r + \frac{\sigma^2}{2} \right)\tau}{\sigma\sqrt{\tau}}$$

Later we make our world more complex by, again, incorporating jumps in the underlying asset price process. In this case we wanted to compare the results of the hedging with the regular **delta hedge** and with a **quadratic hedging** method. In R. _Cont and P. Tankov. Financial modelling with jump processes_ they have arrived at the following formula: $$ \phi^* = \frac{\sigma^2 \frac{\partial C}{\partial X} + \frac{1}{X} \int z[C(t,X(1+z))-C(t,X)] \nu_U(z)dz}{\sigma^2 + \int z^2 \nu_U(z)dz} $$

The effects of both hedging methods can be observed on the following Figure:

<p align="center">
    <img width="550" alt="Screenshot 2022-06-19 at 11 59 14" src="https://user-images.githubusercontent.com/40769239/174475660-ce4e5b68-fd5d-4ba6-b741-b98df78049b1.png">
</p>
