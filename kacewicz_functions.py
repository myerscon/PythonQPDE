import numpy as np
from scipy.special import roots_legendre

#
# Quantum Parameter Functions
#

""" Calculates the parameter for the number of secondary intervals "k" from the number of primary intervals and a
	user supplied error

	This value is calculated from the relationship of the error parameter
		epsilon = 1 / (n ** (k - 1))

	Arguments:
		epsilon:	User supplied error
		n:			Integer parameter for the number of primary intervals

	Return:
		Returns the integer parameter for the number of secondary intervals
	"""
def calculate_k(epsilon, n):
	return int(np.ceil(1 - np.log(epsilon) / np.log(n)))

"""	Iteratively looks for the vlaues of the partition parameters "n" and "k"
	"n" determines the number of primary intervals
	There are n ** (k - 1) secondary intervals for each priamry interval

	Arguments:
		epsilon:	Maximum error between the approximation and the solution to the autonomous system of ODEs
		n0:			Initial guess for the number of primary intervals
		M:			Total number of secondary intervals in the time domain

	Return:
		Returns a tuple 
	"""
def calculate_partition_parameters(epsilon, M, n0 = 2):
	n = n0
	k = calculate_k(epsilon, n)

	while (n ** k < M):
		n += 1
		k = calculate_k(epsilon, n)

	return (n, k)

#
# Quantum Integration Functions
#

""" Left Riemann Integral

		Arguments:
			integrand_function: callable
				-Callable time dependent integrand function valid for batched data
			tlims: (float, float)
				-Tuple of the upper and lower bounds of integration
		"""
def integrate_riemann(integrand_function, tlims, n_samples, \
C = None, t0 = None, quantum = False, delta = None, M = None):
	t = np.linspace(tlims[0], tlims[1], n_samples, endpoint = False)

	f = integrand_function(t, t0, C)

	# Classical Left Riemann Integral
	if (not quantum): return (tlims[1] - tlims[0]) * np.mean(f, axis = 0)

	f_max = np.max(f, axis = 0)
	f_min = np.min(f, axis = 0)
	f_diff = f_max - f_min

	tolerance = 10**-12

	if (f_diff>tolerance):
	
		f_scaled = (f - f_min) / f_diff
		f_avg = np.mean(f_scaled, axis = 0)
		omega = np.arcsin(np.sqrt(f_avg)) / np.pi
		
		QAmpEst_outputs = QAmpEst(M, omega, delta)
		estimates = QAmpEst_outputs

	else:
		estimates = 0.0

	return (tlims[1] - tlims[0]) * (estimates * f_diff + f_min)

""" Gaussian Quadrature Integration

		Arguments:
			integrand_function: callable
				-Callable time dependent integrand function valid for batched data
			tlims: (float, float)
				-Tuple of the upper and lower bounds of integration
		"""
def integrate_quad(integrand_function, tlims, n_samples, \
C = None, t0 = None, quantum = False, delta = None, M = None):
	
	roots, weights = roots_legendre(n_samples)
	a = 2 / (tlims[1] - tlims[0])
	b = 1 - tlims[1] * a

	y = integrand_function((roots - b) / a, t0, C)

	f = y * weights

	if (not quantum): return np.sum(f, axis = 0) / a

	f_max = np.max(f, axis = 0)
	f_min = np.min(f, axis = 0)
	f_diff = f_max - f_min

	tolerance = 10**-12

	if (f_diff>tolerance):

		f_scaled = (f - f_min) / f_diff
		f_avg = np.mean(f_scaled, axis = 0)
		omega = np.arcsin(np.sqrt(f_avg)) / np.pi
		
		QAmpEst_outputs = QAmpEst(M, omega, delta)
		estimates = QAmpEst_outputs

	else:
		estimates = 0.0

	return n_samples * (estimates * f_diff + f_min) / a

# Wrapper function for selecting either integrate_riemann or integrate_quad
def integrate(integrand_function, tlims, n_samples, \
integral_mode = "riemann", C = None, t0 = None, \
quantum = False, delta = None, M = None):
	if (integral_mode == "riemann"):
		return integrate_riemann(integrand_function, tlims, n_samples, C = C, t0 = t0, \
								 quantum = quantum, delta = delta, M = M)
	
	elif (integral_mode == "quad"):
		return integrate_quad(integrand_function, tlims, n_samples, C = C, t0 = t0, \
								 quantum = quantum, delta = delta, M = M)

	else: raise ValueError("kaceiwcz integrate() - Invalid integral_mode")

"""	Estimates unknown quantum amplitude using Quantum Amplitude Estiamtion Algorithm
	-Code written by F. Gaitan
	-See (Brassard et al., quant-ph/0005055)

	Arguments:
		M:		Is the dimension of the Hilbert space
		omega:	Is the unkown scalar used for deriving the quantum amplitude
		delta:	Is one minus the probability that our amplitude estimation is within the error bounds of the true
				amplitude

	Return:
		Returns tuple with the following components:
			(a_estimate, a, error, message, SucccessFlag, upper_bound)
				a_estimate:		Estiamte of the quantum amplitude
				a:				Actual value of the quantum amplitude
				error:			Error between the estimate and the actual amplitude
				message:		Printable message to console as a string
				SuccessFlag:	Integer indicating whether the error violated the upper bound
									SuccessFlag = 0 if upper bound was violated
									SuccessFlag = 1 if upper bound  was NOT violated
				upper_bound:	Upper bound of the error between the estiamte and true value of the
									amplitude
	"""
def QAmpEst(M, omega, delta):
	# Calculate the total number of amplitude estimates needed
	total_runs = int(np.ceil(-8 * np.log(delta)))
	if (total_runs % 2 == 0): total_runs += 1

	estimates = np.array([randQAEA(M, omega) for i in range(total_runs)])
	median_estimate = np.median(estimates)
	estimate_amplitude = np.sin(np.pi * median_estimate / M) ** 2

	return estimate_amplitude

"""	Generates random deviate for Quantum Amplitude Estimation
	-Samples random deviate from the normalized sinc^2(x) function over discrete values of x
	-See (Brassard et al. quant-ph 0005055)
	-Code written by F. Gaitan
	
	Arguments:
		M:		Is the dimension of the Hilbert Space
		omega:	Is the unknown scalar used for deriving the quantum amplitude
				-It is assumed that "omega" is a scalar quantity between 0 and 1 inclusive
	
	Return:
		Returns a random deviate
	"""
def randQAEA(M, omega):
	Momega = M * omega

	""" SubIntCounter identifies which subinterval is currently being processed in while loop below
		Subintervals are numbered 1 -> M, and subinterval j corresponds to QAEA probability p(j - 1)
    	"""
	SubIntCounter = 0

	# Stores QAEA discrete probability distribution. p(j - 1), with 1 <= j <= M
	Probs = []

	# Stores partial sums of p(j - 1) from 1 to M terms
	p_sums = []

	# Integer values of discrete domain
	y = np.array(range(M))
	x = y - Momega # Shifted domain appearing in p(y)

	""" Calculate QAEA probabilities p(j - 1), store in Probs(j) with 1 <= j <= M
		-Note that probs[1] = p(0), probs(j) = p(j - 1), and Probs(M) = p(M - 1)
		"""
	tiny = 1e-24 # Prevents divide by zero errors when x is zero
	tempProb = np.sin(np.pi * (x+tiny)) / np.sin(np.pi * (x+tiny) / M)
	Probs = (1 / M ** 2) * (tempProb ** 2)

	""" Accumulate partial sums of p(j - 1), 1 <= j <= M, store in p_sums, where
		p_sums(j) = p(0) + ... + p(j - 1)
		"""
	p_sums.append(Probs[0])
	for i in range(1, M): p_sums.append(p_sums[i - 1] + Probs[i])

	# Sample uniform deviate
	u = np.random.random()

	""" Determine which subinterval contains u
		1. Loop through subintervals, with SubIntCounter tracking which subinterval is currently being processed
		2. Exit loop when u > A(SubIntCounter) first occurs. Subinterval labeled by SubIntCounter at exit contains u
		"""
	while (p_sums[SubIntCounter] < u): SubIntCounter += 1

	return SubIntCounter 
