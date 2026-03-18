from turtle import speed


def double_rayleigh(path,p0=[7,65,0.5],sfs=20,bins=40,minlike=0.8,percentile=99,method="MCMC",
                    N_MH=50000,MHstepsize=[0.5,3,0.02],burn_in_fraction=1/5,plotting=True):
    """
    Fit a mixture of two Rayleigh distributions to speed data.

    Parameters
    ----------
    path : str
        Path to the CSV file containing tracking data.

    p0 : list or tuple
        Initial guess for parameters [sigma1, sigma2, w].

    sfs : int, default=20
        Step frame size used when calculating displacement speeds.

    bins : int, default=40
        Number of bins used in the histogram.

    minlike : float, default=0.8
        Minimum likelihood threshold for accepting tracking points.

    percentile : float, default=99
        Upper percentile cutoff used when plotting the histogram.

    method : str, default="MCMC"
        Fitting method. Options:
        - "MCMC"
        - "MLE"

    N_MH : int, default=50000
        Number of Metropolis-Hastings iterations.

    stepsize : list
        Proposal step sizes for [sigma1, sigma2, w].

    burn_in_fraction : float
        Fraction of samples discarded as burn-in.

    plotting : bool
        If True, plot histogram and fitted PDF.

    Returns
    -------
    params : list
        [sigma1, sigma2, weight, D1, D2]

    errors : list
        Parameter uncertainties.
    """
    #imports
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.optimize import minimize
    data=np.genfromtxt(path,delimiter=",",skip_header=1)

    #extracting likelihoods and coordinates
    sllike=data[:,12]
    srlike=data[:,15]
    tlike=data[:,18]

    #filtering out rows where any likelihood is below the threshold
    mask = (tlike>minlike) & (srlike>minlike) & (sllike>minlike)

    slx=data[:,10][mask]
    sly=data[:,11][mask]
    srx=data[:,13][mask]
    sry=data[:,14][mask]
    tx=data[:,16][mask]
    ty=data[:,17][mask]

    #calculating midpoints

    Smidx=slx+(srx-slx)/2
    Smidy=sly+(sry-sly)/2
    midx=tx+(Smidx-tx)/2
    midy=ty+(Smidy-ty)/2

    #½truncating to make length divisible by sfs

    n = len(midx) % sfs
    if n != 0:
        x_trunc = midx[:-n]
        y_trunc = midy[:-n]
    else:
        x_trunc = midx
        y_trunc = midy

    #reshaping into blocks of size sfs and summing to get speed

    x_blocks = x_trunc.reshape(-1, sfs)
    y_blocks = y_trunc.reshape(-1, sfs)

    speed = np.abs(np.sqrt((x_blocks[:,-1] - x_blocks[:,0])**2 + (y_blocks[:,-1] - y_blocks[:,0])**2)) 
    
    #korrekte enheder
    fps=20
    pixel_to_mm=1 #midlertidig value
    speed = speed * pixel_to_mm * (fps/sfs)

    #fitting a weighted sum of two Rayleigh distributions to the speed data
    def weighted_rayleigh_pdf(x, sigma1, sigma2, w):
        r1 = (x / sigma1**2) * np.exp(-x**2/(2*sigma1**2))
        r2 = (x / sigma2**2) * np.exp(-x**2/(2*sigma2**2))
        return w*r1 + (1-w)*r2
    
    #negative log-likelihood function to minimize
    def neg_log_likelihood(params, data):
        sigma1, sigma2, w = params
        pdf_vals = weighted_rayleigh_pdf(data, sigma1, sigma2, w)
        return -np.sum(np.log(pdf_vals + 1e-12))
    

    def MCMC(data,p0, N_MH, stepsize):
        sigma1, sigma2, w = p0   
        q_s1, q_s2, q_sw = stepsize
        burn_in = int(N_MH * burn_in_fraction)

        theta_save = np.zeros((N_MH,3))
        for i in range (N_MH):
            sigma1_new = np.abs(sigma1 + q_s1*np.random.normal())
            sigma2_new = np.abs(sigma2 + q_s2*np.random.normal())
            w_new = np.clip(w + q_sw*np.random.normal(), 0, 1)
            
            logP_ratio = np.log(weighted_rayleigh_pdf(data, sigma1_new, sigma2_new, w_new)+ 1e-12) - \
            np.log(weighted_rayleigh_pdf(data, sigma1, sigma2, w)+ 1e-12)
            P_accept = np.exp(np.sum(logP_ratio))

            if P_accept >= 1: 
                sigma1 = sigma1_new
                sigma2 = sigma2_new
                w = w_new
            elif P_accept >= np.random.random():
                sigma1 = sigma1_new
                sigma2 = sigma2_new
                w = w_new
            theta_save[i,0] = sigma1
            theta_save[i,1] = sigma2
            theta_save[i,2] = w
        posterior_samples = theta_save[burn_in:]
        return posterior_samples

    if method=="MCMC":
        posterior_samples = MCMC(speed,p0, N_MH, MHstepsize)
        sigma1, sigma2, weight = np.mean(posterior_samples, axis=0)
        sigma1_err, sigma2_err, weight_err = np.std(posterior_samples, axis=0)

    elif method=="MLE":
        #Using scipy's minimize function to find the parameters that minimize the negative log-likelihood
        res = minimize(neg_log_likelihood, x0=p0, args=(speed,),
            bounds=[(0.01, None), (0.01, None), (1e-3, 1-1e-3)])
        sigma1, sigma2, weight = res.x


    Delta_t = 50e-3 * sfs
    D1 = sigma1**2 / (2 * Delta_t)
    D2 = sigma2**2 / (2 * Delta_t)

    if method=="MCMC":
        D1_err = (sigma1 / Delta_t) * sigma1_err
        D2_err = (sigma2 / Delta_t) * sigma2_err    

    #plotting the histogram and the fitted PDF
    if plotting==True:
        plt.figure(figsize=(8,6))

        #cutting off the histogram at the specified percentile to avoid long tails dominating the plot
        upper = np.percentile(speed, percentile)
        plt.hist(speed, bins=bins, range=(0, upper), density=True, alpha=0.5, color='C0', label="Histogram")
        X = np.linspace(0, upper, 1000)
        Y = weighted_rayleigh_pdf(X, sigma1, sigma2, weight)
        if method=="MCMC":
            plt.plot(X, Y, 'r', lw=2, label=f"Log-likelihood maximized PDF: σ1={sigma1:.2f}±{sigma1_err:.2f}, σ2={sigma2:.2f}±{sigma2_err:.2f}, w={weight:.2f}±{weight_err:.2f}")
        elif method=="MLE":
            plt.plot(X, Y, 'r', lw=2, label=f"Log-likelihood maximized PDF: σ1={sigma1:.2f}, σ2={sigma2:.2f}, w={weight:.2f}")
        plt.legend()
        plt.xlabel("Fart\n[$mm/s$]")
        plt.ylabel("PDF")
        plt.title("2 Rayleigh fit")
        plt.show()
    
    if method=="MCMC":
            return [sigma1, sigma2, weight, D1, D2], [sigma1_err, sigma2_err, weight_err,D1_err, D2_err]
    elif method=="MLE":
            return [sigma1, sigma2, weight, D1, D2], [None, None, None, None, None]


def løbendefit(path,p0=[7,65,0.5],sfs=10,minlike=0.8,windowsize=100,windowstep=10,N_MH=10000,
               MHstepsize=[0.5,3,0.02],burn_in_fraction=1/4,plotting=True):
    """
    Fit a mixture of two Rayleigh distributions to speed data.

    Parameters
    ----------
    path : str
        Path to the CSV file containing tracking data.

    p0 : list or tuple
        Initial guess for parameters [sigma1, sigma2, w].
        Efter første vindue opdateres p0 til at være gennemsnittet af de accepterede samples, så det næste vindue starter tættere på det optimale parameter sæt.

    sfs : int, default=10
        Step frame size used when calculating displacement speeds.

    minlike : float, default=0.8
        Minimum likelihood threshold for accepting tracking points.

    windowsize : int, default=100
        Number of speed data points in each window.

    windowstep : int, default=10
        Step size for moving the window across the speed data.

    N_MH : int, default=10000
        Number of Metropolis-Hastings iterations.

    MHstepsize : list
        Proposal step sizes for [sigma1, sigma2, w].

    burn_in_fraction : float
        Fraction of samples discarded as burn-in.

    plotting : bool
        If True, plot histogram and fitted PDF.
    """

    #imports
    import matplotlib.pyplot as plt
    import numpy as np
    data=np.genfromtxt(path,delimiter=",",skip_header=1)

    #extracting likelihoods and coordinates
    sllike=data[:,12]
    srlike=data[:,15]
    tlike=data[:,18]

    #filtering out rows where any likelihood is below the threshold
    mask = (tlike>minlike) & (srlike>minlike) & (sllike>minlike)

    slx=data[:,10][mask]
    sly=data[:,11][mask]
    srx=data[:,13][mask]
    sry=data[:,14][mask]
    tx=data[:,16][mask]
    ty=data[:,17][mask]

    #calculating midpoints

    Smidx=slx+(srx-slx)/2
    Smidy=sly+(sry-sly)/2
    midx=tx+(Smidx-tx)/2
    midy=ty+(Smidy-ty)/2

    #½truncating to make length divisible by sfs

    n = len(midx) % sfs
    if n != 0:
        x_trunc = midx[:-n]
        y_trunc = midy[:-n]
    else:
        x_trunc = midx
        y_trunc = midy

    #reshaping into blocks of size sfs and summing to get speed

    x_blocks = x_trunc.reshape(-1, sfs)
    y_blocks = y_trunc.reshape(-1, sfs)

    speed = np.abs(np.sqrt((x_blocks[:,-1] - x_blocks[:,0])**2 + (y_blocks[:,-1] - y_blocks[:,0])**2)) 
    if windowsize > len(speed):
        raise ValueError("Window size larger than dataset length")
    #korrekte enheder
    fps=20
    pixel_to_mm=1 #midlertidig value
    speed = speed * pixel_to_mm * (fps/sfs)

    #fitting a weighted sum of two Rayleigh distributions to the speed data
    def weighted_rayleigh_pdf(x, sigma1, sigma2, w):
        r1 = (x / sigma1**2) * np.exp(-x**2/(2*sigma1**2))
        r2 = (x / sigma2**2) * np.exp(-x**2/(2*sigma2**2))
        return w*r1 + (1-w)*r2
    
    def MCMC(data,p0, N_MH, stepsize,burn_in_fraction):
        sigma1, sigma2, w = p0   
        q_s1, q_s2, q_sw = stepsize
        burn_in = int(N_MH * burn_in_fraction)

        theta_save = np.zeros((N_MH,3))
        for i in range (N_MH):
            sigma1_new = np.abs(sigma1 + q_s1*np.random.normal())
            sigma2_new = np.abs(sigma2 + q_s2*np.random.normal())
            w_new = np.clip(w + q_sw*np.random.normal(), 0, 1)
            
            logP_ratio = np.log(weighted_rayleigh_pdf(data, sigma1_new, sigma2_new, w_new)+ 1e-12) - \
            np.log(weighted_rayleigh_pdf(data, sigma1, sigma2, w)+ 1e-12)
            P_accept = np.exp(np.sum(logP_ratio))

            if P_accept >= 1: 
                sigma1 = sigma1_new
                sigma2 = sigma2_new
                w = w_new
            elif P_accept >= np.random.random():
                sigma1 = sigma1_new
                sigma2 = sigma2_new
                w = w_new
            theta_save[i,0] = sigma1
            theta_save[i,1] = sigma2
            theta_save[i,2] = w
        posterior_samples = theta_save[burn_in:]
        return posterior_samples
    
    n_windows = ((len(speed) - windowsize) // windowstep) + 1
    params_save = np.zeros((n_windows, 3))
    err_save = np.zeros((n_windows, 3))

    for i in range(n_windows):
        start = i * windowstep
        end = start + windowsize
        window_data = speed[start:end]
        posterior_samples = MCMC(window_data, p0, N_MH, MHstepsize,burn_in_fraction)
        sigma1, sigma2, weight = np.mean(posterior_samples, axis=0)
        sigma1_err, sigma2_err, weight_err = np.std(posterior_samples, axis=0)
        params_save[i] = [sigma1, sigma2, weight]
        err_save[i] = [sigma1_err, sigma2_err, weight_err]
        p0=[sigma1, sigma2, weight]  # Update p0 for the next window
    sigma1, sigma2, weight = params_save[-1]
    fps=20
    Delta_t = sfs / fps  # seconds per displacement
    D1 = sigma1**2 / (2 * Delta_t)
    D2 = sigma2**2 / (2 * Delta_t)
    sigma1_err, sigma2_err, weight_err = err_save[-1]
    D1_err = (sigma1 / Delta_t) * sigma1_err
    D2_err = (sigma2 / Delta_t) * sigma2_err

    if plotting:
        plt.figure(figsize=(12, 8))
        plt.subplot(3,1,1)
        plt.errorbar(range(n_windows), params_save[:,0], yerr=err_save[:,0], fmt='-o')
        plt.title("sigma1 over time")
        plt.subplot(3,1,2)
        plt.errorbar(range(n_windows), params_save[:,1], yerr=err_save[:,1], fmt='-o')
        plt.title("sigma2 over time")
        plt.subplot(3,1,3)
        plt.errorbar(range(n_windows), params_save[:,2], yerr=err_save[:,2], fmt='-o')
        plt.title("Weight over time")
        plt.tight_layout()
        plt.show()

    return [sigma1, sigma2, weight, D1, D2], [sigma1_err, sigma2_err, weight_err,D1_err, D2_err]
    
