import numpy as np
from scipy.linalg import hankel, lstsq, pinv
from scipy.optimize import minimize

def backward_linear_prediction(data, n_dead, order, train_factor=4, train_len=None):
    """
    Reconstruct missing initial points using Backward Linear Prediction (BLP).
    
    Args:
        data (np.ndarray): The acquired FID signal (1D complex array).
        n_dead (int): Number of missing points to reconstruct at the beginning.
        order (int): The order of the linear prediction (number of coefficients).
        train_factor (int): Multiplier for training length (train_len = train_factor * order). Used if train_len is None.
        train_len (int): Explicit number of points to use for training. Overrides train_factor.
        
    Returns:
        np.ndarray: The reconstructed FID with prepended points.
    """
    if n_dead <= 0:
        return data
        
    N = len(data)
    if N < 2 * order:
        raise ValueError("Data length must be at least 2 * order for LP.")
    
    # 1. Calculate LP coefficients using the "good" part of the signal
    # We use the first 'order' points to predict the next point, sliding forward.
    # To predict backward, we can use the forward prediction coefficients on reversed data
    # or set up the equations specifically for backward prediction.
    # Standard approach: Fit forward LP coefficients on the valid data, 
    # then use them to predict backwards (assuming stationarity/reversibility of decay).
    # For NMR signals (damped sinusoids), forward and backward coefficients are related (conjugate).
    
    # Let's use a standard SVD/Least Squares approach on a Hankel matrix.
    # We want to find coefficients a_k such that:
    # x[n] = - sum_{k=1}^{order} a_k * x[n+k]  (Backward prediction form)
    
    # Construct Hankel matrix from the valid data
    # We use a training region. Let's use the first 50% of valid data or up to 2*order points.
    if train_len is None:
        train_len = min(N, int(train_factor * order))
    else:
        train_len = min(N, int(train_len))
        
    train_data = data[:train_len]
    
    # Create Hankel matrix for backward prediction
    # We want to predict x[n] from x[n+1]...x[n+order]
    # Matrix equation: X * a = y
    # y = [x[0], x[1], ..., x[M]]
    # X = [[x[1], ..., x[order]], [x[2], ..., x[order+1]], ...]
    
    # Actually, for backward prediction:
    # x[n] = sum(c_i * x[n+i])
    
    # Let's set up the matrix to solve for coefficients c
    # Target vector y: data[0] ... data[L-1]
    # Design matrix H: columns are shifted versions of data
    
    # Using scipy.linalg.lstsq
    # We want to predict x[n] using x[n+1]...x[n+order]
    # y = data[0 : L]
    # A = column 0: data[1 : L+1]
    #     column 1: data[2 : L+2]
    #     ...
    
    L = train_len - order
    y = train_data[:L]
    A = np.zeros((L, order), dtype=complex)
    for i in range(order):
        A[:, i] = train_data[i+1 : i+1+L]
        
    # Solve A * c = y
    coeffs, residuals, rank, s = lstsq(A, y)
    
    # 2. Predict backward
    recon_points = np.zeros(n_dead, dtype=complex)
    current_data = data.copy()
    
    # We need to predict x[-1], then x[-2]...
    # x[-1] = sum(c_i * x[0+i]) = c_0*x[0] + c_1*x[1] + ...
    
    # To predict n_dead points:
    # We create a combined array and fill it backwards
    full_data = np.concatenate([np.zeros(n_dead, dtype=complex), data])
    
    # Fill backwards from n_dead-1 down to 0
    # The valid data starts at index n_dead
    for i in range(n_dead - 1, -1, -1):
        # Prediction basis starts at i+1
        basis = full_data[i+1 : i+1+order]
        # Prediction
        val = np.dot(basis, coeffs)
        full_data[i] = val
        
    return full_data

def apply_phase_correction(spectrum, phi0, phi1, pivot_index=None):
    """
    Apply zero-order and first-order phase correction with pivot support.
    
    Args:
        spectrum (np.ndarray): Complex spectrum.
        phi0 (float): Zero-order phase in degrees.
        phi1 (float): First-order phase in degrees (total phase shift across bandwidth).
        pivot_index (int): Index of the pivot point. If None, defaults to center.
        
    Returns:
        np.ndarray: Phase-corrected complex spectrum.
    """
    N = len(spectrum)
    
    if pivot_index is None:
        pivot_index = N // 2
    
    # Convert to radians
    phi0_rad = np.deg2rad(phi0)
    phi1_rad = np.deg2rad(phi1)
    
    # Create normalized frequency axis centered at pivot
    # Range: -pivot/N to (N-pivot)/N
    # This ensures that at index == pivot_index, the first order term is 0.
    freq_norm = (np.arange(N) - pivot_index) / N
    
    # Total phase correction
    # phi(f) = phi0 + phi1 * (f - f_pivot)
    phase_corr = phi0_rad + phi1_rad * freq_norm
    
    return spectrum * np.exp(1j * phase_corr)

def entropy_minimization(spectrum):
    """
    Calculate the entropy of the real part of the spectrum.
    Used as a cost function for auto-phasing.
    """
    # Focus on the real part (Absorption mode)
    real_spec = np.real(spectrum)
    
    # We want to minimize negative values and maximize peak sharpness
    # Simple entropy: -sum(p * ln(p)) where p = |S| / sum(|S|)
    # But for phasing, we often want to minimize the absolute value of the imaginary part
    # or maximize the positivity of the real part.
    
    # "Minimum Entropy" usually refers to the derivative or the magnitude distribution.
    # A common metric for NMR is minimizing the negative area (if peaks are positive).
    
    # Metric 1: Negative penalty
    neg_penalty = np.sum(real_spec[real_spec < 0]**2)
    
    # Metric 2: Area of imaginary part (should be zero for perfect phasing)
    # imag_area = np.sum(np.abs(np.imag(spectrum)))
    
    return neg_penalty

def auto_phase(spectrum):
    """
    Automatically determine phi0 and phi1.
    """
    def cost_func(x):
        p0, p1 = x
        corrected = apply_phase_correction(spectrum, p0, p1)
        # Minimize negative values in real part (assuming positive peaks)
        # This is a simple heuristic.
        real_part = np.real(corrected)
        
        # Penalty for negative values
        neg_sum = np.sum(real_part[real_part < 0]**2)
        
        # Penalty for broad bases (maximize peak height vs area?)
        # Let's stick to negative penalty for now.
        return neg_sum

    # Initial guess
    x0 = [0, 0]
    
    # Optimize
    res = minimize(cost_func, x0, method='Nelder-Mead', tol=1e-4)
    
    return res.x[0], res.x[1]
