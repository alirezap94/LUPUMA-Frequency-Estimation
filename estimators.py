
from numpy import reshape, arange, angle, zeros, conj, transpose, exp, abs, pi, ceil, argmin
from numpy import eye, mod, append, sqrt, ones, fft, argmax

from numpy import sum as np_sum
from numpy import min as np_min
from scipy.linalg.lapack import zgesvd, dgesvd

from numpy import real, expand_dims, imag, arctan, sin, arcsin, tan
from numpy.linalg import pinv, inv


def unitary_mat(M = 16):
    I = eye(int(M/2))
    Pi = eye(int(M/2))[::-1]
    if mod(M,2) == 0:
        T1 = append(I, 1j*I, axis = 1)
        T2 = append(Pi, -1j*Pi, axis = 1)
        T = 1/sqrt(2)*append(T1, T2, axis = 0)
    else:
        T1 = append(I, append(zeros((int(M/2),1)), 1j*I, axis = 1), axis =1 )
        T2 = append(zeros((1,int(M/2))), append(ones((1,1))*sqrt(2), zeros((1,int(M/2))), axis = 1 ), axis = 1)
        T3 = append(Pi, append(zeros((int(M/2),1)), -1j*Pi, axis = 1), axis = 1 )
        T = 1/sqrt(2)*append(append(T1, T2, axis = 0), T3, axis = 0)  
    return T

def CR_mapping(S):
    [M,N] = S.shape
    TMH = conj(unitary_mat(M)).T
    T2N = unitary_mat(2*N)
    PiM = eye(int(M))[::-1]
    PiN = eye(int(N))[::-1]
    return TMH@append(S, PiM@conj(S)@PiN, axis = 1)@T2N


def A_WLS(k , r):
    N = len(r)
    n = arange(len(r))
    return np_sum(r*exp((-1j*2*pi)*(n/N*k)))

def A_par(theta , r):
    N = len(r)
    n = arange(len(r))
    return np_sum(r*exp((-1j*2*pi)*(n*theta)))

def PUMA(observed_signal_vector, rows_num, col_num, max_iter = 5 , min_error = 1e-3):
    '''
    This function estimates the frequency of a complex single-tone signal based on PUMA method. 
    For more information please refer to: 
    H. C. So, F. K. W. Chan, and W. Sun, "Subspace approach for fast and accurate single-tone frequency estimation," 
    IEEE Transactions on Signal Processing, vol. 59, no. 2, pp. 827-831, 2010.

    Parameters
    ----------
    observed_signal_vector : array (K, )
        Vector of time-domain observed signal with the length of K > M*N
    rows_num: int (M)
        Number of rows in the reshaped matrix (constructed from the signal vector)
    col_num: int (N)
        Number of cloumns in the reshaped matrix (constructed from the signal vector)
    max_iter: int
        Maximum number of iterations to update the fine search. 
    min_error: float
        the acceptable error for the fine search to stop.
        Note that fine search will stop if either (iter > max_iter or error < min_error)
        
    Return
    -------
    w_est: float
        The estimation of the frequency between (-pi, pi)
    '''

    r = observed_signal_vector
    M = rows_num
    N = col_num
    R = reshape(r[:M*N] , (N,M)).T
    U, s, VT, info = zgesvd(R, full_matrices = 0)

    u1 = reshape(U[:,0],(-1,1))
    v1 = (reshape(VT[0,:],(-1,1)))
    
    ########## wL: ###################
    x1 = u1[:M-1,0]
    x2 = u1[1:,0]

    x1 = x1.reshape((-1,1))
    x2 = x2.reshape((-1,1))
    
    m = arange(M-1)
    wL = angle(np_sum( (N*(m+1)-(m+1)**2)/(N)*conj(u1[:M-1,0])*(u1[1:M,0]) ))
    W = zeros((M-1,M-1))+1j*zeros((M-1,M-1))
    
    iters = 0
    error = 1
    x1H = conj(transpose(x1))
    
    while max_iter>iters and error > min_error:
        w = wL
        for m in range(1,M):
            for n in range(1,M):
                W[m-1,n-1] = np_min([m,n])*exp(1j*(m-n)*w)
        wL = angle(x1H@W@x2)[0][0]
        error = abs(w-wL)
        iters += 1
   
    ########## sai: ###################
    y1 = (v1[:N-1,0])
    y2 = (v1[1:N,0])

    y1 = y1.reshape((-1,1))
    y2 = y2.reshape((-1,1))
    
    temp = 0
    for n in range(N-1):
        temp += (N*(n+1)-(n+1)**2)/(N)*conj(v1[n,0])*v1[n+1,0]
    sai = angle(temp)

    W = zeros((N-1,N-1))+1j*zeros((N-1,N-1))
    iters = 0
    error = 1
    y1H = conj(transpose(y1))
    while max_iter>iters and error > min_error:
        w = sai
        for m in range(1,N):
            for n in range(1,N):
                W[m-1,n-1] = np_min([m,n])*exp(1j*(m-n)*w)
        sai = angle(y1H@W@y2)
        error = abs(w-sai)
        iters += 1
    
    
    wRk = ( sai + 2*pi*arange(-ceil(M/2), ceil(M/2),1) )/M
    wR = wRk[0, argmin(abs( wRk - wL ) )]
    w_est = ( (M**2-1)*wL + M**2*(N**2-1)*wR ) / (M**2*N**2-1)
    return w_est



def Unitary_PUMA(observed_signal_vector, rows_num, col_num, max_iter = 5, min_error = 1e-5):
    '''
    This function estimates the frequency of a complex single-tone signal based on Unitary-PUMA method. 
    For more information please refer to: 
    C. Qian, L. Huang, H. C. So, N. D. Sidiropoulos, and J. Xie, "Unitary PUMA algorithm for estimating the frequency of a complex sinusoid,"
    IEEE Transactions on Signal Processing, vol. 63, no. 20, pp. 5358-5368, 2015.

    Parameters
    ----------
    observed_signal_vector : array (K, )
        Vector of time-domain observed signal with the length of K > M*N
    rows_num: int (M)
        Number of rows in the reshaped matrix (constructed from the signal vector)
    col_num: int (N)
        Number of cloumns in the reshaped matrix (constructed from the signal vector)
    max_iter: int
        Maximum number of iterations to update the fine search. 
    min_error: float
        the acceptable error for the fine search to stop.
        Note that fine search will stop if either (iter > max_iter or error < min_error)
        
    Return
    -------
    w_est: float
        The estimation of the frequency between (-pi, pi)
    '''
    r = observed_signal_vector
    M = rows_num
    N = col_num
    R_complex = reshape(r[:M*N] , (N,M)).T
    #################### wL ###########
    R = CR_mapping(R_complex)
    E, temp, temp, temp = dgesvd(real(R), full_matrices = 0)

    e = expand_dims(E[:,0], axis = -1)
    J2M = append(zeros((M-1,1)), eye(M-1), axis = 1)
    K = conj(unitary_mat(M-1)).T@J2M@unitary_mat(M)
    
    K1 = real(K)
    K2 = imag(K)

    x1 = K1@e
    x2 = K2@e
    a = pinv(x1)@x2    
    
    iters = 0
    error = 1
    while max_iter>iters and error > min_error:
        A = K2 - a*K1
        W = A@A.T
        num = x1.T@inv(W)@x2
        denom = x1.T@inv(W)@x1
        a_prime = num/denom
        error = abs(a - a_prime)**2
        a = a_prime
        iters += 1 
    wL = 2*arctan(a)
    #################### wR ###########
    R = CR_mapping(R_complex.T)
    F, temp, temp, temp = dgesvd(real(R), full_matrices = 0)

    f = expand_dims(F[:,0], axis = -1)
    J2N = append(zeros((N-1,1)), eye(N-1), axis = 1)
    K = conj(unitary_mat(N-1)).T@J2N@unitary_mat(N)
    
    K1 = real(K)
    K2 = imag(K)

    y1 = K1@f
    y2 = K2@f
    a = pinv(y1)@y2
    
    iters = 0
    error = 1
    while max_iter>iters and error > min_error:
        A = K2 - a*K1
        W = A@A.T
        num = y1.T@inv(W)@y2
        denom = y1.T@inv(W)@y1
        a_prime = num/denom
        error = abs(a - a_prime)**2
        a = a_prime
        iters += 1 
    sai = 2*arctan(a)
    wRk = (sai + 2*pi*arange(-ceil(N/2),ceil(N/2),1) )/(M)
    wR = wRk[0, argmin(abs( wRk - wL ) )]
    w_est = ((M**2-1)*wL + M**2*(N**2-1)*wR) / (M**2*N**2-1)
    return w_est



def DFT_WLS(observed_signal_vector, DFT_interp_num = '7'):
    '''
    This function estimates the frequency of a complex single-tone signal based on DFT-WLS method. 
    For more information please refer to: 
    M. Morelli, M. Moretti, and A. A. D'Amico, "Single-Tone Frequency Estimation by Weighted Least-Squares Interpolation of Fourier Coefficients,"
    IEEE Transactions on Communications, vol. 70, no. 1, pp. 526-537, 2021.

    Parameters
    ----------
    observed_signal_vector : array (K, )
        Vector of time-domain observed signal with the length of K > M*N
    DFT_interp_num: string {'3', '5', or '7'}
           Number of DFT interpolators (L). 
            
    Return
    -------
    w_est: float
        The estimation of the frequency between (-pi, pi)
    '''
    r = observed_signal_vector
    K = len(r)
    X_r = abs(fft.fft(r))
    X_r = append(X_r[int(K/2):], X_r[:int(K/2)])
    k = arange(K)
    k = append(k[int(K/2):]-K, k[:int(K/2)])
    M = k[argmax(X_r)]

    if DFT_interp_num == '3': 
        L = 3 
        c = [0.6969, 1, 0.6969];
        L2 = 1; 
    if DFT_interp_num == '5': 
        L = 5
        c = [0.1347, 0.6969, 1, 0.6969, 0.1347];
        L2 = 2; 
    if DFT_interp_num == '7': 
        L = 7
        c = [0.0567, 0.1300, 0.6138, 1, 0.6138, 0.1300, 0.0567];
        L2 = 3; 
    gamma = sum(c);
    X = zeros(L, dtype = 'complex');
    term1 = zeros(L, dtype = 'complex');
    term2 = zeros(L, dtype = 'complex');
    for k in range(L):
        X[k] = A_WLS(M+k-L2, r);  
        term1[k] = c[k]*X[k];
        term2[k] = c[k]*conj(X[k]);
    term_sum = sum(term1);

    tan_w = 0 ;
    for k in range(L):
        tan_w += term2[k]*(gamma*X[k] - term_sum)*exp(1j*2*pi*(M+k-L2)/K) ; 
    w_est = angle(tan_w)
    return w_est

def Parabolic(observed_signal_vector):
    '''
    This function estimates the frequency of a complex single-tone signal based on Parabolic estimator. 
    For more information please refer to: 
    S. Djukanović, T. Popović, and A. Mitrović: Precise sinusoid frequency estimation based on parabolic 
    interpolation. in Proc. IEEE Telecommunications ForumTelfor (TELFOR) (2016).

    Parameters
    ----------
    observed_signal_vector : array (K, )
        Vector of time-domain observed signal with the length of K
            
    Return
    -------
    w_est: float
        The estimation of the frequency between (-pi, pi)
    '''
    r = observed_signal_vector
    K = len(r)
    X_r = fft.fft(r)
    k = arange(K)
    M = k[argmax(abs(X_r))]
    f_DFT = (M)/K
    
    if M == 0:
        M_neg = K-1
    else:
        M_neg = M - 1
    if M == K-1:
        M_pos = 0
    else:
        M_pos = M + 1
        
    # Step 2: 
    delta_j_num = X_r[M_neg] - X_r[M_pos]
    delta_j_denom = 2*X_r[M] - X_r[M_neg] - X_r[M_pos]

    delta_j = real(delta_j_num / delta_j_denom)
    delta_c1 = (tan(pi / K) / ((pi)/K))*delta_j
    delta_c2 = arctan(delta_c1*pi/K) / (pi/K)
    fc2 = (M + delta_c2)/K

    # Step 3:
    fd = 1/(10*K); 
    theta1 = fc2 - fd
    theta2 = fc2 
    theta3 = fc2 + fd
    P1 = abs(A_par(theta1, r))
    P2 = abs(A_par(theta2, r))
    P3 = abs(A_par(theta3, r))

    # Step 4: 
    theta_num = (theta3**2)*(P1 - P2) + (theta2**2)*(P3 - P1) + (theta1**2)*(P2 - P3)
    theta_denom = theta3*(P1 - P2) + theta2*(P3 - P1) + theta1*(P2 - P3)
    theta_ver = 1/2*theta_num/theta_denom 
    if theta_ver > 0.5:
        theta_ver = theta_ver - 1; 
    return 2*pi*theta_ver


def AM_estimator(observed_signal_vector, iterations = 5):
    '''
    This function estimates the frequency of a complex single-tone signal based on A&M estimator. 
    For more information please refer to: 
    Aboutanios, Elias, and Bernard Mulgrew. "Iterative frequency estimation by interpolation on 
    Fourier coefficients." IEEE Transactions on signal processing 53, no. 4 (2005): 1237-1242.

    Parameters
    ----------
    observed_signal_vector : array (K, )
        Vector of time-domain observed signal with the length of K
    iterations: int
        Number of iterations in the fine search 
            
    Return
    -------
    w_est: float
        The estimation of the frequency between (-pi, pi)
    '''
    r = observed_signal_vector
    Q = iterations
    K = len(r)

    X_r = fft.fft(r)
    k = arange(K)
    M = k[argmax(abs(X_r))]
    
    delta_A = 0
    for q in range(Q):
        S_neg = A_WLS(M - 0.5 + delta_A , r)
        S_pos = A_WLS(M + 0.5 + delta_A , r)
        frac = ( S_pos + S_neg ) / ( S_pos - S_neg )
        sin_int = sin(pi/K)*real(frac)
        
        if abs(sin_int) < 1:
            delta_A += K/(2*pi)*arcsin(sin_int)
        else:
            delta_A += 0 

        f_est = 1/K*(M + delta_A)
        if f_est > 0.5:
            f_est = f_est - 1; 
    return 2*pi*f_est


def LUPUMA(observed_signal_vector, rows_num, col_num):
    '''
    This function estimates the frequency of a complex single-tone signal based on the proposed LUPUMA method. 
    
    Parameters
    ----------
    observed_signal_vector : array (K, )
        Vector of time-domain observed signal with the length of K > M*N
    rows_num: int (M)
        Number of rows in the reshaped matrix (constructed from the signal vector)
    col_num: int (N)
        Number of cloumns in the reshaped matrix (constructed from the signal vector)
            
    Return
    -------
    w_est: float
        The estimation of the frequency between (-pi, pi)
    '''
    r = observed_signal_vector
    M = rows_num
    N = col_num
    R = reshape(r[:M*N], (N,M)).T
    phi_R = CR_mapping(R)
    [U, lam, VT, info] = dgesvd(real(phi_R), full_matrices = 0)
    #################### u:
    u = expand_dims(U[:,0], axis = -1)

    ul = u[:int(M/2)]
    ul0 = ul[:int(M/2)-1]
    ul1 = ul[1:]

    ur = u[int(M/2):]
    ur0 = ur[:int(M/2)-1]
    ur1 = ur[1:]

    ysu = ul0*ur1 - ur0*ul1
    ycu = ul0*ul1 + ur0*ur1
    yeu = ycu + 1j*ysu
    w_LSu = angle(ones((1, int(M/2)-1))@yeu)
    #################### v: 
    v = expand_dims(VT[0,:], axis = -1)
    vl = v[:int(N)]
    vl0 = vl[:int(N)-1]
    vl1 = vl[1:]

    vr = v[int(N):]
    vr0 = vr[:int(N)-1]
    vr1 = vr[1:]

    ysv = vl0*vr1 - vr0*vl1
    ycv = vl0*vl1 + vr0*vr1
    yev = ycv - 1j*ysv
    w_LSv = angle(ones((1, int(N)-1))@yev)
    wVk = ( w_LSv[0,0] + 2*pi*arange(-ceil(M/2), ceil(M/2), 1) )/M
    wV = wVk[argmin(abs( wVk - w_LSu ) )]
    num1 = M-2
    num2 = 2*M**2*(N-1)
    w_est = (num1*w_LSu + num2*wV)/ (num1 + num2)
    return w_est[0,0]
