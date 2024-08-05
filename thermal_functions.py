import sys
from platform import python_version

import matplotlib as mpl
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual

import fnv
import fnv.reduce
import fnv.file
import cmath

from scipy.signal import chirp, find_peaks, peak_widths, medfilt2d, fftconvolve
from scipy.ndimage import laplace, sobel, convolve
import scipy as sp
from scipy.optimize import curve_fit, minimize, fsolve
from scipy.fft import rfft, rfftfreq, irfft
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline, CubicSpline, PchipInterpolator, RegularGridInterpolator, make_interp_spline
import skimage
from skimage.metrics import structural_similarity as ssim
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

from scipy.stats import gaussian_kde
from scipy.stats import norm


from PIL import Image

from datetime import datetime, timezone, time
import glob

import numpy as np

import importlib
import csv
from statsmodels.tsa.stattools import ccf
import math
import seaborn as sns
import pandas as pd

import random

import copy

import pandas as pd 

from IPython.core.display import HTML

import importlib

import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg as splin

# from https://stackoverflow.com/questions/42277989/square-root-scale-using-matplotlib-python
import matplotlib.scale as mscale
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.ticker as ticker

correction_0_degree = 4.38
correction_90_degree = 0.8

def extract_gradient_timestamped(thermal, timestamps, samples=10, size=(28,19)):
    therm_grads = np.zeros(size)
    for x in range(thermal.shape[1]):
        for y in range(thermal.shape[2]):

            x_in = timestamps[:samples]

            #y_out = resize_therm[0:samples,x,y] + soak_value[0:samples,0,0]*0.5

            y_out = thermal[0:samples,x,y]
            #plt.figure()
            #plt.plot(y_out)

            grad, intercept = np.linalg.lstsq(np.vstack([x_in, np.ones(len(x_in))]).T, y_out, rcond=None)[0]

            #resize_therm_grads[x,y] = grad - np.abs(soak_grad) * distance_soak[x,y]
            therm_grads[x,y] = grad

    return therm_grads

# def extract_gradient_timestamped_NUTS(thermal,timestamps,start_indices,cutoff_grad,cutoff_grad2,cutoff_grad3):
#     # example call extract_gradient_timestamped_NUTS(thermal,thermal_timestamps_single[index],start_indices_single[index],cutoff_grad,cutoff_grad2,cutoff_grad3)
    

#     ts = timestamps[start_indices:start_indices+6]
#     gradient_low_sample = extract_gradient_timestamped(thermal,ts,samples=6,size=thermal[0,:,:].shape)
#     ts = timestamps[start_indices:start_indices+10]
#     gradient_high_sample = extract_gradient_timestamped(thermal,ts,samples=10,size=thermal[0,:,:].shape)

#     gradient = gradient_high_sample
#     gradient[gradient >= cutoff_grad] = gradient_low_sample[gradient >= cutoff_grad]
#     ts = timestamps[start_indices:start_indices+5]
#     gradient_lower_sample = extract_gradient_timestamped(thermal,ts,samples=5,size=thermal[0,:,:].shape)

#     gradient[gradient >= cutoff_grad2] = gradient_lower_sample[gradient >= cutoff_grad2]
#     ts = timestamps[start_indices:start_indices+4]
#     gradient_lower_sample2 = extract_gradient_timestamped(thermal,ts,samples=4,size=thermal[0,:,:].shape)

#     gradient[gradient >= cutoff_grad3] = gradient_lower_sample2[gradient >= cutoff_grad3]

#     return gradient

def extract_gradient_timestamped_NUTS(thermal,timestamps,start_indices,cutoffs,samples):
    # example call extract_gradient_timestamped_NUTS(thermal,thermal_timestamps_single[index],start_indices_single[index],cutoff_grad,cutoff_grad2,cutoff_grad3)
    
    ts = timestamps[start_indices:start_indices+samples[0]]
    gradient = extract_gradient_timestamped(thermal,ts,samples=samples[0],size=thermal[0,:,:].shape)

    for index, cutoff in enumerate(cutoffs):
        ts = timestamps[start_indices:start_indices+samples[index+1]]
        gradient_lower_samples = extract_gradient_timestamped(thermal,ts,samples=samples[index+1],size=thermal[0,:,:].shape)
        gradient[gradient>=cutoff] = gradient_lower_samples[gradient>=cutoff]

    return gradient

def extract_gradient_and_temp_timestamped_NUTS(thermal,timestamps,start_indices,cutoffs,samples):
    # example call extract_gradient_timestamped_NUTS(thermal,thermal_timestamps_single[index],start_indices_single[index],cutoff_grad,cutoff_grad2,cutoff_grad3)
    
    ts = timestamps[start_indices:start_indices+samples[0]]
    gradient = extract_gradient_timestamped(thermal,ts,samples=samples[0],size=thermal[0,:,:].shape)
    unsteady_temp = thermal[samples[0],:,:] - thermal[0,:,:]

    for index, cutoff in enumerate(cutoffs):
        ts = timestamps[start_indices:start_indices+samples[index+1]]
        gradient_lower_samples = extract_gradient_timestamped(thermal,ts,samples=samples[index+1],size=thermal[0,:,:].shape)

        unsteady_temp_lower_samples = thermal[samples[index+1],:,:] - thermal[0,:,:]
        unsteady_temp[gradient>=cutoff] = unsteady_temp_lower_samples[gradient>=cutoff]

        gradient[gradient>=cutoff] = gradient_lower_samples[gradient>=cutoff]

    return gradient, unsteady_temp


def lapl(mat, dx, dy):
    """
    Compute the laplacian using `numpy.gradient` twice.
    """
    grad_y, grad_x = np.gradient(mat, dy, dx)
    grad_xx = np.gradient(grad_x, dx, axis=1)
    grad_yy = np.gradient(grad_y, dy, axis=0)
    return(grad_xx + grad_yy)

def lapl_mag(mat, dx, dy):
    """
    Compute the laplacian using `numpy.gradient` twice.
    """
    grad_y, grad_x = np.gradient(mat, dy, dx)
    grad_xx = np.gradient(grad_x, dx, axis=1)
    grad_yy = np.gradient(grad_y, dy, axis=0)
    return np.hypot(grad_xx,grad_yy)

def lapl_sobel(mat,dx,dy):
    grad_x = (sobel(mat, 0)/8)/dx  # horizontal derivative
    grad_y = (sobel(mat, 1)/8)/dy  # vertical derivative

    grad_xx = (sobel(grad_x, 0)/8)/dx  # horizontal derivative
    grad_yy = (sobel(grad_y, 1)/8)/dy  # vertical derivative

    return grad_xx+grad_yy

def lapl_sobel_mag(mat,dx,dy):
    grad_x = (sobel(mat, 0)/8)/dx  # horizontal derivative
    grad_y = (sobel(mat, 1)/8)/dy  # vertical derivative

    grad_xx = (sobel(grad_x, 0)/8)/dx  # horizontal derivative
    grad_yy = (sobel(grad_y, 1)/8)/dy  # vertical derivative

    return np.hypot(grad_xx,grad_yy)

def sobel_2d(mat,dx,dy):
    grad_x = (sobel(mat, 0)/8)/dx  # horizontal derivative
    grad_y = (sobel(mat, 1)/8)/dy  # vertical derivative

    return grad_x+grad_y

def sobel_2d_mag(mat,dx,dy):
    grad_x = (sobel(mat, 0)/8)/dx  # horizontal derivative
    grad_y = (sobel(mat, 1)/8)/dy  # vertical derivative

    return np.hypot(grad_x,grad_y)

def laplace_2d_diag(mat, dx, dy):
    stencil = np.array([[-1, -1, -1],[-1, 8, -1], [-1, -1, -1]])
    return convolve(mat, stencil, mode='nearest') / dx**2

def laplace_2d(mat, dx, dy):
    stencil = np.array([[0, -1, 0],[-1, 4, -1], [0, -1, 0]])
    return convolve(mat, stencil, mode='nearest') / dx**2

def strip_time(time_string):
    time_string_split = time_string.split(":")
    return ":".join(time_string_split[1:])

def load_flir_file(file, stop_frame = 0):
    im = fnv.file.ImagerFile(file)    # open the file
    im.unit = fnv.Unit.TEMPERATURE_FACTORY      # set the desired unit
    if stop_frame == 0:
        stop_frame = im.num_frames
    vid = np.zeros((im.height, im.width,stop_frame),dtype="float32")
    timestamps = np.zeros((stop_frame))
    for i in range(stop_frame):
        im.get_frame(i)                         # get the next frame
        for f in im.frame_info:
            if f['name'] == "Time":
                timestamp = f['value']
                timestamp_time = datetime.strptime(strip_time(timestamp),"%H:%M:%S.%f")
                timestamp_time = timestamp_time.replace(year=1970)
        final = np.array(im.final,copy=True).reshape((im.height, im.width))
        timestamps[i] = timestamp_time.replace(tzinfo=timezone.utc).timestamp()
        vid[:,:,i] = final
    return vid, timestamps
   
def find_elbow(data, theta):

    # make rotation matrix
    co = np.cos(theta)
    si = np.sin(theta)
    rotation_matrix = np.array(((co, -si), (si, co)))

    # rotate data vector
    rotated_vector = data.dot(rotation_matrix)

    # return index of elbow
    return np.where(rotated_vector == rotated_vector.min())[0][0]

def get_data_radiant(data):
  return np.arctan2(data[:, 1].max() - data[:, 1].min(), 
                    data[:, 0].max() - data[:, 0].min())

def read_beast_file(filename,correction=correction_90_degree,plane_axis_1 = "x", plane_axis_2 = "y"):
    size_x = 0
    size_y = 0
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        i=0
        for row in reader:
            i += 1
            if (i==8):
                size_x = np.floor(float(row[0]))
                size_y = np.floor(float(row[1]))
                size_z = np.floor(float(row[2]))
    

    size_axis_1 = size_x

    if plane_axis_1 == "x":
        size_axis_1 = size_x
    elif plane_axis_1 == "y":
        size_axis_1 = size_y
    else:
        size_axis_1 = size_z
        
    with open(filename, newline='') as csvfile:
        i=0
        j=0
        k=0
        reader = csv.reader(csvfile)
        amplitudes = []
        amplitudes_cart = []
        for row in reader:
            i += 1
            if (i >=10 and i%2==0):
                j += 1
                V_Pa = 1.0/1000

                voltages = np.array(row, dtype="float")

                #plt.plot(voltages[7000:8000])
                #plt.show()

                voltages_p = (np.max(voltages) - np.min(voltages))/2

                amplitude = voltages_p / V_Pa

                amplitude = np.sqrt(np.mean(np.square(voltages)))
                amplitude = amplitude / V_Pa
                amplitude = amplitude * np.sqrt(2)

                amplitude = microphone_correction_peak_to_peak(amplitude,correction)

                amplitudes.append(amplitude)
            if( i >=10 and j % size_axis_1 == 0 and amplitudes != []):
                k += 1
                if (k%2 == 0):
                    amplitudes.reverse()
                amplitudes_cart.append(amplitudes)
                amplitudes = []
    amplitudes_cart = np.array(amplitudes_cart)
    #plt.imshow(amplitudes_cart)
    #plt.show()
    return amplitudes_cart

def read_beast_file_rms(filename,correction=correction_90_degree,plane_axis_1 = "x", plane_axis_2 = "y"):
    size_x = 0
    size_y = 0
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        i=0
        for row in reader:
            i += 1
            if (i==8):
                size_x = np.floor(float(row[0]))
                size_y = np.floor(float(row[1]))
                size_z = np.floor(float(row[2]))
    

    size_axis_1 = size_x

    if plane_axis_1 == "x":
        size_axis_1 = size_x
    elif plane_axis_1 == "y":
        size_axis_1 = size_y
    else:
        size_axis_1 = size_z
        
    with open(filename, newline='') as csvfile:
        i=0
        j=0
        k=0
        reader = csv.reader(csvfile)
        amplitudes = []
        amplitudes_cart = []
        for row in reader:
            i += 1
            if (i >=10 and i%2==0):
                j += 1
                V_Pa = 1.0/1000

                voltages = np.array(row, dtype="float")

                #plt.plot(voltages[7000:8000])
                #plt.show()

                voltages_p = (np.max(voltages) - np.min(voltages))/2

                amplitude = voltages_p / V_Pa

                amplitude = np.sqrt(np.mean(np.square(voltages)))
                amplitude = amplitude / V_Pa
                amplitude = amplitude

                amplitude = microphone_correction_peak_to_peak(amplitude,correction)

                amplitudes.append(amplitude)
            if( i >=10 and j % size_axis_1 == 0 and amplitudes != []):
                k += 1
                if (k%2 == 0):
                    amplitudes.reverse()
                amplitudes_cart.append(amplitudes)
                amplitudes = []
    amplitudes_cart = np.array(amplitudes_cart)
    #plt.imshow(amplitudes_cart)
    #plt.show()
    return amplitudes_cart

def read_beast_file_bias(filename,correction=correction_90_degree,plane_axis_1 = "x", plane_axis_2 = "y"):
    size_x = 0
    size_y = 0
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        i=0
        for row in reader:
            i += 1
            if (i==8):
                size_x = np.floor(float(row[0]))
                size_y = np.floor(float(row[1]))
                size_z = np.floor(float(row[2]))
    

    size_axis_1 = size_x

    if plane_axis_1 == "x":
        size_axis_1 = size_x
    elif plane_axis_1 == "y":
        size_axis_1 = size_y
    else:
        size_axis_1 = size_z
        
    with open(filename, newline='') as csvfile:
        i=0
        j=0
        k=0
        reader = csv.reader(csvfile)
        amplitudes = []
        amplitudes_cart = []
        for row in reader:
            i += 1
            if (i >=10 and i%2==0):
                j += 1
                V_Pa = 1.0/1000

                voltages = np.array(row, dtype="float")

                #plt.plot(voltages[7000:8000])
                #plt.show()

                voltages_p = (np.max(voltages) - np.min(voltages))/2

                voltages_p = np.mean(voltages)

                amplitude = voltages_p / V_Pa

                #amplitude = np.sqrt(np.mean(np.square(voltages)))
                #amplitude = amplitude / V_Pa

                amplitude = microphone_correction_peak_to_peak(amplitude,correction)

                amplitudes.append(amplitude)
            if( i >=10 and j % size_axis_1 == 0 and amplitudes != []):
                k += 1
                if (k%2 == 0):
                    amplitudes.reverse()
                amplitudes_cart.append(amplitudes)
                amplitudes = []
    amplitudes_cart = np.array(amplitudes_cart)
    #plt.imshow(amplitudes_cart)
    #plt.show()
    return amplitudes_cart

def read_beast_file_uncorrected(filename):
    size_x = 0
    size_y = 0
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        i=0
        for row in reader:
            i += 1
            if (i==8):
                size_x = np.floor(float(row[0]))
                size_y = np.floor(float(row[1]))
                

    with open(filename, newline='') as csvfile:
        i=0
        j=0
        k=0
        reader = csv.reader(csvfile)
        amplitudes = []
        amplitudes_cart = []
        for row in reader:
            i += 1
            if (i >=10 and i%2==0):
                j += 1
                V_Pa = 1.0/1000

                voltages = np.array(row, dtype="float")

                #plt.plot(voltages[7000:8000])
                #plt.show()

                voltages_p = (np.max(voltages) - np.min(voltages))/2

                amplitude = voltages_p / V_Pa

                amplitudes.append(amplitude)
            if( i >=10 and j % size_x == 0 and amplitudes != []):
                k += 1
                if (k%2 == 0):
                    amplitudes.reverse()
                amplitudes_cart.append(amplitudes)
                amplitudes = []
    amplitudes_cart = np.array(amplitudes_cart)
    #plt.imshow(amplitudes_cart)
    #plt.show()
    return amplitudes_cart

def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

def microphone_correction_peak_to_peak(pressure,correction=correction_90_degree):
    correction_db = correction
    #correction_db = 0
    pressure_rms = pressure / np.sqrt(2)
    ref_p = 0.00002
    pressure_db = 20 * np.log10(pressure_rms/ref_p) - correction_db

    pressure_rms_corrected = np.power(10,pressure_db/20)*ref_p
    return pressure_rms_corrected * np.sqrt(2)


def read_beast_file_mm(filename,correction_db=correction_90_degree):
    measurements = np.zeros((1,300,300))

    with open(filename, newline='') as csvfile:
        i=0
        j=0
        k=0
        reader = csv.reader(csvfile)

        for row in reader:
            i += 1
            if(i >=9 and i %2==1):
                x = int(float(row[0]))
                y = int(float(row[1]))
                z = int(float(row[2]))
                #print(z)
            if (i >=10 and i%2==0):
                j += 1
                V_Pa = 1.0/1000

                voltages = np.array(row, dtype="float")

                #plt.plot(voltages[7000:8000])
                #plt.show()

                voltages_p = (np.max(voltages) - np.min(voltages))/2

                amplitude = voltages_p / V_Pa

                amplitude = microphone_correction_peak_to_peak(amplitude,correction=correction_db)

                measurements[:,y,z] = amplitude

    return measurements

def mic_fft_corr(voltages):
    yf = rfft(normalized_tone)
    xf = rfftfreq(N, 1 / SAMPLE_RATE)

    plt.plot(xf, np.abs(yf))
    plt.show()

    return

    correction_db = correction
    #correction_db = 0
    pressure_rms = pressure / np.sqrt(2)
    ref_p = 0.00002
    pressure_db = 20 * np.log10(pressure_rms/ref_p) - correction_db

    pressure_rms_corrected = np.power(10,pressure_db/20)*ref_p
    return pressure_rms_corrected * np.sqrt(2)



def TVRegDiff(data, itern, alph, u0=None, scale='small', ep=1e-6, dx=None,
              plotflag=True, diagflag=True, precondflag=True,
              diffkernel='abs', cgtol=1e-4, cgmaxit=100):
    """
    Estimate derivatives from noisy data based using the Total 
    Variation Regularized Numerical Differentiation (TVDiff) 
    algorithm.

    Parameters
    ----------
    data : ndarray
        One-dimensional array containing series data to be
        differentiated.
    itern : int
        Number of iterations to run the main loop.  A stopping
        condition based on the norm of the gradient vector g
        below would be an easy modification.  No default value.
    alph : float    
        Regularization parameter.  This is the main parameter
        to fiddle with.  Start by varying by orders of
        magnitude until reasonable results are obtained.  A
        value to the nearest power of 10 is usally adequate.
        No default value.  Higher values increase
        regularization strenght and improve conditioning.
    u0 : ndarray, optional
        Initialization of the iteration.  Default value is the
        naive derivative (without scaling), of appropriate
        length (this being different for the two methods).
        Although the solution is theoretically independent of
        the initialization, a poor choice can exacerbate
        conditioning issues when the linear system is solved.
    scale : {large' or 'small' (case insensitive)}, str, optional   
        Default is 'small'.  'small' has somewhat better boundary
        behavior, but becomes unwieldly for data larger than
        1000 entries or so.  'large' has simpler numerics but
        is more efficient for large-scale problems.  'large' is
        more readily modified for higher-order derivatives,
        since the implicit differentiation matrix is square.
    ep : float, optional 
        Parameter for avoiding division by zero.  Default value
        is 1e-6.  Results should not be very sensitive to the
        value.  Larger values improve conditioning and
        therefore speed, while smaller values give more
        accurate results with sharper jumps.
    dx : float, optional    
        Grid spacing, used in the definition of the derivative
        operators.  Default is the reciprocal of the data size.
    plotflag : bool, optional
        Flag whether to display plot at each iteration.
        Default is True.  Useful, but adds significant
        running time.
    diagflag : bool, optional
        Flag whether to display diagnostics at each
        iteration.  Default is True.  Useful for diagnosing
        preconditioning problems.  When tolerance is not met,
        an early iterate being best is more worrying than a
        large relative residual.
    precondflag: bool, optional
        Flag whether to use a preconditioner for conjugate gradient solution.
        Default is True. While in principle it should speed things up, 
        sometimes the preconditioner can cause convergence problems instead,
        and should be turned off. Note that this mostly makes sense for 'small'
        scale problems; for 'large' ones, the improved preconditioner is one
        of the main features of the algorithms and turning it off defeats the
        point.
    diffkernel: str, optional
        Kernel to use in the integral to smooth the derivative. By default it's
        the absolute value, |u'| (value: "abs"). However, it can be changed to
        being the square, (u')^2 (value: "sq"). The latter produces smoother
        derivatives, whereas the absolute values tends to make them more blocky.
        Default is abs.
    cgtol: float, optional
        Tolerance to use in conjugate gradient optimisation. Default is 1e-4.
    cgmaxit: int, optional
        Maximum number of iterations to use in conjugate gradient optimisation. 
        Default is 100


    Returns
    -------
    u : ndarray
        Estimate of the regularized derivative of data.  Due to
        different grid assumptions, length(u) = length(data) + 1
        if scale = 'small', otherwise length(u) = length(data).
    """

    # Make sure we have a column vector
    data = np.array(data)
    assert len(data.shape) == 1, "data is not one-dimensional"
    # Get the data size.
    n = len(data)

    # Default checking. (u0 is done separately within each method.)
    if dx is None:
        dx = 1.0 / n

    # Different methods for small- and large-scale problems.
    if (scale.lower() == 'small'):

        # Differentiation operator
        d0 = -np.ones(n)/dx
        du = np.ones(n-1)/dx
        dl = np.zeros(n-1)
        dl[-1] = d0[-1]
        d0[-1] *= -1

        D = sparse.diags([dl, d0, du], [-1, 0, 1])
        DT = D.transpose()

        # Antidifferentiation and its adjoint
        def A(x): return (np.cumsum(x) - 0.5 * (x + x[0])) * dx

        def AT(x): return np.concatenate([[sum(x[1:])/2.0],
                                          (sum(x)-np.cumsum(x)+0.5*x)[1:]])*dx

        # Default initialization is naive derivative

        if u0 is None:
            u0 = D*data

        u = u0.copy()
        # Since Au( 0 ) = 0, we need to adjust.
        ofst = data[0]
        # Precompute.
        ATb = AT(ofst - data)        # input: size n

        # Main loop.
        for ii in range(1, itern+1):
            if diffkernel == 'abs':
                # Diagonal matrix of weights, for linearizing E-L equation.
                Q = sparse.spdiags(1. / (np.sqrt((D * u)**2 + ep)), 0, n, n)
                # Linearized diffusion matrix, also approximation of Hessian.
                L = dx * DT * Q * D
            elif diffkernel == 'sq':
                L = dx * DT * D
            else:
                raise ValueError('Invalid diffkernel value')

            # Gradient of functional.
            g = AT(A(u)) + ATb + alph * L * u

            # Prepare to solve linear equation.
            if precondflag:
                # Simple preconditioner.
                P = alph * sparse.spdiags(L.diagonal() + 1, 0, n, n)
            else:
                P = None

            def linop(v): return (alph * L * v + AT(A(v)))
            linop = splin.LinearOperator((n, n), linop)

            s, info_i = sparse.linalg.cg(
                linop, g, x0=None, tol=cgtol, maxiter=cgmaxit,
                callback=None, M=P, atol='legacy')

            # Update solution.
            u = u - s
            # Display plot.
            if plotflag:
                plt.plot(u)
                plt.show()

    elif (scale.lower() == 'large'):

        # Construct anti-differentiation operator and its adjoint.
        def A(v): return np.cumsum(v)

        def AT(w): return (sum(w) * np.ones(len(w)) -
                           np.transpose(np.concatenate(([0.0],
                                                        np.cumsum(w[:-1])))))
        # Construct differentiation matrix.
        c = np.ones(n)
        D = sparse.spdiags([-c, c], [0, 1], n, n) / dx
        mask = np.ones((n, n))
        mask[-1, -1] = 0.0
        D = sparse.dia_matrix(D.multiply(mask))
        DT = D.transpose()
        # Since Au( 0 ) = 0, we need to adjust.
        data = data - data[0]
        # Default initialization is naive derivative.
        if u0 is None:
            u0 = np.concatenate(([0], np.diff(data)))
        u = u0
        # Precompute.
        ATd = AT(data)

        # Main loop.
        for ii in range(1, itern + 1):

            if diffkernel == 'abs':
                # Diagonal matrix of weights, for linearizing E-L equation.
                Q = sparse.spdiags(1. / (np.sqrt((D * u)**2 + ep)), 0, n, n)
                # Linearized diffusion matrix, also approximation of Hessian.
                L = DT * Q * D
            elif diffkernel == 'sq':
                L = DT * D
            else:
                raise ValueError('Invalid diffkernel value')

            # Gradient of functional.
            g = AT(A(u)) - ATd
            g = g + alph * L * u
            # Build preconditioner.
            if precondflag:
                c = np.cumsum(range(n, 0, -1))
                B = alph * L + sparse.spdiags(c[::-1], 0, n, n)
                # droptol = 1.0e-2
                R = sparse.dia_matrix(np.linalg.cholesky(B.todense()))
                P = np.dot(R.transpose(), R)
            else:
                P = None
            # Prepare to solve linear equation.

            def linop(v): return (alph * L * v + AT(A(v)))
            linop = splin.LinearOperator((n, n), linop)

            s, info_i = sparse.linalg.cg(
                linop, -g, x0=None, tol=cgtol, maxiter=cgmaxit, callback=None,
                M=P, atol='legacy')

            # Update current solution
            u = u + s
            # Display plot
            if plotflag:
                plt.plot(u / dx)
                plt.show()

        u = u / dx

    return u

def return_splatial_mean(img, size):
    ones = np.ones_like(img)
    spatial_mean = np.zeros_like(img)
    kernel = np.ones((size, size))
    convol_image = fftconvolve(img[:, :], kernel, mode="same")
    ns = fftconvolve(ones[:, :], kernel, mode="same")
    spatial_mean[:, :] = convol_image / ns

    return spatial_mean

def temporal_noise_reduction_avg(thermal,M,D,threshold):
    for i in range(thermal.shape[0]):
        if i == 0:
            fn_last = 0
        else:
            BF = return_splatial_mean(thermal[i,:,:], D) 
            #BF = skimage.restoration.denoise_bilateral(thermal[i,:,:],D,color_sigma,spatial_sigma,mode="edge")
            xBFr = thermal[i,:,:] - BF
            xBFr[np.abs(xBFr) > threshold] = 0
            fn = ((1.0/M) * xBFr) + ((1-(1.0/M))*fn_last)
            thermal[i,:,:] = thermal[i,:,:] - fn
            fn_last = fn

    return thermal

def temporal_noise_reduction(thermal,M,D=15,color_sigma=25,spatial_sigma=25):
    

    for i in range(thermal.shape[0]):
        if i == 0:
            fn_last = 0
        else:
            BF = cv_mine.bilateralFilter(thermal[i,:,:].astype(np.float32), D, color_sigma, spatial_sigma) 
            #BF = skimage.restoration.denoise_bilateral(thermal[i,:,:],D,color_sigma,spatial_sigma,mode="edge")
            xBFr = thermal[i,:,:] - BF
            fn = ((1.0/M) * xBFr) + ((1-(1.0/M))*fn_last)
            thermal[i,:,:] = thermal[i,:,:] - fn
            fn_last = fn

    return thermal


def remove_banding(frame,corner1,corner2):
    frame_cropped = frame[corner1[0]:corner2[0],corner1[1]:corner2[1]]
    row_noise = np.mean(frame_cropped,axis=(0))
    row_noise_mean = np.mean(row_noise)
    row_noise = row_noise_mean - row_noise

    return frame + row_noise

def remove_banding_horizontal(frame,corner1,corner2):
    frame_cropped = frame[corner1[0]:corner2[0],corner1[1]:corner2[1]]
    row_noise = np.mean(frame_cropped,axis=(1))
    row_noise_mean = np.mean(row_noise)
    row_noise = row_noise-row_noise_mean

    
    return np.einsum("wh -> hw ",np.einsum("hw -> wh" ,frame) - row_noise)

def remove_banding_batch(frame,corner1,corner2):
    frame_cropped = frame[:,corner1[0]:corner2[0],corner1[1]:corner2[1]]
    row_noise = np.mean(frame_cropped,axis=(1))

    row_noise_mean = np.mean(row_noise, axis=(0))

    row_noise = row_noise_mean - row_noise

    frame_new = np.einsum("thw -> twh", frame)

    frame_new = frame_new + row_noise[:,:,np.newaxis]

    return np.einsum("thw -> twh", frame_new)

def remove_banding_both_batch(frame,corner1,corner2, corner3):
    frame_cropped = frame[:,corner1[0]:corner2[0],corner1[1]:corner2[1]]
    row_noise = np.mean(frame_cropped,axis=(1))

    row_noise_mean = np.mean(row_noise, axis=(0))

    row_noise = row_noise_mean - row_noise

    frame_new = np.einsum("thw -> twh", frame)

    frame_new = frame_new + row_noise[:,:,np.newaxis]

    frame_new = np.einsum("thw -> twh", frame_new)

    frame_cropped = frame[:,corner1[0]:corner3[0],corner1[1]:corner3[1]]

    row_noise = np.mean(frame_cropped,axis=(2))

    row_noise_mean = np.mean(row_noise, axis=(0))

    row_noise = row_noise_mean - row_noise

    frame_new = frame_new + row_noise[:,:,np.newaxis]

    return frame_new

# Following function from https://stackoverflow.com/questions/42464334/find-the-intersection-of-two-curves-given-by-x-y-data-with-high-precision-in
def interpolated_intercepts(x, y1, y2):
    """Find the intercepts of two curves, given by the same x data"""

    def intercept(point1, point2, point3, point4):
        """find the intersection between two lines
        the first line is defined by the line between point1 and point2
        the first line is defined by the line between point3 and point4
        each point is an (x,y) tuple.

        So, for example, you can find the intersection between
        intercept((0,0), (1,1), (0,1), (1,0)) = (0.5, 0.5)

        Returns: the intercept, in (x,y) format
        """    

        def line(p1, p2):
            A = (p1[1] - p2[1])
            B = (p2[0] - p1[0])
            C = (p1[0]*p2[1] - p2[0]*p1[1])
            return A, B, -C

        def intersection(L1, L2):
            D  = L1[0] * L2[1] - L1[1] * L2[0]
            Dx = L1[2] * L2[1] - L1[1] * L2[2]
            Dy = L1[0] * L2[2] - L1[2] * L2[0]

            x = Dx / D
            y = Dy / D
            return x,y

        L1 = line([point1[0],point1[1]], [point2[0],point2[1]])
        L2 = line([point3[0],point3[1]], [point4[0],point4[1]])

        R = intersection(L1, L2)

        return R

    idxs = np.argwhere(np.diff(np.sign(y1 - y2)) != 0)

    xcs = []
    ycs = []

    for idx in idxs:
        xc, yc = intercept((x[idx], y1[idx]),((x[idx+1], y1[idx+1])), ((x[idx], y2[idx])), ((x[idx+1], y2[idx+1])))
        xcs.append(xc)
        ycs.append(yc)
    return np.array(xcs), np.array(ycs)

def extract_fwhms_backup(image,window_width=8):
    ind = np.unravel_index([np.argmax(image)], image.shape)

    x = ind[0][0]
    y = ind[1][0]

    max_g = image[x,y]

    image_copy = copy.deepcopy(image)

    #image_copy -= image[x,y] / 2.0

    vertical_slice = image_copy[x,:]

    #plt.figure()
    ##plt.plot(vertical_slice[y-window_width:y+window_width])
    #plt.plot(vertical_slice)
    #plt.show()

    horizontal_slice = image_copy[:,y]

    half_g = max_g / 2

    closest_half = 10000

    fwhms_g = [0,0,0,0]

    for x2 in range(-min(window_width,x),0):
        if abs(image[x+x2,y] - half_g) < closest_half:
            closest_half = abs(image[x+x2,y] - half_g)
            fwhms_g[0] = x2

    closest_half = 10000

    for x2 in range(1,min(image.shape[0]-x,window_width)):
        if abs(image[x+x2,y] - half_g) < closest_half:
            closest_half = abs(image[x+x2,y] - half_g)
            fwhms_g[1] = x2

    closest_half = 10000
    
    for y2 in range(-min(window_width,y),0):
        if abs(image[x,y+y2] - half_g) < closest_half:
            closest_half = abs(image[x,y+y2] - half_g)
            fwhms_g[2] = y2

    closest_half = 10000

    for y2 in range(1,min(image.shape[1]-y,window_width)):
        if abs(image[x,y+y2] - half_g) < closest_half:
            closest_half = abs(image[x,y+y2] - half_g)
            fwhms_g[3] = y2

    def gaus(x,c,a,x0,sigma):
        return c+ a*np.exp(-(x-x0)**2/(2*sigma**2))

    fwhms_final_interp = [0,0]
    try: 
        samples = np.array(list(range(y-window_width,y+window_width+1)))
        samples_plot = np.linspace(y-window_width,y+window_width+1,1000)


        
        n = len(samples)
        u = y
        s = 6.5
        m = np.max(vertical_slice[y-window_width:y+window_width+1])

        popt,pcov = curve_fit(gaus,samples,vertical_slice[y-window_width:y+window_width+1],p0=[0,m,u,s])
        #popt,pcov = curve_fit(gaus,samples,vertical_slice[y-window_width:y+window_width+1])

        fit_data = gaus(samples_plot,popt[0],popt[1],popt[2],popt[3])
        intercept = np.repeat(np.array(np.max(fit_data)/2),1000)
        
        #peaks, _ = find_peaks(fit_data)

        #fwhms_gaussian = peak_widths(fit_data, peaks, rel_height=0.5)

        sample_interval = ((y+window_width+1) - (y-window_width))/1000
        #fwhms_final_interp[0] = (fwhms_gaussian[3] - fwhms_gaussian[2]) * sample_interval

        xcs, ycs = interpolated_intercepts(samples_plot,fit_data,intercept)

        #plt.figure()
        #plt.plot(samples,vertical_slice[y-window_width:y+window_width+1])
        #plt.plot(samples_plot,fit_data)
        #plt.plot(samples_plot, intercept)
        #plt.hlines(np.max(fit_data)/2,x_at_crossing,x_at_crossing2, color="C2")

        #plt.plot((y-window_width)+peaks*sample_interval, fit_data[peaks], "x")

        #plt.hlines(fwhms_gaussian[1],(y-window_width)+fwhms_gaussian[2]*sample_interval,(y-window_width)+fwhms_gaussian[3]*sample_interval, color="C2")

        #plt.show()
        fwhms_final_interp[0] = abs(xcs[1] - xcs[0])
    except:
        fwhms_final_interp[0] = np.array([[0.0]])

    


    try:
        samples = np.array(list(range(x-window_width,x+window_width+1)))
        samples_plot = np.linspace(x-window_width,x+window_width+1,1000)
        
        n = len(samples)
        u = y
        s = 6.5
        m = np.max(horizontal_slice[x-window_width:x+window_width+1])

        popt,pcov = curve_fit(gaus,samples,horizontal_slice[x-window_width:x+window_width+1],p0=[0,m,u,s])
        #popt,pcov = curve_fit(gaus,samples,vertical_slice[y-window_width:y+window_width+1])

        fit_data = gaus(samples_plot,popt[0],popt[1],popt[2],popt[3])
        intercept = np.repeat(np.array(np.max(fit_data)/2),1000)
        
        #peaks, _ = find_peaks(fit_data)

        #fwhms_gaussian = peak_widths(fit_data, peaks, rel_height=0.5)

        sample_interval = ((x+window_width+1) - (x-window_width))/1000
        #fwhms_final_interp[1] = (fwhms_gaussian[3] - fwhms_gaussian[2]) * sample_interval


        #x_at_crossing = fsolve(difference, x0=fwhms_gaussian[2]*sample_interval+(x-window_width))
        #x_at_crossing2 = fsolve(difference, x0=fwhms_gaussian[3]*sample_interval+(x-window_width))

        xcs, ycs = interpolated_intercepts(samples_plot,fit_data,intercept)

        fwhms_final_interp[1] = abs(xcs[1] - xcs[0])
    except:
        fwhms_final_interp[1] = np.array([[0.0]])


    fwhms_final = [0,0]

    fwhms_final[0] = abs(fwhms_g[0]) + abs(fwhms_g[1])
    fwhms_final[1] = abs(fwhms_g[2]) + abs(fwhms_g[3])

    #print()
    #print(fwhms_final_interp)
    #print(fwhms_final)
    #print()

    return fwhms_final_interp

def extract_fwhms(image,window_width=8,print_plots = False):
    ind = np.unravel_index([np.argmax(image)], image.shape)

    x = ind[0][0]
    y = ind[1][0]

    vertical_slice = image[x,:]

    horizontal_slice = image[:,y]

    if print_plots:
        plt.figure()
        ##plt.plot(vertical_slice[y-window_width:y+window_width])
        plt.plot(vertical_slice)
        plt.plot(horizontal_slice)
        plt.show()

    def gaus(x,c,a,x0,sigma):
        return c+ a*np.exp(-(x-x0)**2/(2*sigma**2))

    fwhms_final_interp = [0,0]
    try: 
        samples = np.array(list(range(y-window_width,y+window_width+1)))

        
        u = y
        s = window_width *5/8
        m = np.max(vertical_slice[y-window_width:y+window_width+1])/gaus(u,0,1,u,s)

        popt,pcov = curve_fit(gaus,samples,vertical_slice[y-window_width:y+window_width+1],p0=[0,m,u,s],bounds=([-500,0,u-window_width,0],[1000,20000,u+window_width,100]),ftol=1e-5,xtol=1e-5)


        if print_plots:
            samples_plot = np.linspace(y-window_width*2,y+window_width*2+1,2000)
            fit_data = gaus(samples_plot,popt[0],popt[1],popt[2],popt[3])
            fit_data_guess = gaus(samples_plot,0,m,u,s)
            intercept = np.repeat(np.array(np.max(fit_data)/2),2000)

            xcs, ycs = interpolated_intercepts(samples_plot,fit_data,intercept)
            plt.figure()
            plt.plot(samples,vertical_slice[y-window_width:y+window_width+1], label="data")
            plt.plot(samples_plot,fit_data, label="fit")
            plt.plot(samples_plot,fit_data_guess, label="Guess")
            plt.hlines(np.max(fit_data)/2,np.min(xcs),np.max(xcs), color="C2",label="Intercept FWHM")
            plt.legend()
            plt.show()

        fwhm_y_location_scaler = 0.5 - (popt[0]/(2*popt[1]))
        if fwhm_y_location_scaler <=0:
            fwhms_final_interp[0] = 0
        else:
            fwhms_final_interp[0] = 2 * popt[3] * np.sqrt(np.abs(2*np.log(fwhm_y_location_scaler)))
    except Exception as e:
        fwhms_final_interp[0] = 0

    try:
        samples = np.array(list(range(x-window_width,x+window_width+1)))
        
        u = x
        s = window_width *5/8
        m = np.max(horizontal_slice[x-window_width:x+window_width+1])/gaus(u,0,1,u,s)

        popt,pcov = curve_fit(gaus,samples,horizontal_slice[x-window_width:x+window_width+1],p0=[0,m,u,s],bounds=([-500,0,u-window_width,0],[1000,20000,u+window_width,100]),ftol=1e-5,xtol=1e-5)

        if print_plots:
            samples_plot = np.linspace(x-window_width*2,x+window_width*2+1,2000)
            fit_data = gaus(samples_plot,popt[0],popt[1],popt[2],popt[3])
            fit_data_guess = gaus(samples_plot,0,m,u,s)
            intercept = np.repeat(np.array(np.max(fit_data)/2),2000)

            xcs, ycs = interpolated_intercepts(samples_plot,fit_data,intercept)
            plt.figure()
            plt.plot(samples,horizontal_slice[x-window_width:x+window_width+1],label="data")
            plt.plot(samples_plot,fit_data,label="fit")
            plt.plot(samples_plot,fit_data_guess,label="guess")
            plt.hlines(np.max(fit_data)/2,np.min(xcs),np.max(xcs), color="C2",label="Intercept FWHM")
            plt.legend()
            plt.show()

        fwhm_y_location_scaler = 0.5 - (popt[0]/(2*popt[1]))
        if fwhm_y_location_scaler <=0:
            fwhms_final_interp[1] = 0
        else:
            fwhms_final_interp[1] = 2 * popt[3] * np.sqrt(np.abs(2*np.log(fwhm_y_location_scaler)))
    except Exception as e:
        fwhms_final_interp[1] = 0

    return fwhms_final_interp


def calculate_errors_single(ground_truth, measurement):
    error = ground_truth - measurement
    error_abs = np.abs(ground_truth - measurement)
    rmse = np.sqrt(np.mean(error**2))

    means = np.mean(error)
    maxs = np.max(error_abs)
    mins = np.min(error_abs)

    # fwhms = [np.array(extract_fwhms(ground_truth)), np.array(extract_fwhms(measurement))]
    #return {"raw" :error, "absolute": error_abs, "RMSE": rmse, "mean": means,"max": maxs,"min": mins, "fwhm": np.mean(abs(fwhms[0]-fwhms[1]))}


    return {"raw" :error, "absolute": error_abs, "RMSE": rmse, "mean": means,"max": maxs,"min": mins}

def calculate_errors_double(ground_truth, measurement):
    error = ground_truth - measurement
    error_abs = np.abs(ground_truth - measurement)
    rmse = np.sqrt(np.mean(error**2))

    means = np.mean(error)
    maxs = np.max(error_abs)
    mins = np.min(error_abs)

    #fwhms_left = [np.array(extract_fwhms(ground_truth[:int(ground_truth.shape[0]/2)])), np.array(extract_fwhms(measurement[:int(ground_truth.shape[0]/2)]))]
    #left_error = np.mean(abs(fwhms_left[0]-fwhms_left[1]))
    #fwhms_right = [np.array(extract_fwhms(ground_truth[int(ground_truth.shape[0]/2):])), np.array(extract_fwhms(measurement[int(ground_truth.shape[0]/2):]))]
    #right_error = np.mean(abs(fwhms_right[0]-fwhms_right[1]))
    #return {"raw" :error, "absolute": error_abs, "RMSE": rmse, "mean": means,"max": maxs,"min": mins, "fwhm": np.mean([left_error,right_error])}
    return {"raw" :error, "absolute": error_abs, "RMSE": rmse, "mean": means,"max": maxs,"min": mins}


def plot_6_error_maps(ground_truths, measurements, errors):

    fig, axs = plt.subplots(2, 3,sharex=True, sharey=True,figsize=(15, 10))
    fig2, axs2 = plt.subplots(2, 3,sharex=True, sharey=True,figsize=(15, 10))
    fig3, axs3 = plt.subplots(2, 3,sharex=True, sharey=True,figsize=(15, 10))

    for index in range(1,7):
        error = errors[index]
        ground_truth = ground_truths[index]
        measurement = measurements[index]

        img = axs[(index-1)//3,(index-1)%3].imshow(error, cmap="RdYlBu", vmin=-max(np.abs(np.min(error)),np.max(error)),vmax=max(np.abs(np.min(error)),np.max(error)))
        plt.colorbar(img, ax=axs[(index-1)//3,(index-1)%3])
        axs[(index-1)//3,(index-1)%3].title.set_text(f'{index}kPa target Error')

        img = axs2[(index-1)//3,(index-1)%3].imshow(ground_truth,vmin=0,vmax=np.max((measurement,ground_truth)))
        plt.colorbar(img, ax=axs2[(index-1)//3,(index-1)%3])
        axs2[(index-1)//3,(index-1)%3].title.set_text(f'{index}kPa target - press')

        img = axs3[(index-1)//3,(index-1)%3].imshow(measurement,vmin=0,vmax=np.max((measurement,ground_truth)))
        plt.colorbar(img, ax=axs3[(index-1)//3,(index-1)%3])
        axs3[(index-1)//3,(index-1)%3].title.set_text(f'{index}kPa target - thermal -> press')

def plot_9_error_maps(ground_truths, measurements, errors, labels):

    fig, axs = plt.subplots(3, 3,sharex=True, sharey=True,figsize=(15, 10))
    fig2, axs2 = plt.subplots(3, 3,sharex=True, sharey=True,figsize=(15, 10))
    fig3, axs3 = plt.subplots(3, 3,sharex=True, sharey=True,figsize=(15, 10))

    for index in range(0,9):
        error = errors[index]
        ground_truth = ground_truths[index]
        measurement = measurements[index]

        img = axs[(index)//3,(index)%3].imshow(error, cmap="RdYlBu", vmin=-max(np.abs(np.min(error)),np.max(error)),vmax=max(np.abs(np.min(error)),np.max(error)))
        plt.colorbar(img, ax=axs[(index)//3,(index)%3])
        axs[(index)//3,(index)%3].title.set_text(f'{labels[index]}kPa target Error')

        img = axs2[(index)//3,(index)%3].imshow(ground_truth,vmin=0,vmax=np.max((measurement,ground_truth)))
        plt.colorbar(img, ax=axs2[(index)//3,(index)%3])
        axs2[(index)//3,(index)%3].title.set_text(f'{labels[index]}kPa target - press')

        img = axs3[(index)//3,(index)%3].imshow(measurement,vmin=0,vmax=np.max((measurement,ground_truth)))
        plt.colorbar(img, ax=axs3[(index)//3,(index)%3])
        axs3[(index)//3,(index)%3].title.set_text(f'{labels[index]}kPa target - therm -> press')

def plot_11_error_maps(ground_truths, measurements, errors, labels):

    fig, axs = plt.subplots(4, 3,sharex=True, sharey=True,figsize=(15, 10))
    fig2, axs2 = plt.subplots(4, 3,sharex=True, sharey=True,figsize=(15, 10))
    fig3, axs3 = plt.subplots(4, 3,sharex=True, sharey=True,figsize=(15, 10))

    for index in range(0,11):
        error = errors[index]
        ground_truth = ground_truths[index]
        measurement = measurements[index]

        img = axs[(index)//3,(index)%3].imshow(error, cmap="RdYlBu", vmin=-max(np.abs(np.min(error)),np.max(error)),vmax=max(np.abs(np.min(error)),np.max(error)))
        plt.colorbar(img, ax=axs[(index)//3,(index)%3])
        axs[(index)//3,(index)%3].title.set_text(f'{labels[index]}kPa target Error')

        img = axs2[(index)//3,(index)%3].imshow(ground_truth,vmin=0,vmax=np.max((measurement,ground_truth)))
        plt.colorbar(img, ax=axs2[(index)//3,(index)%3])
        axs2[(index)//3,(index)%3].title.set_text(f'{labels[index]}kPa target - press')

        img = axs3[(index)//3,(index)%3].imshow(measurement,vmin=0,vmax=np.max((measurement,ground_truth)))
        plt.colorbar(img, ax=axs3[(index)//3,(index)%3])
        axs3[(index)//3,(index)%3].title.set_text(f'{labels[index]}kPa target - therm -> press')

def locate_start_thermal(thermal,xs,ys):
    max_timestamp = 100 # previously 100
    ind = np.unravel_index([np.argmax(thermal[max_timestamp,xs[0]:xs[1],ys[0]:ys[1]])], (thermal[max_timestamp,xs[0]:xs[1],ys[0]:ys[1]]).shape)
    #max_timestamp = 100 # previously 100
    #ind = np.unravel_index([np.argmax(thermal[max_timestamp,xs[0]:xs[1],ys[0]:ys[1]]-thermal[0,xs[0]:xs[1],ys[0]:ys[1]])], (thermal[max_timestamp,xs[0]:xs[1],ys[0]:ys[1]]).shape)

    x = ind[0][0] + xs[0]
    y = ind[1][0] + ys[0]

    elbow_data = np.array(list(enumerate((thermal[0:150,x,y]-thermal[0,x,y])*100))) # previously 150 not 200
    #elbow_data = np.array(list(enumerate((thermal[0:150,x,y])*100))) # previously 150 not 200

    #plt.figure()
    #plt.plot(thermal[0:150,x,y]-thermal[0,x,y])
    #plt.show()

    start = find_elbow( elbow_data, get_data_radiant(elbow_data) )
    return start

def get_max_gradient(thermal, start, xs, ys, samples=10, framerate=30.0):
    ind = np.unravel_index([np.argmax(thermal[start+50,xs[0]:xs[1],ys[0]:ys[1]])], (thermal[start+50,xs[0]:xs[1],ys[0]:ys[1]]).shape)

    x = ind[0][0] + xs[0]
    y = ind[1][0] + ys[0]

    x_in = np.array(list(range(samples))) * (1.0/framerate)

    #y_out = resize_therm[0:samples,x,y] + soak_value[0:samples,0,0]*0.5

    y_out = thermal[start:start+samples,x,y]
    #plt.figure()
    #plt.plot(y_out)

    max_gradient, _ = np.linalg.lstsq(np.vstack([x_in, np.ones(len(x_in))]).T, y_out, rcond=None)[0]

    return max_gradient, (x,y)
    

def crop_and_trim_thermal(thermal,start,end,xs,ys):

    trim_thermal = thermal[start:end,:,:]

    therm_start = trim_thermal[:,xs[0]:xs[1],ys[0]:ys[1]]
    return therm_start

def downscale_thermal(cropped_thermal, size, order,frames):
    resize_therm = np.zeros((frames,size[0],size[1]))

    for i in range(frames):
        resize_therm[i] = skimage.transform.resize(cropped_thermal[i],size,order=order,anti_aliasing=True)

    return resize_therm

def downsize_thermal(cropped_thermal, scale_factor, order,frames):
    new_size = skimage.transform.rescale(cropped_thermal[0],1/scale_factor,order=order).shape
    resize_therm = np.zeros((frames,new_size[0],new_size[1]))

    for i in range(frames):
        resize_therm[i] = skimage.transform.rescale(cropped_thermal[i],1/scale_factor,order=order)

    return resize_therm

def extract_gradient(thermal, samples=10, framerate = 30.0, size=(28,19)):
    therm_grads = np.zeros(size)
    for x in range(thermal.shape[1]):
        for y in range(thermal.shape[2]):

            x_in = np.array(list(range(samples))) * (1.0/framerate)

            #y_out = resize_therm[0:samples,x,y] + soak_value[0:samples,0,0]*0.5

            y_out = thermal[0:samples,x,y]
            #plt.figure()
            #plt.plot(y_out)

            grad, intercept = np.linalg.lstsq(np.vstack([x_in, np.ones(len(x_in))]).T, y_out, rcond=None)[0]

            #resize_therm_grads[x,y] = grad - np.abs(soak_grad) * distance_soak[x,y]
            therm_grads[x,y] = grad

    return therm_grads



def extract_gradient_from_curve(thermal, samples=10, framerate = 30.0, size=(28,19)):
    therm_grads = np.zeros(size)
    for x in range(thermal.shape[1]):
        for y in range(thermal.shape[2]):

            x_in = np.array(list(range(samples))) * (1.0/framerate)
            x_in_long = np.array(list(range(230))) * (1.0/framerate)

            #y_out = resize_therm[0:samples,x,y] + soak_value[0:samples,0,0]*0.5

            y_out = thermal[0:samples,x,y]
            y_out_long = thermal[0:230,x,y]
            #plt.figure()
            #plt.plot(y_out)

            grad, intercept = np.linalg.lstsq(np.vstack([x_in, np.ones(len(x_in))]).T, y_out, rcond=None)[0]

            a_g = np.max(y_out_long)-y_out_long[0]
            b_g = 0
            c_g = y_out_long[0]


            try:
                popt, pcov = curve_fit(lambda t, a, b, c: a * (1-np.exp(-b * t)) + c, x_in_long, y_out_long, p0=(a_g,b_g,c_g),)
                a = popt[0]
                b = popt[1]
                c = popt[2]

                #popt, pcov = curve_fit(lambda t, a, b: a * (1-np.exp(-b * t)) + c_g, x_in_long, y_out_long, p0=(a_g,b_g),)
                #a = popt[0]
                #b = popt[1]
                #c = c_g
                
                #popt, pcov = curve_fit(lambda t, b: a_g * (1-np.exp(-b * t)) + c_g, x_in_long, y_out_long, p0=(b_g))
                #b = popt[0]
                #a = a_g
                #c = c_g
                x_time = framerate/samples

                y_ambient = thermal[0,x,y]
                #y_ambient = np.min(thermal)
                x_time = np.log(1-((y_ambient-c)/a)) / (-b)  # interpolate to ambient



                therm_grads[x,y] = a*b*np.exp(-b*x_time)
                if np.isnan(a*b*np.exp(-b*x_time)):
                    therm_grads[x,y] = 0.0001
            except:
                therm_grads[x,y] = 0.0001

            #if (grad<therm_grads[x,y]):
            #    therm_grads[x,y] = grad

    return therm_grads 


def heat_diffusion_model_dual(thermal,mag,therm_center, therm_center2, fwhms=(7,6)):
    mag = mag*np.max(thermal)
    large_size = thermal.shape[1]
    small_size = thermal.shape[0]
    gauss1 = mag*makeGaussian(large_size,fwhms[0],center=therm_center)[:small_size,:]
    gauss2 = mag*makeGaussian(large_size,fwhms[1],center=therm_center)[:small_size,:]

    gauss3 = mag*makeGaussian(large_size,fwhms[0],center=therm_center2)[:small_size,:]
    gauss4 = mag*makeGaussian(large_size,fwhms[1],center=therm_center2)[:small_size,:]
    thermal_corrected = thermal -  (gauss1+gauss3-gauss2-gauss4).clip(max=np.max(gauss1-gauss2))
    return thermal_corrected

def heat_diffusion_model_dual_corrected(thermal,mag,mag2,therm_center, therm_center2, fwhms=(7,6)):
    large_size = thermal.shape[1]
    small_size = thermal.shape[0]

    mag_scaled = mag*np.max(thermal[:,:int(large_size/2)])
    mag2_scaled = mag2*np.max(thermal[:,:int(large_size/2)])

    gauss1 = mag_scaled*makeGaussian(large_size,fwhms[0],center=therm_center)[:small_size,:]
    gauss2 = mag2_scaled*makeGaussian(large_size,fwhms[1],center=therm_center)[:small_size,:]

    mag_scaled = mag*np.max(thermal[:,int(large_size/2):])
    mag2_scaled = mag2*np.max(thermal[:,int(large_size/2):])

    gauss3 = mag_scaled*makeGaussian(large_size,fwhms[0],center=therm_center2)[:small_size,:]
    gauss4 = mag2_scaled*makeGaussian(large_size,fwhms[1],center=therm_center2)[:small_size,:]
    thermal_corrected = thermal -  (gauss1+gauss3-gauss2-gauss4)

    return thermal_corrected

def heat_diffusion_model(thermal,mag,thermal_center,fwhms=(7,6)):
    mag = mag*np.max(thermal)
    large_size = thermal.shape[1]
    small_size = thermal.shape[0]
    gauss1 = mag*makeGaussian(large_size,fwhms[0],center=thermal_center)[:small_size,:]
    gauss2 = mag*makeGaussian(large_size,fwhms[1],center=thermal_center)[:small_size,:]


    thermal_corrected = thermal -  (gauss1-gauss2).clip(max=np.max(gauss1-gauss2))
    return thermal_corrected

def heat_diffusion_model_corrected(thermal,mag,mag2,thermal_center,fwhms=(7,6)):
    mag = mag*np.max(thermal)
    mag2 = mag2*np.max(thermal)
    large_size = thermal.shape[1]
    small_size = thermal.shape[0]
    gauss1 = mag*makeGaussian(large_size,fwhms[0],center=thermal_center)[:small_size,:]
    gauss2 = mag2*makeGaussian(large_size,fwhms[1],center=thermal_center)[:small_size,:]


    thermal_corrected = thermal -  (gauss1-gauss2).clip(max=np.max(gauss1-gauss2)*1.3)
    return thermal_corrected

def heat_diffusion_model_corrected_new(thermal,mag,mag2,thermal_center,fwhms=(7,6)):
    mag = mag*np.max(thermal)
    mag2 = mag2*np.max(thermal)
    large_size = thermal.shape[1]
    small_size = thermal.shape[0]
    gauss1 = mag*makeGaussian(large_size,fwhms[0],center=thermal_center)[:small_size,:]
    gauss2 = 1.5*mag2*(laplace_2d_diag(thermal,0.001,0.001).clip(min=0)/np.max(laplace_2d_diag(thermal,0.001,0.001).clip(min=0)))
    #gauss2 = 0
    


    thermal_corrected = thermal -  (gauss1-gauss2).clip(max=np.max(gauss1-gauss2)*1.3)
    return thermal_corrected

def gradient_to_pressure_25um(gradient, attenuation_coef = 1140, coupling_coef = 1, heat_transfer=1):
    #gradient += 0.25
    return np.sqrt(2) * ( np.sqrt(gradient) * np.sqrt( (1.18**2 * 1006 * 347 * 1140 * 1070) / (2 * attenuation_coef * coupling_coef * 1.18 * 347 + 2 * heat_transfer * 0.115 * 1140 * 1070) ))

def sqrt_physical(x, attenuation, absorbtion):

    air_speed = 347
    air_density = 1.18
    nylon_density = 1140
    nylon_heat_capacity = 1670

    nylon_width = ((np.pi*(3.3e-5)**2)/4) * 17.5 * (2.514888458) * 1e-3 # volume so set height and depth to 0
    nylon_height = 1
    nylon_depth = 1

    air_height = 0.001
    air_width = 0.001 
    air_depth = 6.5e-5

    return np.sqrt( (nylon_heat_capacity * nylon_density * nylon_width * nylon_height * nylon_depth * air_density * air_speed)/((1-np.exp(-2*attenuation*air_depth))*absorbtion*air_width*air_height) ) * np.sqrt(x)

def sqrt_mine(x,a,b):

    return np.where(x >= 0, a*np.power(x,b), 0)

    #return a*np.power(x,b)

def sqrt_mine_bounded(x,a):
    return sqrt_mine(x,a,0.5)

def sqrt_mine_inv(x,a,b):
    return np.power(x/a,1/b)

def calibrate_gradient(gradients, pressures, print_report=False):

    #optimisable_function = lambda x: np.mean(np.abs(pressures[3:] - gradient_to_pressure_25um(gradients[3:], attenuation_coef=x)))

    #result = minimize(optimisable_function, (1000),bounds=[(0,20000)])
    result = {}
    result["x"] = 1000

    params = curve_fit(sqrt_mine_bounded, gradients[:],pressures[:],bounds=([0],[1000000000]))
    [a5,b5] = (params[0][0],0.5)
    #assert(result["success"]) # Fail if optimisation fail

    if print_report:
        print("Gradient Quad coeffs")
        print(a5)
        print(b5)
        #print(f"Attenuation coef: {result['x']}")

        plt.figure()
        plt.xlabel("Pressure (Pa)")
        plt.ylabel("Initial Temperature Gradient (°C/s)")
        plt.plot(sqrt_mine(gradients,a5, b5), gradients, label="Fit Square Root", color="tab:orange")
        plt.plot(sqrt_physical(gradients,835,0.9242634572090549), gradients, label="Phsyical Model")
        plt.plot(pressures, gradients, color="k", marker="*", linestyle='None', label="Measurements")
        plt.legend()
        plt.show()


    return result["x"], a5, b5


def calibrate_steady_state(temp_increases, pressures, print_report=False):

    temp_increases = [0] + temp_increases
    pressures = [0] + pressures

    def sigmoid(x, L ,x0, k, b):
        x = x/10000.0
        y = L / (1 + np.exp(-k*(x-x0))) + b
        return (y*200)

    def inv_sigmoid(y,L,x0,k,b):
        y = y/200
        x = np.log((L/(y-b+1e-8))-1)/-k + x0
        return x*10000
    
    def log_func_inv(x, a, c):
        return a*np.log(x+0.00000001) + c
    
    def log_func(y,a,c):
        return np.exp((y-c)/(a))-0.00000001
    
    coef = np.polyfit(pressures,temp_increases,1)
    poly1d_fn_line = np.poly1d(coef) 

    coef_inv = np.polyfit(temp_increases[13:],pressures[13:],1)
    poly1d_fn_inv_line = np.poly1d(coef_inv) 

    temp_increases_zero = [0,0,0] + list(temp_increases)
    pressures_zero = [0,0,0] + list(pressures)

    params = curve_fit(sqrt_mine, temp_increases[:13],pressures[:13],bounds=([0,0],[1000000000,1]))
    [a6,b6] = params[0]

    coef_inv3 = np.polyfit(temp_increases,pressures,3)
    poly3d_fn_inv_line = np.poly1d(coef_inv3) 

    p0 = [np.max(temp_increases), np.median(pressures/10000.0),1,np.min(temp_increases)]

    params = curve_fit(sigmoid, pressures, temp_increases, p0=p0)
    [L_sig,x0_sig,k_sig,b_sig] = params[0]

    p0 = [300,400]
    params = curve_fit(log_func, pressures[:3], temp_increases[:3], p0=p0)
    [a,c] = params[0]

    if print_report:
        print(pressures)
        print(temp_increases)
        print("Log coefs")
        print(a)
        print(c)
        print("linear coefs")
        print(coef_inv)
        print("Quad coeffs")
        print(a6)
        print(b6)

    pressure_range = np.arange(7000,dtype=np.float64)

    def sig_line(pressure,cutoff=1300):
        temps = np.empty_like(pressure)
        temps[pressure<cutoff] = sigmoid(pressure[pressure<cutoff],L_sig,x0_sig,k_sig,b_sig) - sigmoid(0,L_sig,x0_sig,k_sig,b_sig)
        temps[pressure>=cutoff] = poly1d_fn_line(pressure[pressure>=cutoff])
        return temps

    def sig_line_inv(temps,cutoff=4.67):
        if cutoff >= 0:
            pressure = np.empty_like(temps)
            pressure[temps<cutoff] = inv_sigmoid(temps[temps<cutoff],L_sig,x0_sig,k_sig,b_sig) - inv_sigmoid(0,L_sig,x0_sig,k_sig,b_sig)
            #pressure[temps>=cutoff] = poly1d_fn_inv_line(temps[temps>=cutoff])
            pressure[temps>=cutoff] = poly3d_fn_inv_line(temps[temps>=cutoff])
            #pressure[temps>=cutoff] = poly1d_fn_d_2cm_inv_new(temps[temps>=cutoff])
        else:
            cutoff = abs(cutoff)
            pressure = np.empty_like(temps)
            pressure[temps<cutoff] = inv_sigmoid(temps[temps>cutoff],L_sig,x0_sig,k_sig,b_sig) - inv_sigmoid(0,L_sig,x0_sig,k_sig,b_sig)
            #pressure[temps>=cutoff] = poly1d_fn_inv_line(temps[temps>=cutoff])
            pressure[temps>=cutoff] = poly3d_fn_inv_line(temps[temps<=cutoff])
            #pressure[temps>=cutoff] = poly1d_fn_d_2cm_inv_new(temps[temps>=cutoff])
        return pressure
    
    def log_linear_inv(temps,cutoff=4.67):
        temps[temps<0] = 0
        if cutoff >= 0:
            pressure = np.empty_like(temps)
            pressure[temps<cutoff] = log_func_inv(temps[temps<cutoff],a,c)
            pressure[temps>=cutoff] = poly1d_fn_inv_line(temps[temps>=cutoff])
            #pressure[temps>=cutoff] = poly3d_fn_inv_line(temps[temps>=cutoff])
            #pressure[temps>=cutoff] = poly1d_fn_d_2cm_inv_new(temps[temps>=cutoff])
        else:
            cutoff = abs(cutoff)
            pressure = np.empty_like(temps)
            pressure[temps<cutoff] = log_func_inv(temps[temps<cutoff],a,c)
            pressure[temps>=cutoff] = poly1d_fn_inv_line(temps[temps>=cutoff])
            #pressure[temps<=cutoff] = poly3d_fn_inv_line(temps[temps<=cutoff])
            #pressure[temps>=cutoff] = poly1d_fn_d_2cm_inv_new(temps[temps>=cutoff])

        pressure[pressure<0] = 0
        return pressure
    
    def quad_linear_inv(temps, cutoff=15):
        temps[temps<0] = 0
        pressure = np.empty_like(temps)
        pressure[temps<cutoff] = sqrt_mine(temps[temps<cutoff],a6,b6)
        pressure[temps>=cutoff] = poly1d_fn_inv_line(temps[temps>=cutoff])

        pressure[pressure<0] = 0
        return pressure
    
    temp_range = np.arange(0,45,0.1)

    temp_range_low = np.arange(0,11,0.1)
    temp_range_high = np.arange(11,45,0.1)

    if print_report:
        """
        fig, ax1 = plt.subplots()
        ax1.plot(pressures ,temp_increases,label="Measurements")
        ax1.plot(sig_line_inv(temp_range,cutoff=100),temp_range,label="Sigmoid")
        ax1.plot(poly3d_fn_inv_line(temp_range),temp_range,label="3-degree polynomial")
        ax1.set_xlabel("Pressure measured by microphone (Pa)")
        ax1.set_ylabel("Max temperature difference reached (°C)", color="b")
        ax1.legend()

        plt.show()
        fig, ax1 = plt.subplots()
        ax1.plot(pressures ,temp_increases,label="Measurements")
        #ax1.plot(poly1d_fn_inv_line(temp_range),temp_range,label="Linear")
        ax1.plot(log_linear_inv(temp_range,cutoff=3),temp_range,label="log with linear")
        ax1.set_xlabel("Pressure measured by microphone (Pa)")
        ax1.set_ylabel("Max temperature difference reached (°C)", color="b")
        ax1.legend()
        """

        
        fig, ax1 = plt.subplots()
        ax1.plot(pressures ,temp_increases,label="Measurements",color="k", marker="*", linestyle="None")
        #ax1.plot(poly1d_fn_inv_line(temp_range),temp_range,label="Linear")
        ax1.plot(quad_linear_inv(temp_range_high,cutoff=11),temp_range_high,label="Linear", color="tab:blue")
        ax1.plot(quad_linear_inv(temp_range_low,cutoff=11),temp_range_low,label="0.59 Fractional Power", color="tab:orange")
        ax1.set_xlabel("Pressure (Pa)")
        ax1.set_ylabel("Max Steady State Temperature (°C)")
        ax1.legend()

        """
        plt.show()
        fig, ax1 = plt.subplots()
        ax1.plot(pressures ,temp_increases,"o",label="Measurements")
        #ax1.plot(poly1d_fn_inv_line(temp_range),temp_range,label="Linear")
        ax1.set_xlabel("Pressure measured by microphone (Pa)")
        ax1.set_ylabel("Max temperature difference reached (°C)", color="b")
        ax1.legend()
        """

        #fig, ax1 = plt.subplots()
        #ax1.plot(pressures ,temp_increases,label="Measurements")
        #ax1.plot(sig_line_inv(temp_range,cutoff=6),temp_range,label="Sig with 3-degree polynomial")
        #ax1.set_xlabel("Pressure measured by microphone (Pa)")
        #ax1.set_ylabel("Max temperature difference reached (*C)", color="b")
        #ax1.legend()

    return quad_linear_inv
    #return log_linear_inv


def get_steady_state_functions(coeffs, streaming_cutoff=2.8):

    
    def water_quadratic(x,a,b):

        disc = np.sqrt(b**2 + 4*a*x)/(2*a)

        first_part = -b/(2*a)
        

        return first_part + disc
    
    def true_water_quadratic(x,a):
        return a*np.sqrt(x)
    
    def true_water_quadratic_with_emission(x,a,b,c):
        stefan_boltz = 5.6703e-8
        emissitivity = 0.88
        #return np.sqrt(a*x+b*(x**4))
        return b*np.sqrt(a*x+c*stefan_boltz*emissitivity*((x+294.15)**4-294.15**4))

    def true_water_quadratic_with_emission_variable_h(x,a,b,c):
        stefan_boltz = 5.6703e-8
        emissitivity = 0.88
        #return np.sqrt(a*x+b*(x**4))
        return b*np.sqrt(a*x*x.clip(min=streaming_cutoff)+c*stefan_boltz*emissitivity*((x+294.15)**4-294.15**4))
    
    def true_water_quadratic_variable_h(x,a,b):
        #return np.sqrt(a*x+b*(x**4))
        return b*np.sqrt(a*x*x.clip(min=streaming_cutoff))

    def water_paper_curve(temps):
        temps[temps<0] = 0
        pressure = np.empty_like(temps)
        pressure = sqrt_mine(temps,coeffs[0][0],coeffs[0][1])

        pressure[pressure<0] = 0
        return pressure
    
    def water_paper_curve_quad(temps):
        temps[temps<0] = 0
        pressure = np.empty_like(temps)
        pressure = water_quadratic(temps,coeffs[1][0],coeffs[1][1])

        pressure[pressure<0] = 0
        return pressure
    
    def water_paper_curve_quad_true(temps):
        temps[temps<0] = 0
        pressure = np.empty_like(temps)
        pressure = true_water_quadratic(temps,coeffs[2][0])

        pressure[pressure<0] = 0
        return pressure
    
    def water_paper_curve_quad_emission(temps):
        temps[temps<0] = 0
        pressure = np.empty_like(temps)
        pressure = true_water_quadratic_with_emission(temps,coeffs[3][0],coeffs[3][1],coeffs[3][2])

        pressure[pressure<0] = 0
        return pressure
    
    def water_paper_curve_quad_emission_var_h(temps):
        temps[temps<0] = 0
        pressure = np.empty_like(temps)
        pressure = true_water_quadratic_with_emission_variable_h(temps,coeffs[4][0],coeffs[4][1],coeffs[4][2])

        pressure[pressure<0] = 0
        return pressure
    
    def water_paper_curve_quad_var_h(temps):
        temps[temps<0] = 0
        pressure = np.empty_like(temps)
        pressure = true_water_quadratic_variable_h(temps,coeffs[5][0],coeffs[5][1])

        pressure[pressure<0] = 0
        return pressure
    
    return water_paper_curve, water_paper_curve_quad, water_paper_curve_quad_true, water_paper_curve_quad_emission, water_paper_curve_quad_emission_var_h, water_paper_curve_quad_var_h


def calibrate_steady_state_naive(temp_increases, pressures, print_report=False, sigma=0, use_sigma=False,streaming_cutoff=2.8):

    temp_increases = [0] + temp_increases
    pressures = [0] + pressures

    temp_increases[temp_increases<=0] = 0.0000000000001
    pressures[pressures<=0] = 0.0000000000001

    def water_quadratic(x,a,b):

        disc = np.sqrt(b**2 + 4*a*x)/(2*a)

        first_part = -b/(2*a)
        

        return first_part + disc
    
    def true_water_quadratic(x,a):
        return a*np.sqrt(x)
    
    def true_water_quadratic_with_emission(x,a,b,c):
        stefan_boltz = 5.6703e-8
        emissitivity = 0.88
        #return np.sqrt(a*x+b*(x**4))
        return b*np.sqrt(a*x+c*stefan_boltz*emissitivity*((x+294.15)**4-294.15**4))
    
    def air_speed_estimate(x):

        result = np.empty_like(x)
        result[x<1] = 1
        result[x>=1] = x[x>=1]

        return result

    def true_water_quadratic_with_emission_variable_h(x,a,b,c):
        stefan_boltz = 5.6703e-8
        emissitivity = 0.88
        #return np.sqrt(a*x+b*(x**4))
        return b*np.sqrt(a*x*x.clip(min=streaming_cutoff)+c*stefan_boltz*emissitivity*((x+294.15)**4-294.15**4))
    
    def true_water_quadratic_variable_h(x,a,b):
        #return np.sqrt(a*x+b*(x**4))
        return b*np.sqrt(a*x*x.clip(min=streaming_cutoff))
    
    if not use_sigma:
        params = curve_fit(sqrt_mine, temp_increases,pressures,bounds=([0,0],[1000000000,1]))
        [a7,b7] = params[0]

        params = curve_fit(water_quadratic, temp_increases,pressures,bounds=([0.000000000001,0.000000000001],[10000,10000]))
        [a8,b8] = params[0]

        params = curve_fit(true_water_quadratic, temp_increases,pressures,bounds=([0.000000000001],[10000]))
        [a9] = params[0]

        params = curve_fit(true_water_quadratic_with_emission, temp_increases,pressures,bounds=([0.0000000000000000000000000000000000001,0.0000000000000000000000000000000000001,0],[np.inf,np.inf,np.inf]))
        [a10,b10,c10] = params[0]
        #print(a10,b10,c10)

        params = curve_fit(true_water_quadratic_with_emission_variable_h, temp_increases,pressures,bounds=([0.0000000000000000000000000000000000001,-np.inf,-np.inf],[np.inf,np.inf,np.inf]),maxfev=100000)
        [a11,b11,c11] = params[0]
        #print(a11,b11,c11)

        params = curve_fit(true_water_quadratic_variable_h, temp_increases,pressures,bounds=([0.0000000000000000000000000000000000001,-np.inf],[np.inf,np.inf]),maxfev=100000)
        [a12,b12] = params[0]
        #print(a12,b12)

    else:
        params = curve_fit(sqrt_mine, temp_increases,pressures,bounds=([0,0],[1000000000,1]),sigma=sigma)
        [a7,b7] = params[0]

        params = curve_fit(water_quadratic, temp_increases,pressures,bounds=([0.000000000001,0.000000000001],[10000,10000]),sigma=sigma)
        [a8,b8] = params[0]

        params = curve_fit(true_water_quadratic, temp_increases,pressures,bounds=([0.000000000001],[10000]),sigma=sigma)
        [a9] = params[0]

        params = curve_fit(true_water_quadratic_with_emission, temp_increases,pressures,bounds=([0.0000000000000000000000000000000000001,0.0000000000000000000000000000000000001,-np.inf],[np.inf,np.inf,np.inf]),sigma=sigma)
        [a10,b10,c10] = params[0]
        #print(a10,b10,c10)

        params = curve_fit(true_water_quadratic_with_emission_variable_h, temp_increases,pressures,bounds=([0.0000000000000000000000000000000000001,-np.inf,-np.inf],[np.inf,np.inf,np.inf]),maxfev=100000,sigma=sigma)
        [a11,b11,c11] = params[0]
        #print(a11,b11,c11)

        params = curve_fit(true_water_quadratic_variable_h, temp_increases,pressures,bounds=([0.0000000000000000000000000000000000001,-np.inf],[np.inf,np.inf]),maxfev=100000,sigma=sigma)
        [a12,b12] = params[0]
        #print(a12,b12)

    coeffs = [[a7,b7],[a8,b8],[a9],[a10,b10,c10],[a11,b11,c11],[a12,b12]]

    def water_paper_curve(temps):
        temps[temps<0] = 0
        pressure = np.empty_like(temps)
        pressure = sqrt_mine(temps,a7,b7)

        pressure[pressure<0] = 0
        return pressure
    
    def water_paper_curve_quad(temps):
        temps[temps<0] = 0
        pressure = np.empty_like(temps)
        pressure = water_quadratic(temps,a8,b8)

        pressure[pressure<0] = 0
        return pressure
    
    def water_paper_curve_quad_true(temps):
        temps[temps<0] = 0
        pressure = np.empty_like(temps)
        pressure = true_water_quadratic(temps,a9)

        pressure[pressure<0] = 0
        return pressure
    
    def water_paper_curve_quad_emission(temps):
        temps[temps<0] = 0
        pressure = np.empty_like(temps)
        pressure = true_water_quadratic_with_emission(temps,a10,b10,c10)

        pressure[pressure<0] = 0
        return pressure
    
    def water_paper_curve_quad_emission_var_h(temps):
        temps[temps<0] = 0
        pressure = np.empty_like(temps)
        pressure = true_water_quadratic_with_emission_variable_h(temps,a11,b11,c11)

        pressure[pressure<0] = 0
        return pressure
    
    def water_paper_curve_quad_var_h(temps):
        temps[temps<0] = 0
        pressure = np.empty_like(temps)
        pressure = true_water_quadratic_variable_h(temps,a12,b12)

        pressure[pressure<0] = 0
        return pressure
    
    
    if print_report:
        print("Quad coeffs")
        print(a7)
        print(b7)
        temp_range = np.arange(0,45,0.1)
        fig, ax1 = plt.subplots()
        ax1.plot(pressures ,temp_increases,label="Measurements",color="k", marker="*", linestyle="None")
        ax1.plot(water_paper_curve(temp_range),temp_range,label="water paper power", color="tab:orange")
        ax1.set_xlabel("Pressure (Pa)")
        ax1.set_ylabel("Max Steady State Temperature (°C)")
        ax1.legend()

        fig, ax1 = plt.subplots()
        ax1.plot(pressures ,temp_increases,label="Measurements",color="k", marker="*", linestyle="None")
        ax1.plot(water_paper_curve_quad(temp_range),temp_range,label="water paper quadratic", color="tab:orange")
        ax1.set_xlabel("Pressure (Pa)")
        ax1.set_ylabel("Max Steady State Temperature (°C)")
        ax1.legend()

        fig, ax1 = plt.subplots()
        ax1.plot(pressures ,temp_increases,label="Measurements",color="k", marker="*", linestyle="None")
        ax1.plot(water_paper_curve_quad_true(temp_range),temp_range,label="water paper quadratic true", color="tab:orange")
        ax1.set_xlabel("Pressure (Pa)")
        ax1.set_ylabel("Max Steady State Temperature (°C)")
        ax1.legend()

        fig, ax1 = plt.subplots()
        ax1.plot(pressures ,temp_increases,label="Measurements",color="k", marker="*", linestyle="None")
        ax1.plot(water_paper_curve_quad_emission(temp_range),temp_range,label="water paper quadratic true w/ emission", color="tab:orange")
        ax1.set_xlabel("Pressure (Pa)")
        ax1.set_ylabel("Max Steady State Temperature (°C)")
        ax1.legend()

        fig, ax1 = plt.subplots()
        ax1.plot(pressures ,temp_increases,label="Measurements",color="k", marker="*", linestyle="None")
        ax1.plot(water_paper_curve_quad_emission_var_h(temp_range),temp_range,label="Quadratic into linear w/ emission", color="tab:orange")
        ax1.set_xlabel("Pressure (Pa)")
        ax1.set_ylabel("Max Steady State Temperature (°C)")
        ax1.legend()

        fig, ax1 = plt.subplots()
        ax1.plot(water_paper_curve_quad_true(temp_range),temp_range,label="Square root", color="tab:orange")
        ax1.plot(water_paper_curve_quad_emission(temp_range),temp_range,label="Square root w/ emission", color="tab:blue")
        ax1.plot(water_paper_curve_quad_var_h(temp_range),temp_range,label="Square root w/ streaming", color="tab:green")
        ax1.plot(water_paper_curve_quad_emission_var_h(temp_range),temp_range,label="Square root w/ emission & streaming", color="tab:red")
        ax1.plot(pressures ,temp_increases,label="Measurements",color="k", marker="*", linestyle="None")
        ax1.set_xlabel("Pressure (Pa)")
        ax1.set_ylabel("Max Steady State Temperature (°C)")
        ax1.set_ylim(0,40)
        ax1.set_xlim(0,3500)
        ax1.legend()
    
    return water_paper_curve, water_paper_curve_quad, water_paper_curve_quad_true, water_paper_curve_quad_emission, water_paper_curve_quad_emission_var_h, water_paper_curve_quad_var_h, coeffs
    

def calibrate_steady_state_naive_constant_sigma(temp_increases, pressures,sigma, print_report=False):

    temp_increases = [0] + temp_increases
    pressures = [0] + pressures

    def water_quadratic(x,a,b):

        disc = np.sqrt(b**2 + 4*a*x)/(2*a)

        first_part = -b/(2*a)
        

        return first_part + disc
    
    def true_water_quadratic(x,a):
        return a*np.sqrt(x)
    
    def true_water_quadratic_with_emission(x,a,b):
        stefan_boltz = 5.6703e-8
        emissitivity = 0.88
        #return np.sqrt(a*x+b*(x**4))
        return b*np.sqrt(a*x+stefan_boltz*emissitivity*((x+294.15)**4-294.15**4))
    
    def air_speed_estimate(x):

        result = np.empty_like(x)
        result[x<1] = 1
        result[x>=1] = x[x>=1]

        return result

    def true_water_quadratic_with_emission_variable_h(x,a,b,c):
        stefan_boltz = 5.6703e-8
        emissitivity = 0.88
        #return np.sqrt(a*x+b*(x**4))
        x = x+c
        temp_val = (a*x*air_speed_estimate(x)+stefan_boltz*emissitivity*((x+294.15)**4-294.15**4))
        temp_val[temp_val<=0] = -c
        return b*np.sqrt(temp_val)

    params = curve_fit(sqrt_mine, temp_increases,pressures,bounds=([0,0],[1000000000,1]),sigma=sigma)
    [a7,b7] = params[0]

    params = curve_fit(water_quadratic, temp_increases,pressures,bounds=([0.000000000001,0.000000000001],[10000,10000]),sigma=sigma)
    [a8,b8] = params[0]

    params = curve_fit(true_water_quadratic, temp_increases,pressures,bounds=([0.000000000001],[10000]),sigma=sigma)
    [a9] = params[0]

    params = curve_fit(true_water_quadratic_with_emission, temp_increases,pressures,bounds=([0.0000000000000000000000000000000000001,0.0000000000000000000000000000000000001],[np.inf,np.inf]),sigma=sigma)
    [a10,b10] = params[0]
    print(a10,b10)

    params = curve_fit(true_water_quadratic_with_emission_variable_h, temp_increases,pressures,bounds=([0.0000000000000000000000000000000000001,-np.inf,-np.inf],[np.inf,np.inf,np.inf]),maxfev=10000,sigma=sigma)
    [a11,b11,c11] = params[0]
    print(a11,b11,c11)

    def water_paper_curve(temps):
        temps[temps<0] = 0
        pressure = np.empty_like(temps)
        pressure = sqrt_mine(temps,a7,b7)

        pressure[pressure<0] = 0
        return pressure
    
    def water_paper_curve_quad(temps):
        temps[temps<0] = 0
        pressure = np.empty_like(temps)
        pressure = water_quadratic(temps,a8,b8)

        pressure[pressure<0] = 0
        return pressure
    
    def water_paper_curve_quad_true(temps):
        temps[temps<0] = 0
        pressure = np.empty_like(temps)
        pressure = true_water_quadratic(temps,a9)

        pressure[pressure<0] = 0
        return pressure
    
    def water_paper_curve_quad_emission(temps):
        temps[temps<0] = 0
        pressure = np.empty_like(temps)
        pressure = true_water_quadratic_with_emission(temps,a10,b10)

        pressure[pressure<0] = 0
        return pressure
    
    def water_paper_curve_quad_emission_var_h(temps):
        temps[temps<0] = 0
        pressure = np.empty_like(temps)
        pressure = true_water_quadratic_with_emission_variable_h(temps,a11,b11,c11)

        pressure[pressure<0] = 0
        return pressure
    
    if print_report:
        print("Quad coeffs")
        print(a7)
        print(b7)
        temp_range = np.arange(0,45,0.1)
        fig, ax1 = plt.subplots()
        ax1.plot(pressures ,temp_increases,label="Measurements",color="k", marker="*", linestyle="None")
        ax1.plot(water_paper_curve(temp_range),temp_range,label="water paper power", color="tab:orange")
        ax1.set_xlabel("Pressure (Pa)")
        ax1.set_ylabel("Max Steady State Temperature (°C)")
        ax1.legend()

        fig, ax1 = plt.subplots()
        ax1.plot(pressures ,temp_increases,label="Measurements",color="k", marker="*", linestyle="None")
        ax1.plot(water_paper_curve_quad(temp_range),temp_range,label="water paper quadratic", color="tab:orange")
        ax1.set_xlabel("Pressure (Pa)")
        ax1.set_ylabel("Max Steady State Temperature (°C)")
        ax1.legend()

        fig, ax1 = plt.subplots()
        ax1.plot(pressures ,temp_increases,label="Measurements",color="k", marker="*", linestyle="None")
        ax1.plot(water_paper_curve_quad_true(temp_range),temp_range,label="water paper quadratic true", color="tab:orange")
        ax1.set_xlabel("Pressure (Pa)")
        ax1.set_ylabel("Max Steady State Temperature (°C)")
        ax1.legend()

        fig, ax1 = plt.subplots()
        ax1.plot(pressures ,temp_increases,label="Measurements",color="k", marker="*", linestyle="None")
        ax1.plot(water_paper_curve_quad_emission(temp_range),temp_range,label="water paper quadratic true w/ emission", color="tab:orange")
        ax1.set_xlabel("Pressure (Pa)")
        ax1.set_ylabel("Max Steady State Temperature (°C)")
        ax1.legend()

        fig, ax1 = plt.subplots()
        ax1.plot(pressures ,temp_increases,label="Measurements",color="k", marker="*", linestyle="None")
        ax1.plot(water_paper_curve_quad_emission_var_h(temp_range),temp_range,label="water paper quadratic true w/ emission w/ variable h", color="tab:orange")
        ax1.set_xlabel("Pressure (Pa)")
        ax1.set_ylabel("Max Steady State Temperature (°C)")
        ax1.legend()
    
    return water_paper_curve, water_paper_curve_quad, water_paper_curve_quad_true, water_paper_curve_quad_emission, water_paper_curve_quad_emission_var_h

def heat_soak_model(thermal):
    soak = np.min(thermal)
    soak = np.mean(thermal[0:3,0:3])

    return (thermal-soak).clip(min=0)

def heat_soak_model_no_clip(thermal):
    soak = np.min(thermal)
    soak = np.mean(thermal[0:3,0:3])

    return (thermal-soak)

def extract_steady_state(thermal):
    return np.max(thermal[:,:,:] - thermal[0,:,:],axis=(0))

def extract_steady_state_from_diff(thermal):
    return np.max(thermal[:,:,:]- thermal[0,:,:],axis=(0))

def extract_non_max_steady_state_from_diff(thermal):
    return thermal[257,:,:]- thermal[0,:,:]
    #return thermal[-1,:,:]- thermal[0,:,:]

def extract_mean_non_max_steady_state_from_diff(thermal):
    return np.mean(thermal[237:257,:,:]- thermal[0,:,:],axis=0)
    #return thermal[-1,:,:]- thermal[0,:,:]


def correct(thermal):
    mtx = np.array([[339.97299553,   0.,         170.17580737],
            [  0.,         338.57102361, 153.92281482],
            [  0.,           0.,           1.        ]])
    dist = np.array([[-0.35088421,-0.7898929,0.02189668, -0.01589549,  4.39172435]])
    h,  w = thermal.shape[:2]
    newcameramtx, roi = cv_mine.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # undistort
    mapx, mapy = cv_mine.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
    dst = cv_mine.remap(thermal, mapx, mapy, cv_mine.INTER_LINEAR)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst


test_thermal = f"checkerboard-000001.seq"

thermal, thermal_timestamps = load_flir_file(test_thermal)
import cv2 as cv_mine

def get_size_from_mic_file(mic_file):
    mic_values = read_beast_file_rms(mic_file,plane_axis_1="y")
    #mic_bias_values = read_beast_file_bias(mic_files[index],plane_axis_1="y")
    # TODO: FIX THIS BULLSHIT
    if np.max(mic_values) < 50:
        #mic_values = skimage.transform.resize(mic_values,size,order=3)
        mic_values =  np.reshape(random.sample(list(mic_values.flatten())*5, 37*37),(37,37))
        #mic_bias_values = np.reshape(random.sample(list(mic_bias_values.flatten())*5, 37*37),(37,37))
    mic_values = np.flip(mic_values, axis=(0,1))

    size = mic_values.shape
    return size

def load_thermal_file(thermal_file, xs=None, ys=None, start_lower_bound=0, start_upper_bound=10000, start_default=0):
    un_corr_thermal, thermal_timestamps = load_flir_file(thermal_file)

    un_corr_thermal = np.einsum("hwt -> thw", un_corr_thermal)

    un_corr_thermal_nuc = remove_banding_batch(un_corr_thermal,(0,0),(75,320))

    thermal_corrected = correct(un_corr_thermal[0,:,:])
    thermal_corrected_nuc = correct(un_corr_thermal_nuc[0,:,:])

    thermal = np.zeros((600,thermal_corrected.shape[0],thermal_corrected.shape[1]))
    thermal_nuc = np.zeros((600,thermal_corrected_nuc.shape[0],thermal_corrected_nuc.shape[1]))

    for i in range(600):
        thermal[i,:,:] = correct(un_corr_thermal[i,:,:])
        thermal_nuc[i,:,:] = correct(un_corr_thermal_nuc[i,:,:])
        thermal_nuc[i,:,:] = cv_mine.blur(thermal_nuc[i,:,:],(7,7))

    if xs == None or ys == None:
        # fill in xs and ys with the whole size
        xs = [0,-1]
        ys = [0,-1]

    start = locate_start_thermal(thermal_nuc,xs, ys)
    if (start < start_lower_bound or start > start_upper_bound):
        #print(index,start)
        start = start_default # ADJUST THIS TO WORK FOR SINGLE AND DOUBLE

    length = 260
    end = start + length


    cropped_thermal = crop_and_trim_thermal(thermal, start, end, xs, ys)

    cropped_thermal_nr = crop_and_trim_thermal(thermal_nuc, start, end, xs, ys)

    ambient = np.percentile(thermal[0],10)

    del thermal
    del thermal_nuc

    return thermal_timestamps, start, ambient, cropped_thermal, cropped_thermal_nr
    

def load_thermal_and_mic_file(thermal_file,mic_file, xs, ys,start_lower_bound=10, start_upper_bound=30, start_default=25):
        
    un_corr_thermal, thermal_timestamps = load_flir_file(thermal_file)

    un_corr_thermal = np.einsum("hwt -> thw", un_corr_thermal)

    un_corr_thermal_nuc = remove_banding_batch(un_corr_thermal,(0,0),(75,320))

    thermal_corrected = correct(un_corr_thermal[0,:,:])
    thermal_corrected_nuc = correct(un_corr_thermal_nuc[0,:,:])

    thermal = np.zeros((600,thermal_corrected.shape[0],thermal_corrected.shape[1]))
    thermal_nuc = np.zeros((600,thermal_corrected_nuc.shape[0],thermal_corrected_nuc.shape[1]))

    for i in range(600):
        thermal[i,:,:] = correct(un_corr_thermal[i,:,:])
        thermal_nuc[i,:,:] = correct(un_corr_thermal_nuc[i,:,:])
        thermal_nuc[i,:,:] = cv_mine.blur(thermal_nuc[i,:,:],(7,7))

    start = locate_start_thermal(thermal_nuc,xs, ys)
    if (start < start_lower_bound or start > start_upper_bound):
        #print(index,start)
        start = start_default # ADJUST THIS TO WORK FOR SINGLE AND DOUBLE

    length = 260
    end = start + length

    mic_values = read_beast_file_rms(mic_file,plane_axis_1="y")
    #mic_bias_values = read_beast_file_bias(mic_files[index],plane_axis_1="y")
    # TODO: FIX THIS BULLSHIT
    if np.max(mic_values) < 50:
        #mic_values = skimage.transform.resize(mic_values,size,order=3)
        mic_values =  np.reshape(random.sample(list(mic_values.flatten())*5, 37*37),(37,37))
        #mic_bias_values = np.reshape(random.sample(list(mic_bias_values.flatten())*5, 37*37),(37,37))
    mic_values = np.flip(mic_values, axis=(0,1))

    size = mic_values.shape

    # START OF REGISTRATION
    #if index not in [0,1,2]:

    xs2 = xs
    ys2 = ys

    if np.max(mic_values) > 400:


        cropped_thermal = crop_and_trim_thermal(thermal_nuc, start, end, [0,-1], [0,-1])
        img = extract_steady_state(cropped_thermal).astype(np.float32)

        compensated_thermal = extract_gradient(cropped_thermal,samples=10,framerate=30,size=cropped_thermal[0,:,:].shape)

        compensated_thermal[compensated_thermal<0] = 0.000001
        #pressure_from_thermal_non_crop = gradient_to_pressure_25um(compensated_thermal,attenuation_coef=1950,heat_transfer=1)
        pressure_from_thermal_non_crop = sqrt_mine(compensated_thermal,770.4764592460847,0.5114099301314744)
        img2 = pressure_from_thermal_non_crop.astype(np.float32)

        #template = skimage.transform.resize(mic_values,cropped_thermal[0,:,:].shape,order=3).astype(np.float32)

        results = []
        results2 = []
        # 3.0819 optimal scale      

        method = cv_mine.TM_CCOEFF_NORMED

        scaling_factor = 3

        for scale in np.arange(3,3.1,0.0001): # 3.392 optimal value?
            template = skimage.transform.rescale(mic_values,scale,order=3).astype(np.float32)
            #template = mic_values.astype(np.float32)

            w, h = template.shape[::-1]
            # All the 6 methods for comparison in a list
            method = cv_mine.TM_CCOEFF_NORMED
            # Apply template Matching
            res = cv_mine.matchTemplate(img,template,method)
            min_val, max_val, min_loc, max_loc = cv_mine.minMaxLoc(res)
            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            results.append([max_val,scale])

        scaling_factor = sorted(results,key=lambda x: x[0],reverse=True)[0][1]
        template = skimage.transform.rescale(mic_values,scaling_factor,order=3).astype(np.float32)
        w, h = template.shape[::-1]

        # Apply template Matching
        res = cv_mine.matchTemplate(img,template,method)
        min_val, max_val, min_loc, max_loc = cv_mine.minMaxLoc(res)
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        xs = [top_left[1],bottom_right[1]]
        ys = [top_left[0],bottom_right[0]]

        for scale in np.arange(3,3.1,0.0001): # 3.392 optimal value?
            template = skimage.transform.rescale(mic_values,scale,order=3).astype(np.float32)
            #template = mic_values.astype(np.float32)

            w, h = template.shape[::-1]
            # All the 6 methods for comparison in a list
            method = cv_mine.TM_CCOEFF_NORMED
            # Apply template Matching
            res = cv_mine.matchTemplate(img2,template,method)
            min_val, max_val, min_loc, max_loc = cv_mine.minMaxLoc(res)
            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            results2.append([max_val,scale])

        scaling_factor = sorted(results2,key=lambda x: x[0],reverse=True)[0][1]
        template = skimage.transform.rescale(mic_values,scaling_factor,order=3).astype(np.float32)
        w, h = template.shape[::-1]

        # Apply template Matching
        res = cv_mine.matchTemplate(img2,template,method)
        min_val, max_val, min_loc, max_loc = cv_mine.minMaxLoc(res)
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        xs2 = [top_left[1],bottom_right[1]]
        ys2 = [top_left[0],bottom_right[0]]

        # END OF REGISTRATION

    cropped_thermal = crop_and_trim_thermal(thermal, start, end, xs, ys)
    cropped_thermal2 = crop_and_trim_thermal(thermal, start, end, xs2, ys2)

    cropped_thermal_nr = crop_and_trim_thermal(thermal_nuc, start, end, xs, ys)
    cropped_thermal2_nr = crop_and_trim_thermal(thermal_nuc, start, end, xs2, ys2)

    ambient = np.percentile(thermal[0],10)

    # TESTING DOWNSCALE AGAIN!
    downscaled_thermals = (downscale_thermal(cropped_thermal,size,3,length))
    downscaled_thermals2 = (downscale_thermal(cropped_thermal2,size,3,length))

    del cropped_thermal
    del cropped_thermal2

    downscaled_thermals_nr = downscale_thermal(cropped_thermal_nr,size,3,length)
    downscaled_thermals2_nr = downscale_thermal(cropped_thermal2_nr,size,3,length)

    del cropped_thermal_nr
    del cropped_thermal2_nr

    del thermal
    del thermal_nuc

    cutoffs = [1,5,12.5,15,25]
    samples = [10,9,8,6,5,4]

    return thermal_timestamps, start, mic_values, ambient, downscaled_thermals, downscaled_thermals2, downscaled_thermals_nr, downscaled_thermals2_nr

def calculate_attenuation_mesh(frequency, air_density, dynamic_viscosity_air, air_heat_capacity, air_conductivity, width, air_speed, adiabatic_index):

    # attenuation

    k_v2 = -1j * 2 * np.pi * frequency * air_density / dynamic_viscosity_air
    k_th2 = -1j * 2 * np.pi * frequency * air_density* air_heat_capacity / air_conductivity

    

    m_dash = lambda m: np.pi*(m+0.5)
    a_m_v = lambda m: np.sqrt(k_v2 - (2*m_dash(m)/width)**2)
    a_m_t = lambda m: np.sqrt(k_th2- (2*m_dash(m)/width)**2)

    inner_sum_th = lambda m: 2* np.power(a_m_t(m)*m_dash(m),-2) * (1 - (np.tan(a_m_t(m)*width/2))/(a_m_t(m) * width/2))

    inner_sum_v = lambda m: 2* np.power(a_m_v(m)*m_dash(m),-2) * (1 - (np.tan(a_m_v(m)*width/2))/(a_m_v(m) * width/2))

    upsilon_thermal = k_th2 * np.sum( [ inner_sum_th(m) for m in range(0,51) ] )
    upsilon_viscous = k_v2 * np.sum( [ inner_sum_v(m) for m in range(0,51) ] )

    complex_wave_number_manual = ((2*np.pi*frequency/air_speed)**2) * ((adiabatic_index - (adiabatic_index-1)*upsilon_thermal)/(upsilon_viscous))

    attenuation =  np.abs(np.imag(np.sqrt(complex_wave_number_manual)))
    return attenuation

def calculate_absorbtion_mesh(nylon_depth, width, porosity, air_density, dynamic_viscosity_air, frequency, air_speed):

    t = D_n = nylon_depth
    a = width/2
    d = width
    sigma = porosity
    rho = air_density
    mu = dynamic_viscosity_air
    omega = 2*np.pi*frequency
    c = air_speed

    k = d*np.sqrt((omega * rho)/(4*mu))

    k_r = np.sqrt(1+(k**2)/32) + np.sqrt(2)/32 * k * (d/t)

    kappa = (32*mu*t*k_r) / (sigma * rho * c * (d**2))

    k_m = 1 + np.power(1+(k**2)/2, -1/2) + 0.85 * d/t
    omega_m = omega * t / (porosity * c) * k_m

    absorbtion = 4*kappa / ((1+kappa)**2 + omega_m**2) 

    return absorbtion

def press_from_gradient_mesh(gradients, attenuation, absorbtion, nylon_thread_radius, pixel_size, nylon_depth, nylon_heat_capacity, nylon_density, air_density, air_speed):

    pixel_size = 1e-3


    nylon_volume = ((np.pi*(nylon_thread_radius)**2)/4) * 17.5 * (2.514888458) * pixel_size
    #nylon_width = ((np.pi*(3.3e-5)**2)/4) * 17.5 * (2.514888458) * 1e-3 # volume so set height and depth to 0
    #nylon_height = 1
    #nylon_depth = 1

    air_height = pixel_size
    air_width = pixel_size
    air_depth = nylon_depth

    return np.sqrt( (nylon_heat_capacity * nylon_density *  nylon_volume * air_density * air_speed)/((1-np.exp(-2*attenuation*air_depth))*absorbtion*air_width*air_height) ) * np.sqrt(gradients)

def press_from_grad_fit(x,a,b):
    return a*np.power(x,b)