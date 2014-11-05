from __future__ import division
from abc import abstractmethod, ABCMeta
import numpy as np
from guiqwt.widgets.fit import FitParam, guifit

class FitFunction(object):
    """Generic class for defining new fit functions.

    Notes
    -----
    TODO: Documentation on how to define new classes.

    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, xdata, ydata):
        pass

    @abstractmethod
    def func(self, x, params):
        pass

# Oscillations
# -----------------------------------------------------------------------------
        
class Sine(FitFunction):
    """Sinusoids::

      f(x, A, B, f, phi) = A*sin(f*x + phi) + B
    
    """
    def __init__(self, xdata, ydata):
        A = FitParam("Amplitude", abs(np.max(ydata)), 0, 100)
        B = FitParam("Offset", np.mean(ydata), -100, 100)
        _f0 = np.median(xdata)
        f = FitParam("Frequency", _f0, xdata[0], xdata[-1])
        phi = FitParam("Phase", 0, 0, 2*np.pi)
        self.params = [A, B, f, phi]
        self.xdata = xdata
        self.ydata = ydata

    def func(self, x, params):
        A, B, f, phi = params
        return A*np.sin(f*x + phi) + B

# Peak fitting
# -----------------------------------------------------------------------------

class Guassian(FitFunction):
    """Gaussian profiles::

      f(x, A, B, x0, s) = A*exp(-(x - x0)**2/(2*s**2)) + B

    """
    def __init__(self, xdata, ydata):
        A = FitParam("Amplitude", np.max(ydata), 0, 1.5*np.max(ydata))
        _ys = 4*np.std(ydata)
        B = FitParam("Offset", np.mean(ydata), -_ys, _ys)
        x0 = FitParam("Center", xdata[np.argmax(ydata)], xdata[0], xdata[-1])
        s = FitParam("Std Dev", (xdata[-1] - xdata[0])/3., 0, np.max(xdata))
        self.params = [A, B, x0, s]
        self.xdata = xdata
        self.ydata = ydata

    def func(self, x, params):
        A, B, x0, s = params
        return A*np.exp(-(x - x0)**2/(2*s**2)) + B

# Exponentials
# -----------------------------------------------------------------------------

class ExpDecay(FitFunction):
    """Exponential decay::

      f(x, A, B, tau) = A*exp(-x/tau) + B
    
    """
    def __init__(self, xdata, ydata):
        A = FitParam("Amplitude", np.max(ydata), 0, 1.5*np.max(ydata))
        B = FitParam("Offset", 0, -np.mean(ydata), np.mean(ydata))
        tau = FitParam("Lifetime", 0.3*(xdata[-1] - xdata[0]), 0, xdata[0])
        self.params = [A, B, tau]
        self.xdata = xdata
        self.ydata = ydata

    def func(self, x, params):
        A, B, tau = params
        return A*np.exp(-x/tau) + B

# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    x = np.linspace(-10, 10, 1000)
    y = np.cos(1.5*x) + np.random.rand(x.shape[0])*.2
    func = Sine(x, y)
    values = guifit(func.xdata, func.ydata, func.func, func.params)
