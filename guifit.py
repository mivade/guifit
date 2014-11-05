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

class Sine(FitFunction):
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

if __name__ == "__main__":
    x = np.linspace(-10, 10, 1000)
    y = np.cos(1.5*x) + np.random.rand(x.shape[0])*.2
    func = Sine(x, y)
    values = guifit(func.xdata, func.ydata, func.func, func.params)
