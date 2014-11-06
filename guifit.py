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
        s = FitParam("Std Dev", (xdata[-1] - xdata[0])/3., 0, np.max(xdata) - np.min(xdata))
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

# Interactive fitting
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os.path
    import inspect
    import pickle
    import guidata
    import guidata.dataset.datatypes as dt
    import guidata.dataset.dataitems as di
    app = guidata.qapplication()

    # Get all possible functions
    functions = []
    _excluded = ['ABCMeta', 'FitFunction', 'FitParam']
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj) and name not in _excluded:
            functions.append((name, name))

    # Get last used values, if present
    config_file = os.path.expanduser('~/.guifit')
    if not os.path.exists(config_file):
        config = {
            'file': '',
            'directory': os.path.dirname(config_file),
            'function': 'ExpDecay',
            'xcol': 0,
            'ycol': 1,
            'skiprows': 1
        }
    else:
        with open(config_file, 'r') as f:
            config = pickle.load(f)

    # Request data file and fit function to use
    class InteractiveFitSettings(dt.DataSet):
        """Specify file and fit function"""
        filename = di.FileOpenItem(
            "Data file", ('dat', 'csv', 'tsv', 'txt'),
            default=config['file'], basedir=config['directory']
        )
        funcname = di.ChoiceItem("Fit function", functions, default=config['function'])
        xcol = di.IntItem("x data column", default=config['xcol'], min=0)
        ycol = di.IntItem("y data column", default=config['ycol'], min=0)
        skiprows = di.IntItem("Rows to skip", default=config['skiprows'], min=0)
    interactive = InteractiveFitSettings()
    if not interactive.edit(size=(640, 1)):
        sys.exit(0)
    else:
        config['file'] = interactive.filename
        config['function'] = interactive.funcname
        config['xcol'] = interactive.xcol
        config['ycol'] = interactive.ycol
        config['skiprows'] = interactive.skiprows
        with open(config_file, 'w') as f:
            pickle.dump(config, f)

    # Open data file
    name, extension = os.path.splitext(interactive.filename)
    if extension == 'csv':
        delimiter = ','
    else:
        delimiter = None
    xdata, ydata = np.loadtxt(
        interactive.filename, delimiter=delimiter,
        usecols=(interactive.xcol, interactive.ycol),
        unpack=True, skiprows=interactive.skiprows
    )

    # Fit
    #x = np.linspace(-10, 10, 1000)
    #y = np.cos(1.5*x) + np.random.rand(x.shape[0])*.2
    #func = Sine(x, y)
    func = getattr(sys.modules[__name__], interactive.funcname)(xdata, ydata)
    values = guifit(func.xdata, func.ydata, func.func, func.params)
