from typing import List, NoReturn, Optional
import math

import bdsim
import control as ctrl
import numpy as np


class Derivator(bdsim.SubsystemBlock):
    """A derivator block.

    This block implements the derivator as a low-pass filter with a fixed
    cutoff frequency.

    Parameters
    ----------
    simulation: bdsim.BDSim
        The runtime for the block diagram.
    cutoff : float
        The cutoff frequency of the low-pass filter.
    gain : float, optional
        The derivator gain. (The default is ``1.0``)
    """
    nin : int = 1
    nout : int = 1

    def __init__(self, simulation: bdsim.BDSim, cutoff: float, gain: float = 1.0, **blockargs) -> NoReturn:
        self._cutoff, self._gain = cutoff, gain
        super().__init__(**blockargs)
        self.type = "subsystem"
        diagram = simulation.blockdiagram()
        self.inport = diagram.INPORT(1)
        self.outport = diagram.OUTPORT(1)
        self.subsystem = diagram
        self.ssname = "derivator"
        if gain:
            self._build_low_pass_filter(diagram)
        else:
            self._build_null_gain(diagram)
    
    def _build_low_pass_filter(self, diagram: bdsim.BlockDiagram) -> NoReturn:
        feedback = diagram.SUM("+-")
        integrator = diagram.INTEGRATOR()
        derivator_gain = diagram.GAIN(self._gain)
        cutoff_gain = diagram.GAIN(self._cutoff)
        diagram.connect(self.inport, derivator_gain)
        diagram.connect(derivator_gain, feedback[0])
        diagram.connect(feedback, cutoff_gain)
        diagram.connect(cutoff_gain, self.outport, integrator)
        diagram.connect(integrator, feedback[1])

    def _build_null_gain(self, diagram: bdsim.BlockDiagram) -> NoReturn:
        gain = diagram.GAIN(0.0)
        diagram.connect(self.inport, gain)
        diagram.connect(gain, self.outport)
    
    @property
    def h(self) -> ctrl.TransferFunction:
        if self._gain:
            return ctrl.tf([self._gain, 0], [1 / self._cutoff, 1])
        return ctrl.tf(0, 1)


class PID(bdsim.SubsystemBlock):
    """A PID controller.

    This block implements a controller (P, I, D, PI, PD or PID).

    Parameters
    ----------
    simulation: bdsim.BDSim
        The runtime for the block diagram.
    k_p : float
        The proportional gain.
    k_i : float
        The integrator gain. If set to ``0.0``, the integrator is replaced by
        a zero gain block.
    k_d : float
        The derivator gain. If set to ``0.0``, the derivator is replaced by a
        zero gain block.
    cutoff : float, optional
        The cutoff frequency of the derivator. (The default is ``0.0``)
    """
    nin : int = 1
    nout : int = 1

    def __init__(self, simulation: bdsim.BDSim, k_p: float, k_i: float, k_d: float, derivator_cutoff: float = 0.0, **blockargs) -> NoReturn:
        self.k_p, self.k_i, self.k_d = k_p, k_i, k_d
        self.cutoff = derivator_cutoff
        super().__init__(**blockargs)
        self.type = "subsystem"
        diagram = simulation.blockdiagram()
        self.inport = diagram.INPORT(1)
        self.outport = diagram.OUTPORT(1)
        self.subsystem = diagram
        self.ssname = "controller"
        self.sum = diagram.SUM("+++")
        proportional = diagram.GAIN(k_p)
        integrator = diagram.INTEGRATOR(gain=k_i) if k_i else diagram.GAIN(0)
        self.derivator = Derivator(simulation, derivator_cutoff, k_d)
        diagram.add_block(self.derivator)
        diagram.connect(self.inport, proportional, integrator, self.derivator)
        diagram.connect(proportional, self.sum[0])
        diagram.connect(integrator, self.sum[1])
        diagram.connect(self.derivator, self.sum[2])
        diagram.connect(self.sum, self.outport)
    
    @property
    def h_p(self) -> ctrl.TransferFunction:
        return ctrl.tf(self.k_p, 1)
    
    @property
    def h_i(self) -> ctrl.TransferFunction:
        return ctrl.tf(self.k_i, [1, 0]) if self.k_i else ctrl.tf(0, 1)
    
    @property
    def h_d(self) -> ctrl.TransferFunction:
        return self.derivator.h
    
    @property
    def h(self) -> ctrl.TransferFunction:
        return self.h_p + self.h_i + self.h_d


class Exponential(bdsim.SourceBlock, bdsim.EventSource):
    """A source block generating an exponential curve.
    
    Parameters
    ----------
    T : float, optional
        The time at which the signal starts. (The default is ``1.0``)
    off : float, optional
        The value of the signal before it starts. (The default is ``0.0``)
    exp : float, optional
        The exponent of the signal. (The default is ``2.0``)
    """
    nin : int = 0
    nout : int = 1

    def __init__(self, T: float = 1.0, off: float = 0.0, exp: float = 2.0 , **blockargs) -> NoReturn:
        super().__init__(**blockargs)
        self.T = T
        self.off = off
        self.exp = exp
    
    def start(self, state: Optional[bdsim.BDSimState] = None) -> NoReturn:
        state.declare_event(self, self.T)
    
    def output(self, t: Optional[float] = None) -> List[float]:
        if t >= self.T:
            return [self.off + (t - self.T) ** self.exp]
        return [self.off]


class Reference(bdsim.SourceBlock, bdsim.EventSource):
    """A source block generating different signals.

    This block outputs different signal based on the value of the `shape`
    parameter.
    
    Parameters
    ----------
    shape : str, {"step", "ramp", "parabole"}
        The shape of the signal.
    T : float, optional
        The time at which the signal starts. (The default is ``1.0``)
    off : float, optional
        The value of the signal before it starts. (The default is ``0.0``)
    val : float, optional
        The characteristic of the signal. (The default is ``1.0``)
    """
    nin : int = 0
    nout : int = 1

    def __init__(self, shape: str, T: float = 1.0, off: float = 0.0, val: float = 1.0 , **blockargs) -> NoReturn:
        super().__init__(**blockargs)
        self.shape = shape
        self.T = T
        self.off = off
        self.val = val
    
    def start(self, state: Optional[bdsim.BDSimState] = None) -> NoReturn:
        state.declare_event(self, self.T)
    
    def output(self, t: Optional[float] = None) -> List[float]:
        if t >= self.T:
            if self.shape == "step":
                return [self.val]
            elif self.shape == "ramp":
                return [self.off + (t - self.T) * self.val]
            elif self.shape == "parabole":
                return [self.off + (t - self.T)**2 * self.val / 2]
        return [self.off]


class Noise(bdsim.SourceBlock):
    """A source block generating noise.
    
    This block outputs a normally distributed noise.
    
    Parameters
    ----------
    mean : float
        The mean of the normal distribution.
    std : float
        The standard deviation of the distribution.
    buffer_length : int, optional
        The number of random values drawn at a time. This is intended to
        reduce the overhead in order to lower the computation time. (The
        default is ``1000``)
    """
    nin : int = 0
    nout : int = 1

    def __init__(self, mean: float, std: float, buffer_length: int = 1000, **blockargs) -> NoReturn:
        self._buffer_length = buffer_length
        self.mean = mean
        assert 0 <= std, "Standard deviation must be positive"
        self.std = std
        self._initialize_buffer()
        super().__init__(**blockargs)
    
    def _initialize_buffer(self) -> NoReturn:
        self._buffer = np.random.normal(self.mean, self.std, self._buffer_length)
        self._i = 0

    def output(self, t=None) -> list[float]:
        try:
            sample = self._buffer[self._i]
        except IndexError:
            self._initialize_buffer()
            sample = self._buffer[self._i]
        self._i += 1
        return [sample]


class Disturbance(bdsim.SourceBlock):
    """A source block generating disturbances.
    
    This block outputs different signal based on the value of the `shape`
    parameter.
    """
    nin : int = 0
    nout : int = 1

    def __init__(self, shape: str, vals: List[float], T: float = 1.0, off: float = 0.0, **blockargs):
        super().__init__(**blockargs)
        self.shape = shape
        self.T = T
        self.off = off
        self.vals = vals
    
    def start(self, state: Optional[bdsim.BDSimState] = None) -> NoReturn:
        state.declare_event(self, self.T)
        if self.shape == "step":
            state.declare_event(self, self.T + self.vals[1])
    
    def output(self, t: Optional[float] = None) -> List[float]:
        if self.shape == "step" and self.T <= t <= (self.T + self.vals[1]):
            return [self.vals[0]]
        elif self.shape == "sin" and t >= self.T:
            return [self.off + self.vals[0] * math.sin(self.vals[1] * 2 * math.pi * (t - self.T))]
        return [self.off]