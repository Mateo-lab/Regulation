from contextlib import redirect_stdout
import os
import sys
from typing import Any, Dict, NoReturn

import bdsim
import control as ctrl
from scipy.interpolate import interp1d
import numpy as np
import numpy.typing as npt
import pandas as pd

import blocks


class System:

    def __init__(self, plant: bdsim.SubsystemBlock, reference_kwargs: Dict[str, Any], controller_kwargs: Dict[str, Any], noise_kwargs: Dict[str, Any], disturbance_kwargs: Dict[str, Any], **kwargs) -> NoReturn:
        self._build_diagram(plant, reference_kwargs, controller_kwargs, noise_kwargs, disturbance_kwargs, **kwargs)
    
    def _build_diagram(self, plant: bdsim.SubsystemBlock, reference_kwargs: Dict[str, Any], controller_kwargs: Dict[str, Any], noise_kwargs: Dict[str, Any], disturbance_kwargs: Dict[str, Any], **kwargs) -> NoReturn:
        """Build the system bloc diagram."""
        with redirect_stdout(open(os.devnull, "w")):
            self._simulation = bdsim.BDSim(animation=False)
            self._diagram = self._simulation.blockdiagram()
            reference = blocks.Reference(**reference_kwargs, name="reference")
            self._diagram.add_block(reference)
            controller = blocks.PID(self._simulation, **controller_kwargs, name="controller")
            self._diagram.add_block(controller)
            limit = kwargs.get("saturation", 0)
            if limit == 0:
                limit = np.inf
            saturation = self._diagram.CLIP(min=-limit, max=limit, name="saturation")
            disturbance = blocks.Disturbance(**disturbance_kwargs, name="disturbance")
            self._diagram.add_block(disturbance)
            disturbed = self._diagram.SUM("++", name="disturbed")
            self._diagram.add_block(plant)
            feedback = self._diagram.SUM("+-", name="feedback")
            noise = blocks.Noise(**noise_kwargs, name="noise")
            measure = self._diagram.SUM("++", name="measure")
            self._diagram.add_block(noise)
            self._diagram.connect(reference, feedback[0])
            self._diagram.connect(feedback, controller)
            self._diagram.connect(controller, saturation)
            self._diagram.connect(saturation, disturbed[0])
            self._diagram.connect(disturbance, disturbed[1])
            self._diagram.connect(disturbed, plant)
            self._diagram.connect(plant, measure[0])
            self._diagram.connect(measure, feedback[1])
            self._diagram.connect(noise, measure[1])
        self._blocks = {
            "reference": reference,
            "feedback": feedback,
            "controller": controller,
            "saturation": saturation,
            "disturbance": disturbance,
            "disturbed": disturbed,
            "noise": noise,
            "measure": measure,
            "plant": plant,
        }
        self._watch = {"r(t)": reference, "e(t)": feedback, "n(t)": noise, "u(t)": saturation, "d(t)": disturbance}

    @property
    def h_dc(self) -> ctrl.TransferFunction:
        """Return the direct chain transfer function of the system."""
        return self._blocks["controller"].h * self._blocks["plant"].h

    @property
    def h_r(self) -> ctrl.TransferFunction:
        """Return the return transfer function of the system."""
        return ctrl.tf(1, 1)

    @property
    def h_ol(self) -> ctrl.TransferFunction:
        """Return the open loop transfer function of the system."""
        return self.h_dc * self.h_r
    
    @property
    def h(self) -> ctrl.TransferFunction:
        """Return the closed loop transfer function of the system."""
        return ctrl.feedback(self.h_dc, self.h_r)
    
    def simulate(self, duration: float = 5.0, dt: float = 0.05, silent: bool = True) -> pd.DataFrame:
        results = self._simulate_from_dynamics(duration, dt, silent)
        # Compute optimal dt for resampling so that all samples are taken
        fixed_dt = np.gcd.reduce((results["t"].to_numpy() * 1e5).astype(np.int64)) / 1e5
        approx_results = self._simulate_from_transfer_function(results, duration, fixed_dt)
        for ak, k in [("y_r(t)", "~y_r(t)"), ("y_n(t)", "~y_n(t)"), ("y_d(t)", "~y_d(t)")]:
            results[k] = interp1d(
                approx_results["t"], approx_results[ak]
            )(results["t"])
        results["~y(t)"] = results["~y_r(t)"] + results["~y_n(t)"] + results["~y_d(t)"]
        return results
    
    def _simulate_from_dynamics(self, duration: float = 5.0, dt: float = 0.05, silent: bool = True) -> pd.DataFrame:
        """Simulate the system using its dynamics."""
        out_stream = open(os.devnull, "w") if silent else sys.stdout
        with redirect_stdout(out_stream):
            self._diagram.compile()
            results = self._simulation.run(
                self._diagram,
                duration,
                dt,
                watch=self._watch.values(),
            )
        data = {"t": results.t.flatten()}
        data.update({k: results.x[:, i].flatten() for i, k in enumerate(results.xnames) if "/" not in k})
        data.update({k: getattr(results, f"y{i}").flatten() for i, k in enumerate(self._watch.keys())})
        return pd.DataFrame(data)
    
    def _simulate_from_transfer_function(self, data: pd.DataFrame, duration: float = 5.0, dt: float = 0.05) -> pd.DataFrame:
        """Simulate the system using its transfer function."""
        time = np.linspace(0, duration, int(duration / dt) + 1)
        reference = np.array([self._blocks["reference"].output(t) for t in time]).flatten()
        _, y = ctrl.forced_response(self.h, time, reference)
        noise = interp1d(np.r_[0, data["t"]], np.r_[self._blocks["noise"].mean, data["n(t)"]])(time)
        _, y_noise = ctrl.forced_response(ctrl.feedback(-self.h_dc, self.h_r, sign=1),time, noise)
        disturbance = interp1d(np.r_[0, data["t"]], np.r_[self._blocks["disturbance"].off, data["d(t)"]])(time)
        _, y_disturbance = ctrl.forced_response(ctrl.feedback(self._blocks["plant"].h, self.h_r * (-self._blocks["controller"].h), sign=1),time, disturbance)
        return pd.DataFrame({"t": time, "y_r(t)": y, "y_n(t)": y_noise, "y_d(t)": y_disturbance})
        
    