from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from qiskit.transpiler import PassManager
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.options import SamplerOptions, SimulatorOptions

from qbraid.programs import load_program
from qbraid.runtime.device import QuantumDevice
from qbraid.runtime.enums import DeviceStatus
from qbraid.runtime.options import RuntimeOptions

from qiskit_aer import AerSimulator  # type: ignore

from qbraid.runtime.ibm import QiskitJob

if TYPE_CHECKING:
    import qiskit

    import qbraid.runtime.ibm



class LocalAERBackend(QuantumDevice):
    """Wrapper class for Local AerSimulator ``Backend`` objects."""

    def __init__(
        self,
        profile: qbraid.runtime.TargetProfile,
        backend: Optional[AerSimulator] = None,
        seed: Optional[int] = None,
    ):
        options = RuntimeOptions(pass_manager=None)
        options.set_validator("pass_manager", lambda x: x is None or isinstance(x, PassManager))

        super().__init__(profile=profile, options=options)
        
        self._backend = backend or AerSimulator()
        self._seed = seed

    def __str__(self):
        """String representation of the QiskitBackend object."""
        return f"{self.__class__.__name__}('{self._backend.name}')"

    def status(self):
        """Return the status of this Device.

        Returns:
            str: The status of this Device
        """
        return DeviceStatus.ONLINE


    def transform(self, run_input: qiskit.QuantumCircuit) -> qiskit.QuantumCircuit:
        """Transpile a circuit for the device."""
        program = load_program(run_input)
        program.transform(self)
        return program.program

    def submit(
        self,
        run_input: qiskit.QuantumCircuit | list[qiskit.QuantumCircuit],
        shots: int | None = None,
        **kwargs,
    ) -> qbraid.runtime.ibm.QiskitJob:
        """Runs circuit(s) on qiskit backend via :meth:`~SamplerV2.run`.

        Uses the :meth:`SamplerV2.execute` method to create a
        :class:`~qbraid.runtime.ibm.QiskitJob`, and return the result.

        Args:
            run_input: A circuit object to run on the IBM device.
            shots (int, optional): The number of times to run the task on the device. If None,
                number of shots is determined by the sampler.

        Keyword Args:
            shots (int, optional): The number of times to run the task on the device. If None,
                number of shots is determined by the sampler.

        Returns:
            qbraid.runtime.ibm.QiskitJob: The job like object for the run.

        """
        _seed = None
        if "noise" in kwargs and "seed" in kwargs["noise"]:
            _seed = kwargs["noise"]["seed"]
        elif self._seed is not None:
            _seed = self._seed
        
        backend = self._backend
        
        if _seed is not None:
            options = SamplerOptions(
                simulator=SimulatorOptions(seed_simulator=_seed),
            )
        else:
            options = SamplerOptions()
        
        sampler = Sampler(mode=backend, options=options)
        pubs = run_input if isinstance(run_input, list) else [run_input]
        job = sampler.run(pubs, shots=shots)
        return QiskitJob(job.job_id(), job=job, device=self)