from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import qiskit

from qbraid._caching import cached_method
from qbraid.programs import ProgramSpec
from qbraid.runtime.profile import TargetProfile
from qbraid.runtime.provider import QuantumProvider

from .device import LocalAERBackend

from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2
from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error, phase_damping_error  # type: ignore

if TYPE_CHECKING:

    import qbraid.runtime.ibm
    
    
def create_noisy_simulator(
    noise_strength: float) -> AerSimulator:
    """Create a noisy AerSimulator backend."""
        # Define errors for different noise processes
    depol_error = depolarizing_error(noise_strength, 1)
    amp_damp_error = amplitude_damping_error(noise_strength)
    phase_damp_error = phase_damping_error(noise_strength)

    composite_error = depol_error.compose(amp_damp_error)
    composite_error = composite_error.compose(phase_damp_error)

    # Build the noise model
    noise_model = NoiseModel()
    # Add a combination of errors to single-qubit gates (for illustration, applying them sequentially)
    for gate in ['u1', 'u2', 'u3']:
        noise_model.add_all_qubit_quantum_error(composite_error, [gate])
        
    # Instantiate the simulator with the noise model
    simulator = AerSimulator(noise_model=noise_model)
    
    return simulator


class LocalAERProvider(QuantumProvider):

    def _build_runtime_profile(
        self, backend: AerSimulator, program_spec: Optional[ProgramSpec] = None
    ) -> TargetProfile:
        """Builds a runtime profile from a backend."""
        program_spec = program_spec or ProgramSpec(qiskit.QuantumCircuit)
        config = backend.configuration()

        return TargetProfile(
            device_id=backend.name,
            simulator=True,
            num_qubits=config.n_qubits,
            program_spec=program_spec,
            provider_name="LocalAER",
        )

    @cached_method
    def get_devices(self, **kwargs) -> list[qbraid.runtime.ibm.QiskitBackend]:
        """Returns the IBM Quantum provider backends."""

        backends = [AerSimulator()] + [b for b in FakeProviderForBackendV2().backends()]
        program_spec = ProgramSpec(qiskit.QuantumCircuit)

        return [
            LocalAERBackend(
                profile=self._build_runtime_profile(backend, program_spec=program_spec),
            )
            for backend in backends
        ]

    @cached_method
    def get_device(
        self, device_id: str,
    ) -> qbraid.runtime.ibm.QiskitBackend:
        """Returns the local backend."""
        
        if device_id == "aer_simulator":
            backend = AerSimulator()
        elif device_id.startswith("aer_simulator_noisy"):        
            noise_strength = int(device_id.split("-")[-1])
            noise_strength = noise_strength / 100.0  # Convert to a fraction
            backend = create_noisy_simulator(noise_strength)
        else:
            try:
                backend = FakeProviderForBackendV2().backend(device_id)
            except:
                raise ValueError(f"Device '{device_id}' not found in local AerSimulator backends.")
            
        program_spec = ProgramSpec(qiskit.QuantumCircuit)
        return LocalAERBackend(
            profile=self._build_runtime_profile(backend,
            program_spec=program_spec),
            backend=backend,
        )
        
    def __hash__(self):
        if not hasattr(self, "_hash"):
            object.__setattr__(self, "_hash", hash(str(self)))
        return self._hash  # pylint: disable=no-member