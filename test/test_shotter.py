from src.shotter import Shotter


if __name__ == "__main__":
    shotter = Shotter(api_keys={"ionq":"nH92KkP3apE4lRfUimfzIek4yIUFK3Eo"})
    circuit = """OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    h q[0];
    cx q[0],q[1];
    measure q -> c;
    """
    shots = 1024
    backends = {
        "ionq": ["simulator", ("simulator",{"noise":{"model" : "aria-1", "seed": 42}, "name": "aria-1"})],
    }
    policy = "uniform"
    blob = None
    multiprocess = False
    single_thread = True
    
    results, parsed_single, blob = shotter.run(
        circuit=circuit,
        shots=shots,
        backends=backends,
        policy=policy,
        blob=blob,
        multiprocess=multiprocess,
        single_thread=single_thread
    )
    print("Results:", results)
    print("Parsed Single Results:", parsed_single)
    print("Blob:", blob)