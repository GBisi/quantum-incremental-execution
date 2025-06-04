import os
import importlib.util
import json
import logging
import hashlib
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Callable, Optional

from qbraid.runtime import DeviceStatus, IonQProvider, QiskitRuntimeProvider
from src.shotter.local_aer_provider.provider import LocalAERProvider
from qbraid import transpile

PROVIDERS = {
    "local": LocalAERProvider,
    "ionq": IonQProvider,
}

# Non utilizziamo più una variabile globale per i provider
# GLOBAL_PROVIDERS = None

# Set default logging level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

##################################################
# GLOBAL PROVIDER INITIALIZATION (Local per worker)
##################################################

def init_global_providers(api_keys: Optional[dict[str, str]] = None, raise_exc: bool = True) -> dict:
    """
    Inizializza i provider in maniera locale.
    """
    
    if api_keys is None:
        api_keys = {}
    providers = {}
    
    for provider_name, provider_cls in PROVIDERS.items():
        try:
            if provider_name in api_keys:
                providers[provider_name] = provider_cls(api_keys[provider_name])
            else:
                providers[provider_name] = provider_cls()
        except Exception as e:
            if raise_exc:
                logging.error(f"Failed to initialize {provider_name} provider: {e}")
                raise ValueError(f"Failed to initialize {provider_name} provider") from e
            
    return providers

##################################################
# WORKER FUNCTION (Local initialization per worker)
##################################################

def run_single_job_static(
    provider_name: str,
    backend_name: str,
    circuit: Any,
    shots: int,
    config: str,
    api_keys: dict[str, str],
    raise_exc: bool = True,
    seed: Optional[int] = None
) -> dict:
    """
    Worker function to run a single job.
    Initializes providers locally and clears the global alias registration to avoid
    conflicts due to a library-maintained global state.
    """
        
    local_providers = init_global_providers(api_keys=api_keys, raise_exc=raise_exc)
    if provider_name not in local_providers:
        raise ValueError("Provider not available")
    provider = local_providers[provider_name]
    
    backend = provider.get_device(backend_name)
    qc = circuit
    if provider_name.lower() == "ionq":
        qc = transpile(circuit, "qiskit").remove_final_measurements(inplace=False)
    
    conf_dict = Shotter.json_to_config(config)
    
    if seed is not None:
        if "noise" not in conf_dict:
            conf_dict["noise"] = {}
        conf_dict["noise"]["seed"] = seed
    
    job = backend.run(qc, shots=shots, **conf_dict)
    result = job.result()
    return result.data.get_counts()

##################################################
# HELPER FUNCTIONS
##################################################

def load_policies_from_folder(folder_path: str) -> dict:
    """
    Carica dinamicamente le policy dai file Python presenti in una cartella specificata.
    """
    if not os.path.exists(folder_path):
        logging.warning(f"Folder {folder_path} does not exist.")
        os.makedirs(folder_path)
        return {}
    
    policies = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".py"):
            policy_name = file_name[:-3]
            file_path = os.path.join(folder_path, file_name)

            spec = importlib.util.spec_from_file_location(policy_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if hasattr(module, "split") and hasattr(module, "merge"):
                policies[policy_name] = {
                    "split": module.split,
                    "merge": module.merge,
                }
            else:
                logging.warning(f"Policy file {file_name} does not define 'split' and 'merge' functions.")
    return policies

##################################################
# MAIN SHOTTER CLASS
##################################################

class Shotter:
    """
    La classe Shotter orchestra l'esecuzione di circuiti quantistici
    su diversi provider e backend, con supporto per policy di split e merge configurabili.
    """
    
    SUPPORTED_PROVIDERS = list(PROVIDERS.keys())
    
    def __init__(self, api_keys: Optional[dict[str, str]] = None, max_workers: Optional[int] = None, raise_exc: bool = False, seed: Optional[int] = None):
        """
        Inizializza la classe Shotter.
        
        :param api_keys: Dizionario {provider_name: api_key, ...}
        :param max_workers: Numero massimo di processi/thread worker.
        """
        # In questo caso non utilizziamo una variabile globale per i provider:
        self._api_keys = api_keys if api_keys is not None else {}
        self._providers = init_global_providers(api_keys=api_keys, raise_exc=raise_exc)
        self._max_workers = max_workers
        self._policies = load_policies_from_folder("policies")
        self._raise_exc = raise_exc
        self._seed = seed

    def _init_backends_for_provider(
        self,
        provider_name: str,
        provider: Any
    ) -> tuple[str, dict[str, Any]]:
        """
        Recupera i dispositivi (backend) da un determinato provider.
        """
        try:
            raw_backends = provider.get_devices()
            backends = {}
            for backend in raw_backends:
                device_id = backend.metadata().get("device_id")
                if device_id:
                    backends[device_id] = backend
            logging.info(f"Retrieved {len(backends)} backends from provider '{provider_name}'.")
        except Exception as e:
            logging.error(f"Failed to get devices from provider {provider_name}: {e}")
            raise
        return provider_name, backends

    def get_backends(self, online: bool = True) -> dict[str, dict[str, Any]]:
        """
        Ritorna un dizionario {provider_name: {backend_name: backend_obj, ...}, ...}.
        """
        results_backends = {}
        # Recupera i backend in sequenza per evitare problemi di pickling
        for pn, pobj in self._providers.items():
            name, backends_dict = self._init_backends_for_provider(pn, pobj)
            results_backends[name] = backends_dict

        if online:
            filtered_backends = {}
            for name, backends_dict in results_backends.items():
                online_backends = {}
                for backend_name, backend_obj in backends_dict.items():
                    status = backend_obj.status()
                    if status == DeviceStatus.ONLINE:
                        online_backends[backend_name] = backend_obj
                    else:
                        logging.debug(f"Skipping offline backend {backend_name} for provider {name}. Status: {status}")
                filtered_backends[name] = online_backends
            return filtered_backends
        else:
            return results_backends

    def get_backend(
        self,
        provider_name: str,
        backend_name: str,
        online: bool = True
    ) -> Any:
        """
        Recupera un singolo backend da un provider dato.
        """
        try:
            provider = self._providers[provider_name]
            backend = provider.get_device(backend_name)
            if online and backend.status() != DeviceStatus.ONLINE:
                raise RuntimeError(f"Backend {backend_name} is offline.")
            return backend
        except KeyError:
            logging.error(f"Provider '{provider_name}' not found.")
            raise ValueError(f"Provider '{provider_name}' is not initialized.")
        except Exception as e:
            logging.error(f"Failed to get backend {backend_name} from provider {provider_name}: {e}")
            raise

    @staticmethod
    def config_to_json(config: dict) -> str:
        """
        Converte un dizionario di configurazione in una stringa JSON.
        """
        return "{}" if not config else json.dumps(config, sort_keys=True)

    @staticmethod
    def json_to_config(config_json: str) -> dict:
        """
        Converte una stringa JSON in un dizionario.
        """
        return {} if not config_json or config_json == "{}" else json.loads(config_json)

    def _generate_config_key(self, backend_name: str, config_json: str) -> str:
        """
        Genera una chiave unica e stabile per la coppia (backend_name, config_json).
        """
        stable_hash = hashlib.md5(config_json.encode()).hexdigest()
        return f"{backend_name}${stable_hash}"

    def run(
        self,
        circuit: Any,
        shots: int,
        backends: dict[str, list[tuple[str, dict]]],
        policy: Optional[str] = None,
        blob: Any = None,
        multiprocess: bool = False,
        single_thread: bool = True,
        seed: Optional[int] = None
    ) -> tuple[dict, dict, Any]:
        """
        Entry point principale per eseguire un circuito sui backend specificati o scoperti.
        """
        backends = deepcopy(backends)
        
        # Risolvi i nomi dei backend e congela la configurazione (non si memorizza l'oggetto backend)
        resolved_backends: dict[str, dict[str, tuple[str, str]]] = {}
        for provider_name, backend_specs in backends.items():
            resolved_backends[provider_name] = {}
            for spec in backend_specs:
                if isinstance(spec, str):
                    backend_name = spec
                    config_dict = {}
                else:
                    backend_name, config_dict = spec

                if config_dict is None or config_dict == {}:
                    config_key = backend_name
                else:
                    if "name" in config_dict:
                        config_key = config_dict.pop("name")
                    else:
                        config_json = self.config_to_json(config_dict)
                        config_key = self._generate_config_key(backend_name, config_json)
                
                config_json = self.config_to_json(config_dict)
                resolved_backends[provider_name][config_key] = (backend_name, config_json)
        
        # Split
        split_func = self.get_split_policy(policy)
        dispatch, blob = split_func(circuit, shots, resolved_backends, blob, seed)
        
        # Dispatch usando la modalità scelta:
        single_results = self._dispatch(
            dispatch,
            resolved_backends,
            multiprocess=multiprocess,
            single_thread=single_thread,
            seed=seed if seed is not None else self._seed
        )
        
        # Merge
        merge_func = self.get_merge_policy(policy)
        results, blob = merge_func(single_results, blob, seed)
        
        # Parsing dei risultati
        parsed_single = self._parse_single_results(single_results, resolved_backends)
        
        return results, parsed_single, blob

    def _dispatch(
        self,
        dispatch_dict: dict[str, dict[str, list[tuple[Any, int]]]],
        resolved_backends: dict[str, dict[str, tuple[str, str]]],
        multiprocess: bool = False,
        single_thread: bool = True,
        seed: Optional[int] = None
    ) -> dict:
        """
        Inoltra i job verso i provider/backend utilizzando:
          - Esecuzione diretta sul main thread se single_thread è True.
          - ProcessPoolExecutor se multiprocess è True.
          - Altrimenti, ThreadPoolExecutor.
        """
        all_results = {}
        
        # Modalità esecuzione diretta
        if single_thread:
            for provider_name, bk_data in dispatch_dict.items():
                all_results[provider_name] = {}
                for backend_key, job_list in bk_data.items():
                    all_results[provider_name][backend_key] = []
                    for circuit_job, shots_job in job_list:
                        (bname, config_str) = resolved_backends[provider_name][backend_key]
                        res = run_single_job_static(provider_name, bname, circuit_job, shots_job, config_str, self._api_keys, self._raise_exc, seed)
                        all_results[provider_name][backend_key].append(res)
            return all_results

        # Seleziona il tipo di executor in base alla flag multiprocess
        ExecutorClass = ProcessPoolExecutor if multiprocess else ThreadPoolExecutor
        
        if ExecutorClass == ThreadPoolExecutor:
            logging.error("ThreadPoolExecutor not working with qbraid. Using multiprocess=True.")
            ExecutorClass = ProcessPoolExecutor

        with ExecutorClass(max_workers=self._max_workers) as executor:
            futures = {}
            for provider_name, bk_data in dispatch_dict.items():
                all_results[provider_name] = {}
                for backend_key, job_list in bk_data.items():
                    all_results[provider_name][backend_key] = []
                    for circuit_job, shots_job in job_list:
                        (bname, config_str) = resolved_backends[provider_name][backend_key]
                        future = executor.submit(
                            run_single_job_static,
                            provider_name,
                            bname,
                            circuit_job,
                            shots_job,
                            config_str,
                            self._api_keys,
                            self._raise_exc,
                            seed
                        )
                        futures[future] = (provider_name, backend_key)

            for future in as_completed(futures):
                provider_name, backend_key = futures[future]
                try:
                    res = future.result()
                    all_results[provider_name][backend_key].append(res)
                except Exception as e:
                    logging.error(f"Exception while running job on {provider_name}/{backend_key}: {e}")
                    raise e
        return all_results

    def _parse_single_results(
        self,
        single_results: dict,
        resolved_backends: dict[str, dict[str, tuple[str, str]]]
    ) -> dict:
        """
        Converte il dizionario annidato dei risultati in un formato più leggibile.
        """
        parsed = {}
        for provider_name, bk_data in single_results.items():
            parsed[provider_name] = {}
            for backend_key, list_of_counts in bk_data.items():
                (backend_name, config_json) = resolved_backends[provider_name][backend_key]
                config_dict = self.json_to_config(config_json)
                parsed[provider_name][backend_key] = []
                for result_dict in list_of_counts:
                    parsed[provider_name][backend_key].append({
                        "result": result_dict,
                        "configuration": config_dict
                    })
        return parsed

    def add_policy(
        self,
        name: str,
        split_policy: Callable,
        merge_policy: Callable
    ) -> None:
        """
        Aggiunge una nuova policy al dizionario delle policy.
        """
        self._policies[name] = {"split": split_policy, "merge": merge_policy}
    
    def add_policy_from_file(self, file_path: str) -> None:
        """
        Aggiunge una nuova policy da un file Python.
        """
        spec = importlib.util.spec_from_file_location("policy_module", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        policy_name = os.path.basename(file_path)[:-3]
        
        if hasattr(module, "split") and hasattr(module, "merge"):
            self.add_policy(policy_name, module.split, module.merge)
            folder = "policies"
            if not os.path.exists(folder):
                os.makedirs(folder)
            new_path = os.path.join(folder, f"{policy_name}.py")
            with open(new_path, "w") as f:
                f.write(module.__doc__ if module.__doc__ else "")
                f.write("\n\n")
                with open(file_path, "r") as pf:
                    f.write(pf.read())
            logging.info(f"Policy '{policy_name}' added successfully.")
            load_policies_from_folder(folder)
        else:
            logging.warning(f"Policy file {file_path} does not define 'split' and 'merge' functions.")
    
    def get_split_policy(self, name: Optional[str]) -> Callable:
        """
        Recupera una split policy per nome, oppure quella di default 'uniform' se non specificata.
        """
        if not name:
            name = "uniform"
        try:
            return self._policies[name]["split"]
        except KeyError:
            logging.error(f"Split policy '{name}' not found.")
            raise ValueError(f"Split policy '{name}' not found.")

    def get_merge_policy(self, name: Optional[str]) -> Callable:
        """
        Recupera una merge policy per nome, oppure quella di default 'uniform' se non specificata.
        """
        if not name:
            name = "uniform"
        try:
            return self._policies[name]["merge"]
        except KeyError:
            logging.error(f"Merge policy '{name}' not found.")
            raise ValueError(f"Merge policy '{name}' not found.")

    @property
    def backends(self) -> dict[str, list[str]]:
        """
        Proprietà di comodo per elencare tutti i backend online dei provider.
        """
        online_backends = self.get_backends(online=True)
        output = {}
        for provider_name, backends_dict in online_backends.items():
            output[provider_name] = list(backends_dict.keys())
        return output
    
    @property
    def providers(self) -> list[str]:
        """Elenca tutti i provider inizializzati."""
        return list(self._providers.keys())
    
    @property
    def policies(self) -> list[str]:
        """Elenca tutte le policy disponibili."""
        return list(self._policies.keys())
    