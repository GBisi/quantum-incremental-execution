from typing import Any

def split(
    circuit: Any,
    shots: int,
    backends: dict[str, dict[str, tuple[str, str]]],
    blob: Any,
    seed: int | None = None,
) -> tuple[dict[str, dict[str, list[tuple[Any, int]]]], Any]:
    """
    Uniformly split the same circuit and shots across all backends.
    Instead of storing the backend object in `backends`, we store (backend_name, config_json).
    """
    num_backends = 0
    for provider_name, bk_data in backends.items():
        num_backends += len(bk_data)
    if num_backends == 0:
        raise ValueError("No backends provided.")
    dispatch = {}
    total = 0
    for provider_name, bk_data in backends.items():
        dispatch[provider_name] = {}
        for backend_key in bk_data:
            dispatch[provider_name][backend_key] = [(circuit, shots//num_backends)]
            total += shots//num_backends
    if total < shots:
        for provider_name, bk_data in backends.items():
            for backend_key in bk_data:
                if total < shots:
                    if dispatch[provider_name][backend_key]:
                        old_shots = dispatch[provider_name][backend_key][0][1]
                        dispatch[provider_name][backend_key].pop()
                        dispatch[provider_name][backend_key] = [(circuit, old_shots + 1)]
                        total += 1
                else:
                    break
    if total > shots:
        for provider_name, bk_data in backends.items():
            for backend_key in bk_data:
                if total > shots:
                    if dispatch[provider_name][backend_key]:
                        old_shots = dispatch[provider_name][backend_key][0][1]
                        dispatch[provider_name][backend_key].pop()
                        dispatch[provider_name][backend_key] = [(circuit, old_shots - 1)]
                        total -= 1
                else:
                    break
    return dispatch, blob


def merge(
    results: dict,
    blob: Any,
    seed: int | None = None,
) -> tuple[dict, Any]:
    """
    Merges results by summing bitstring counts.
    """
    merged_results = {}
    for provider_name, provider_results in results.items():
        for backend_key, backend_results in provider_results.items():
            for result_dict in backend_results:
                for bitstring, count in result_dict.items():
                    merged_results[bitstring] = merged_results.get(bitstring, 0) + count
    return merged_results, blob