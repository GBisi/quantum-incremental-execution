from typing import Any

def split(
    circuit: Any,
    shots: int,
    backends: dict[str, dict[str, tuple[str, str]]],
    blob: Any,
    seed: int | None = None,
) -> tuple[dict[str, dict[str, list[tuple[Any, int]]]], Any]:
    """
    Runs the same circuit and shots across all backends.
    """
    dispatch = {}
    for provider_name, bk_data in backends.items():
        dispatch[provider_name] = {}
        for backend_key in bk_data:
            dispatch[provider_name][backend_key] = [(circuit, shots)]
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