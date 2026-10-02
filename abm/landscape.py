"""Who owns which paddy: farmer holdings scattered across the district."""
import numpy as np


def assign_owners_along_canal(order: np.ndarray, n: int, paddies_per_farmer: int) -> np.ndarray:
    """Chunk paddies in canal-sorted order into equal-size farmer holdings."""
    owners = np.zeros(n, dtype=int)
    if paddies_per_farmer <= 0 or paddies_per_farmer >= n:
        return owners
    farmer = 0
    for start in range(0, n, paddies_per_farmer):
        chunk = order[start : start + paddies_per_farmer]
        owners[chunk] = farmer
        farmer += 1
    return owners


def assign_owners_along_canal_variable(
    order: np.ndarray,
    n: int,
    avg_paddies_per_farmer: int,
    variability: float,
    seed: int,
) -> np.ndarray:
    """Shuffle parcels reproducibly, then assign holdings whose *size* varies
    around ``avg_paddies_per_farmer`` — every paddy is still owned by exactly one
    farmer, scattered across the district, but holdings are no longer all equal.

    ``variability`` 0 is the fixed/uniform case (every holding has the requested size); 1 draws
    each holding log-uniformly between 0.5x and 2x the average — symmetric in
    log space, so a farm is as likely to be twice the average as half of it, and
    ``avg_paddies_per_farmer`` stays the true median holding size (not skewed up
    by the longer tail toward 2x, the way a plain linear draw would be). The last
    holding is clipped to whatever remains, so the total always covers every
    paddy exactly.
    """
    order = np.random.default_rng(int(seed)).permutation(np.asarray(order))
    owners = np.zeros(n, dtype=int)
    if avg_paddies_per_farmer <= 0 or avg_paddies_per_farmer >= n:
        return owners
    if variability <= 0:
        return assign_owners_along_canal(order, n, avg_paddies_per_farmer)

    log_half_range = min(float(variability), 1.0) * np.log(2.0)  # 1.0 -> exactly [0.5x, 2x]
    rng = np.random.default_rng(int(seed))
    farmer = 0
    start = 0
    while start < n:
        multiplier = float(np.exp(rng.uniform(-log_half_range, log_half_range)))
        size = int(round(multiplier * avg_paddies_per_farmer))
        size = max(1, min(size, n - start))
        owners[order[start : start + size]] = farmer
        farmer += 1
        start += size
    return owners
