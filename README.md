# Parallel Window Decoder

A parallel window decoder integrated with `stim` and `pymatching`.

## How to use

```python
import stim
import pymatching as pm
import numpy as np
from decoders.window_decoder import ParallelDecoder

# Parameters
d = 15
T = 100
shots = 1_000

# Create a stim circuit
circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_z",
    distance=d,
    rounds=T,
    after_clifford_depolarization=0.0, 
    after_reset_flip_probability=0.02, 
    before_measure_flip_probability=0.02, 
    before_round_data_depolarization=0.02
)

# Generate dem and syndromes 
dem = circuit.detector_error_model(decompose_errors=True)
matching = pm.Matching.from_detector_error_model(dem)
syndrome, actual_observables = circuit.compile_detector_sampler().sample(shots,seperate_observables=True)

# Create the parallel decoder
buffer_size = commit_size = d
window_decoder = ParallelDecoder(
    (commit_size,buffer_size),
    circuit=circuit
)

# decode from sampled syndromes
parallel_pred = window_decoder.decode(syndrome=syndrome)

# decode from global
global_preds = matching.decode_batch(syndrome)


ler_global = np.sum(np.any(global_preds != actual_observables, axis=1)) / shots
ler_parallel = np.sum(np.any(parallel_pred != actual_observables, axis=1)) / shots

```
