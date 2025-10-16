import numpy as np
import stim
import pymatching as pm
import pandas as pd
import time
from dataclasses import dataclass
from src.utils import det_to_coords
from parallel_decoder.src.decoders.window_decoder import ParallelWindowScheduler, ParallelDecoder
from src.plots import plot


@dataclass(frozen=True)
class NoiseModel:
    after_clifford_depolarization: float
    after_reset_flip_probability: float 
    before_round_data_depolarization: float
    before_measure_flip_probability: float

    def get_params(self) -> dict:
        return {
            'after_clifford_depolarization':self.after_clifford_depolarization,
            'after_reset_flip_probability':self.after_reset_flip_probability,
            'before_round_data_depolarization':self.before_round_data_depolarization,
            'before_measure_flip_probability':self.before_measure_flip_probability
        }


def run_parallel_decoder(circuit: stim.Circuit, dem: stim.DetectorErrorModel, syndrome, actual_observables, matching, d, shots):
    
    d2c = det_to_coords(circuit)
    n_buffer = n_commit = d
    sched = ParallelWindowScheduler(100, n_commit, n_buffer, d2c)
    pwindecoder = ParallelDecoder(sched, dem, matching, d2c, syndrome)
    
    pwindecoder.run()
    
    global_preds = matching.decode_batch(syndrome)
    parallel_pred = matching.decode_batch(pwindecoder.Z_global)

    ler_global = np.sum(np.any(global_preds != actual_observables, axis=1)) / shots
    ler_parallel = np.sum(np.any(parallel_pred != actual_observables, axis=1)) / shots

    return ler_global,ler_parallel

if __name__ == "__main__":

    physical_error_rate = 0.02 # physical error rate of 2%
    distances = [5, 7, 9, 11, 13, 15, 17]
    rounds = np.linspace(10,225,10)
    shots=2_000 
    correlated = False

    data = []
    columns = ['physical_error_rate','lattice_size','rounds','logical_error_rate','time','type']

    noise_model = NoiseModel(
        after_clifford_depolarization=0.0, 
        after_reset_flip_probability=physical_error_rate, 
        before_measure_flip_probability=physical_error_rate, 
        before_round_data_depolarization=physical_error_rate
    )


    for d in distances:
        print(f"Simulating L={d}x{d}...")
        for n in rounds:
            circuit = stim.Circuit.generated(
                "surface_code:rotated_memory_z",
                distance=d,
                rounds=int(n),
                # phenomological Pauli noise
                **noise_model.get_params()
            )

            dem = circuit.detector_error_model(decompose_errors=True)
            matching = pm.Matching.from_detector_error_model(dem)
            sampler = circuit.compile_detector_sampler()
            syndrome, actual_observables = sampler.sample(shots=shots, separate_observables=True)
            
            start = time.perf_counter()
            ler_global, ler_parallel = run_parallel_decoder(circuit, dem, np.copy(syndrome), actual_observables, matching, d, shots)
            elapsed = time.perf_counter() - start

            data.append([physical_error_rate,d,n,ler_global,elapsed,'global'])
            data.append([physical_error_rate,d,n,ler_parallel,elapsed,'parallel'])
            
    df = pd.DataFrame(data, columns=columns)

    plot(df, shots)


