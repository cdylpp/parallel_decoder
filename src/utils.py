import stim
import pymatching as pm
import numpy as np
import pandas as pd
import time
from dataclasses import dataclass
from typing import List, Tuple
from tqdm import tqdm
from src.decoders.window_decoder import ParallelDecoder
from src.noise import NoiseModel


@dataclass
class Parameters:
    """
    Experiment parameters
    """
    physical_error_rate:float
    distances:List[int]
    rounds:List[int]
    shots:int
    correlated:bool = False
    mode:str = "barrier"

def make_circuit(distance, rounds, noise, basis='Z'):
    code_task = 'surface_code:rotated_memory_z' if basis == 'Z' else 'surface_code:rotated_memory_x'
       
    circ = stim.Circuit.generated(
        code_task=code_task,
        distance=distance,
        rounds=rounds,        
        after_clifford_depolarization=noise,
        after_reset_flip_probability=noise,
        before_measure_flip_probability=noise,
        before_round_data_depolarization=noise
    )
    return circ


def run_parallel_decoder(
        circuit: stim.Circuit, 
        d: int,
        shots: int,
        mode: str, **kwargs) -> Tuple[list, list]:
    """
    Runs a single parallel decoder on the given circuit.
    
    Returns (ler_global, ler_parallel)
    """
    
    n_buffer, n_commit = int(kwargs.get('n_buffer', d)), int(kwargs.get('n_commit', d))
    
    dem = circuit.detector_error_model(decompose_errors=True)
    matching = pm.Matching.from_detector_error_model(dem)

    sampler = circuit.compile_detector_sampler()
    syndrome, actual_observables = sampler.sample(shots=shots, separate_observables=True)
    
    window_decoder = ParallelDecoder((n_commit, n_buffer), circuit=circuit)
    
    global_preds = matching.decode_batch(syndrome)
    parallel_pred = window_decoder.decode(syndrome, mode=mode)

    ler_global = np.sum(np.any(global_preds != actual_observables, axis=1)) / shots
    ler_parallel = np.sum(np.any(parallel_pred != actual_observables, axis=1)) / shots

    return ler_global,ler_parallel


def run_experiment(parameters: Parameters, **kwargs) -> pd.DataFrame:
    """
    Run parallel decoder experiment given the set of parameters.

    Returns a pandas DataFrame object with the following features:
        - physical_error_rate
        - lattice_size
        - rounds
        - logical_error_rate
        - time
        - type
    """
    p = parameters.physical_error_rate
    D = parameters.distances
    r = parameters.rounds
    shots = parameters.shots
    mode = parameters.mode

    data = []
    columns = ['physical_error_rate','lattice_size','rounds','logical_error_rate','time','type']

    noise_model = NoiseModel(
        after_clifford_depolarization=0.0, 
        after_reset_flip_probability=p, 
        before_measure_flip_probability=p, 
        before_round_data_depolarization=p
    )

    total = len(D) * len(r)
    with tqdm(total=total, desc=f"Simulating...") as pbar:
        for d in D:
            print(f'Running L={d}x{d}...')
            for n in r:
                circuit = stim.Circuit.generated(
                    "surface_code:rotated_memory_z",
                    distance=d,
                    rounds=int(n),
                    **noise_model.get_params()
                )

                start = time.perf_counter()
                ler_global, ler_parallel = run_parallel_decoder(circuit, d, shots, mode, **kwargs)
                elapsed = time.perf_counter() - start

                data.append([p,d,n,ler_global,elapsed,'global'])
                data.append([p,d,n,ler_parallel,elapsed,'parallel'])
                pbar.update(1)
            
    df = pd.DataFrame(data, columns=columns)
    return df