import stim
import pymatching as pm
import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class Coord:
    x:float
    y:float
    t:int


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

def det_to_coords(circuit: stim.Circuit):
    coords = circuit.get_detector_coordinates()
    det_coords = {d : Coord(coord[0], coord[1], coord[2]) for d, coord in coords.items()}
    return det_coords