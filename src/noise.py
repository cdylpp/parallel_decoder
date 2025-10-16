from dataclasses import dataclass

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