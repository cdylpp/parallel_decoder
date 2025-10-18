from dataclasses import dataclass
from typing import Literal, Tuple, List, Dict
Kind = Literal["A","B"]

@dataclass
class Coord:
    x:float
    y:float
    t:int

@dataclass
class Window:
    kind: Kind
    k: int
    commit: Tuple[int, int]
    span: Tuple[int, int]
    dets: List[int] = None
    commit_dets: List[int] = None
    buffer_dets: List[List[int]] = None
    boundary_dets: Dict[str, List[int]] = None
    @property
    def t0(self) -> int: return self.span[0]
    @property
    def t1(self) -> int: return self.span[1]
    @property
    def c0(self) -> int: return self.commit[0]
    @property
    def c1(self) -> int: return self.commit[1]