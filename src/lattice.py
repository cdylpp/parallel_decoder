from __future__ import annotations

from typing import Dict, List, Tuple, Any

from dataclasses import dataclass
import stim
import networkx as nx
import numpy as np
import scipy.sparse as sp
import uuid

@dataclass
class Coord:
    """
    3D coord (x,y,t)
    """
    x:float
    y:float
    t:int

class Node:
    """
    Decoding graph node
    """
    def __init__(self, pos: Coord, index, node_type: str | None = None, defect: bool = False):
        self.node_type = node_type
        self.pos = pos
        self.index = index
        self._id = uuid.uuid4()
        self.defect = defect
    
    def __eq__(self, node: Node):
        return self._id == node._id
    
    def __repr__(self):
        return f"Node(idx={self.index}, type={self.node_type}, {self.pos})"


class Lattice:
    """Square L × L-node lattice with edges as qubits and nodes as stabilizers."""

    def __init__(self, L: int) -> None:
        if not isinstance(L, int):
            raise TypeError("L must be an integer.")
        if L <= 0:
            raise ValueError("L must be positive.")
        self._L = L
        self._check_matrix: sp.csr_matrix | None = None
        self._node_coords_cache: dict[int, Tuple[int, int]] | None = None
    
    @classmethod
    def from_dem(self, dem: stim.DetectorErrorModel) -> nx.Graph: 

        d2c = dem.get_detector_coordinates()

        xs, ys = [], []
        for coord in d2c.values():
            xs.append(coord[0])
            ys.append(coord[1])
        x_min, x_max = min(xs), max(xs)
        x_pad = 0.5 * max(1.0, (x_max - x_min) / 20.0)

        self.G = nx.Graph()
        # Add detectors
        for d, (x,y,t) in d2c.items():
            self.G.add_node(d, pos=(x,y), t=t, kind="detector", defect=False)

        left_rail, right_rail = {}, {}
        def _nearest_side(x):
            return "left" if (x - x_min) <= (x_max - x) else "right"
        
        
        # for each instruction
        for ins in dem:
            
            if ins.type != "error": # look for error types
                continue
            
            dets = []
            # for each target
            for tgt in ins.targets_copy():
                # if tgt is a detector
                if tgt.is_relative_detector_id():
                    # add to the list
                    dets.append(tgt.val)
            
            # short circuit
            if len(dets) == 0:
                continue

            # attach to a boundary
            if len(dets) == 1:
                d = dets[0]
                (x,y,t) = d2c[d]
                side = _nearest_side(x)
                if side == "left":
                    y_key = round(y,6)
                    if y_key not in left_rail:
                        node = ("B", "L", y_key, len(left_rail))
                        self.G.add_node(node, pos=(x_min - x_pad, y), kind="boundary", defect=False)
                        left_rail[y_key] = node
                    b = left_rail[y_key]
                else:
                    # side is right
                    y_key = round(y, 6)
                    if y_key not in right_rail:
                        node = ("B", "R", y_key, len(right_rail))
                        self.G.add_node(node, pos=(x_max + x_pad, y), kind="boundary", defect=False)
                        right_rail[y_key] = node
                    b = right_rail[y_key]
                self.G.add_edge(d, b, source="boundary")
                continue

            if len(dets) == 2:
                u, v = dets
                if u != v:
                    self.G.add_edge(u, v, source="dem")
        
        return self.G

    @property
    def size(self) -> int:
        """
        Lattice size
        """
        return self._L

    @property
    def num_nodes(self) -> int:
        """
        Number of nodes.
        """
        return self._L * self._L

    @property
    def num_edges(self) -> int:
        """
        Number of edges.
        """
        return 2 * self._L * (self._L - 1) if self._L > 1 else 0

    def node_index(self, i: int, j: int) -> int:
        """
        Return the node index of the (i,j)-th coordinate.
        """
        self._validate_coords(i, j)
        return j * self._L + i

    def node_coord(self, nid: int) -> Tuple[int, int]:
        """Return the (x, y) coordinate for node nid with origin at top-left."""
        self._validate_node_index(nid)
        i = nid % self._L
        j = nid // self._L
        return i, j

    @property
    def node_coords(self) -> dict[int, Tuple[int, int]]:
        """Dictionary mapping node indices to (x, y) coordinates (top-left origin)."""
        if self._node_coords_cache is None:
            self._node_coords_cache = {
                nid: self.node_coord(nid) for nid in range(self.num_nodes)
            }
        return self._node_coords_cache

    def edge_index_h(self, i: int, j: int) -> int:
        if self._L < 2:
            raise ValueError("Horizontal edges undefined for L < 2.")
        if not (0 <= i < self._L - 1 and 0 <= j < self._L):
            raise ValueError("Horizontal edge coordinates out of bounds.")
        return j * (self._L - 1) + i

    def edge_index_v(self, i: int, j: int) -> int:
        if self._L < 2:
            raise ValueError("Vertical edges undefined for L < 2.")
        if not (0 <= i < self._L and 0 <= j < self._L - 1):
            raise ValueError("Vertical edge coordinates out of bounds.")
        horizontal_block = self._L * (self._L - 1)
        return horizontal_block + j * self._L + i

    def edge_endpoints(self, eid: int) -> Tuple[int, int]:
        self._validate_edge_index(eid)
        if self._L < 2:
            raise ValueError("Edges undefined for L < 2.")
        horizontal_block = self._L * (self._L - 1)
        if eid < horizontal_block:
            row, offset = divmod(eid, self._L - 1)
            nid1 = self.node_index(offset, row)
            nid2 = self.node_index(offset + 1, row)
            return nid1, nid2
        offset = eid - horizontal_block
        row, col = divmod(offset, self._L)
        nid1 = self.node_index(col, row)
        nid2 = self.node_index(col, row + 1)
        return nid1, nid2

    def neighbors_of_node(self, nid: int) -> List[int]:
        i, j = self.node_coord(nid)
        neighbors: List[int] = []
        if self._L > 1 and i > 0:
            neighbors.append(self.edge_index_h(i - 1, j))
        if self._L > 1 and i < self._L - 1:
            neighbors.append(self.edge_index_h(i, j))
        if self._L > 1 and j > 0:
            neighbors.append(self.edge_index_v(i, j - 1))
        if self._L > 1 and j < self._L - 1:
            neighbors.append(self.edge_index_v(i, j))
        return neighbors

    def to_check_matrix(self) -> sp.csr_matrix:
        if self._check_matrix is not None:
            return self._check_matrix

        num_edges = self.num_edges
        if num_edges == 0:
            self._check_matrix = sp.csr_matrix((self.num_nodes, 0), dtype=np.uint8)
            return self._check_matrix

        rows = np.empty(2 * num_edges, dtype=np.int32)
        cols = np.empty(2 * num_edges, dtype=np.int32)
        data = np.ones(2 * num_edges, dtype=np.uint8)

        cursor = 0
        if self._L > 1:
            for j in range(self._L):
                for i in range(self._L - 1):
                    eid = self.edge_index_h(i, j)
                    nid1 = self.node_index(i, j)
                    nid2 = self.node_index(i + 1, j)
                    rows[cursor : cursor + 2] = (nid1, nid2)
                    cols[cursor : cursor + 2] = eid
                    cursor += 2
            for j in range(self._L - 1):
                for i in range(self._L):
                    eid = self.edge_index_v(i, j)
                    nid1 = self.node_index(i, j)
                    nid2 = self.node_index(i, j + 1)
                    rows[cursor : cursor + 2] = (nid1, nid2)
                    cols[cursor : cursor + 2] = eid
                    cursor += 2

        self._check_matrix = sp.csr_matrix(
            (data, (rows, cols)), shape=(self.num_nodes, self.num_edges), dtype=np.uint8
        )
        return self._check_matrix

    def make_random_defects(self, p_edge: float, rng: np.random.Generator) -> dict:
        if not (0.0 <= p_edge <= 1.0):
            raise ValueError("p_edge must satisfy 0 ≤ p_edge ≤ 1.")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be an instance of numpy.random.Generator.")

        edge_flips = rng.random(self.num_edges) < p_edge
        H = self.to_check_matrix()
        syndrome = (H @ edge_flips.astype(np.uint8)) % 2
        syndrome = syndrome.astype(np.uint8, copy=False)
        return {
            "edge_flips": edge_flips.astype(bool, copy=False),
            "syndrome": syndrome,
        }

    def draw(self, mode: str, syndrome: np.ndarray | None = None) -> None:
        mode = mode.lower()
        if mode not in {"text", "graph"}:
            raise ValueError("mode must be 'text' or 'graph'.")
        if syndrome is not None:
            self._validate_syndrome(syndrome)
        if mode == "text":
            self._draw_text(syndrome)
        else:
            self._draw_graph(syndrome)

    def _draw_text(self, syndrome: np.ndarray | None) -> None:
        num_nodes = self._L
        lines: List[str] = []

        for j in range(num_nodes):
            glyphs: List[str] = []
            for i in range(num_nodes):
                nid = self.node_index(i, j)
                has_defect = bool(syndrome[nid]) if syndrome is not None else False
                glyphs.append("●" if has_defect else "o")
            node_row = "─".join(glyphs)
            full_row = f"•─{node_row}─•"
            lines.append(full_row)

            if j == num_nodes - 1:
                continue

            vertical = [" "] * len(full_row)
            vertical[0] = "•"
            vertical[-1] = "•"
            for idx in range(num_nodes):
                pos = 2 + 2 * idx
                vertical[pos] = "│"
            lines.append("".join(vertical))

        print("\n".join(lines))

    def _draw_graph(self, syndrome: np.ndarray | None) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        # Draw physical edges
        if self._L > 1:
            for j in range(self._L):
                for i in range(self._L - 1):
                    ax.plot([i, i + 1], [j, j], color="black", linewidth=1.0, zorder=1)
            for j in range(self._L - 1):
                for i in range(self._L):
                    ax.plot([i, i], [j, j + 1], color="black", linewidth=1.0, zorder=1)

        # Draw nodes
        coords = self.node_coords
        x_coords: list[float] = []
        y_coords: list[float] = []
        filled: list[bool] = []
        for nid in range(self.num_nodes):
            x, y = coords[nid]
            x_coords.append(float(x))
            y_coords.append(float(y))
            filled.append(bool(syndrome[nid]) if syndrome is not None else False)
        x_arr = np.asarray(x_coords)
        y_arr = np.asarray(y_coords)
        mask_filled = np.array(filled, dtype=bool)
        mask_empty = ~mask_filled
        if mask_empty.any():
            ax.scatter(
                x_arr[mask_empty],
                y_arr[mask_empty],
                s=100,
                facecolors="white",
                edgecolors="black",
                linewidths=1.5,
                zorder=3,
            )
        if mask_filled.any():
            ax.scatter(
                x_arr[mask_filled],
                y_arr[mask_filled],
                s=100,
                facecolors="black",
                edgecolors="black",
                linewidths=1.5,
                zorder=4,
            )

        # Virtual boundaries (left/right)
        rows = np.arange(self._L, dtype=float)
        left_x = -1.0
        right_x = float(self._L)
        for y in rows:
            ax.plot([left_x, 0.0], [y, y], color="black", linewidth=1.0, zorder=1)
            ax.plot([self._L - 1, right_x], [y, y], color="black", linewidth=1.0, zorder=1)
        if rows.size:
            ax.scatter(
                np.full_like(rows, left_x),
                rows,
                s=80,
                facecolors="gold",
                edgecolors="black",
                linewidths=1.0,
                zorder=2,
            )
            ax.scatter(
                np.full_like(rows, right_x),
                rows,
                s=80,
                facecolors="gold",
                edgecolors="black",
                linewidths=1.0,
                zorder=2,
            )

        ax.set_aspect("equal")
        ax.set_xlim(left_x - 0.5, right_x + 0.5)
        ymax = self._L - 0.5 if self._L else 0.5
        ymin = -0.5
        ax.set_ylim(ymax, ymin)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        plt.show()

    def _validate_coords(self, i: int, j: int) -> None:
        if not (0 <= i < self._L and 0 <= j < self._L):
            raise ValueError("Node coordinates out of bounds.")

    def _validate_node_index(self, nid: int) -> None:
        if not (0 <= nid < self.num_nodes):
            raise ValueError("Node index out of bounds.")

    def _validate_edge_index(self, eid: int) -> None:
        if not (0 <= eid < self.num_edges):
            raise ValueError("Edge index out of bounds.")

    def _validate_syndrome(self, syndrome: np.ndarray) -> None:
        if syndrome.shape != (self.num_nodes,):
            raise ValueError("syndrome must have shape (num_nodes,).")
        if syndrome.dtype.kind not in {"b", "i", "u"}:
            raise TypeError("syndrome must have boolean or integer dtype.")
        if not np.all((syndrome == 0) | (syndrome == 1)):
            raise ValueError("syndrome must be binary (0/1).")     
    def __repr__(self):
        return f"Lattice({self._L}x{self._L})"


