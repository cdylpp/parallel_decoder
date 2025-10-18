from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import stim
import pymatching as pm
import numpy as np
import threading
from decoders.window import Window, Kind, Coord



def decode_window(Z_win: np.ndarray, matching) -> list[list[tuple[int, int]]]:
    """
    Z_win: (shots, num_detectors_global) masked global-shaped copy (only window dets set).
    Returns: list of length `shots`; each item is a list of (u,v) global detector indices.
    """
    shots = Z_win.shape[0]
    pairs_per_shot: list[list[tuple[int, int]]] = []
    for i in range(shots):
        pairs_i = matching.decode_to_matched_dets_array(Z_win[i])  # shape (m_i, 2)
        pairs_per_shot.append([(int(u), int(v)) for (u, v) in pairs_i])
    return pairs_per_shot


def write_commit(Z_global: np.ndarray, commit_set: set[int], pairs_per_shot: list[list[tuple[int, int]]]) -> np.ndarray:
    """
    XOR only on commit columns using matched detector pairs i.e., global indices.
    """
    for i, pairs in enumerate(pairs_per_shot):
        for (u, v) in pairs:
            if (u in commit_set) and (v in commit_set):
                Z_global[i, u] ^= True
                Z_global[i, v] ^= True
    return Z_global


def make_window_view(global_syndromes: np.ndarray, win_dets: np.ndarray) -> np.ndarray:
    """
    Return a global-shaped masked copy with only window columns populated.
    """
    win_syndrome = np.zeros_like(global_syndromes, dtype=bool)
    win_syndrome[:, win_dets] = global_syndromes[:, win_dets]
    return win_syndrome  # shape (shots, num_detectors_global)

def apply_time_boundary_mask(s_view: np.ndarray, left_mask: dict | None, right_mask: dict | None) -> None:
    """
    s_view: global-shaped masked copy (shots, num_detectors_global).
    Masks:
      {"cols": np.ndarray[int], "parity": np.ndarray[shots, n_cols]} or None

    XOR per-detector artificial defects into the boundary faces.
    """
    if left_mask is not None:
        cols = np.asarray(left_mask["cols"], dtype=int)
        par = np.asarray(left_mask["parity"], dtype=bool)  # [shots, len(cols)]
        if cols.size:
            s_view[:, cols] ^= par

    if right_mask is not None:
        cols = np.asarray(right_mask["cols"], dtype=int)
        par = np.asarray(right_mask["parity"], dtype=bool)  # [shots, len(cols)]
        if cols.size:
            s_view[:, cols] ^= par
    return

class ParallelWindowScheduler:
    """
    Temporal A/B windowing for parallel window decoding.
    Partition time [0, T) into length-n_commit commit blocks.
    set even blocks (0,2,4,...) -> Type B with commit-only.
    set odd blocks (1,3,5,...) -> Type A with buffer regions before/after.
    During execution:
        Layer 0: all Type A windows in parallel
        Layer 1: all Type B windows in parallel
    """
    def __init__(
        self,
        T: int,
        n_commit: int,
        n_buffer: int,
        d2c: Dict[int, List[Coord]],
        include_partial_tail: bool = True
    ):
        assert T > 0 and n_buffer > 0 and n_commit > 0
        self.T = T
        self.n_commit = n_commit
        self.n_buffer = n_buffer
        self.include_partial_tail = include_partial_tail
        
        t_vals = sorted({coord.t for coord in d2c.values()})

        def time_from_index(i):
            return t_vals[i]

        def clamp_index(i):
            return max(0, min(i, len(t_vals)-1))

        # Build commit blocks
        blocks = []
        k = 0
        while True:
            i0 = k * n_commit
            if i0 >= len(t_vals): 
                break
            i1 = min(len(t_vals), i0 + n_commit)
            if i1 - i0 == n_commit or include_partial_tail:
                blocks.append((i0, i1))
            else:
                break
            k += 1
        
        N = len(blocks)
        desired: List[Kind] = [("B" if i % 2 == 0 else "A") for i in range(N)]
        # Build the windows based on what 'kind' they are.
        # A windows have forward and back buffer; B windows are all commit.
        windows = []
        for i, (i0, i1) in enumerate(blocks):
            if desired[i] == "A":
                j0 = clamp_index(i0 - n_buffer)
                j1 = clamp_index(i1 + n_buffer - 1) + 1
                t0, t1 = time_from_index(j0), time_from_index(j1-1) + 1
                c0, c1 = time_from_index(i0), time_from_index(i1-1) + 1
                kind = "A" if (j0 == i0 - n_buffer and j1 == i1 + n_buffer) else "B"
            else:
                t0, t1 = time_from_index(i0), time_from_index(i1 - 1) + 1
                c0, c1 = t0, t1
                kind = "B"
            windows.append(Window(kind=kind, k=i, commit=(c0, c1), span=(t0, t1)))

        # If last ended up A but couldn't be buffered, force to B with commit span
        if windows and windows[-1].kind == "A":
            last = windows[-1]
            if last.t0 < 0 or last.t1 > T:
                windows[-1] = Window(kind="B", k=last.k, commit=last.commit, span=last.commit)

        self._windows = windows

    def layers(self) -> Tuple[List[Window], List[Window]]:
        A = [w for w in self._windows if w.kind == "A"]
        B = [w for w in self._windows if w.kind == "B"]
        return A, B

    # does the actual assignment of detectors to windows.
    def build_window_view(self, det_to_coords, window: Window):
        t0, t1 = window.t0, window.t1
        c0, c1 = window.c0, window.c1

        dets = sorted(d for d, coord in det_to_coords.items() if t0 <= coord.t < t1)

        commit_dets = []
        back_buffer = []
        forward_buffer = []
        boundary_dets = {"back": [], "forward": []}

        for d in dets:
            t = det_to_coords[d].t
            if c0 <= t < c1:
                commit_dets.append(d)
            elif t0 <= t < c0:
                back_buffer.append(d)
            elif c1 <= t < t1:
                forward_buffer.append(d)

        # Boundary faces depend on window
        if window.kind == "A":
            boundary_dets['back'] = [d for d in commit_dets if det_to_coords[d].t == c0]
            boundary_dets['forward'] = [d for d in commit_dets if (c1 - 1 >= c0) and det_to_coords[d].t == (c1 - 1)]
        else:
            boundary_dets['back'] = [d for d in dets if det_to_coords[d].t == t0]
            boundary_dets['forward'] = [d for d in dets if det_to_coords[d].t == (t1 - 1)]

        window.dets = dets
        window.commit_dets = commit_dets
        window.buffer_dets = [back_buffer, forward_buffer]
        window.boundary_dets = boundary_dets
        return

    def all_windows(self) -> List[Window]:
        return list(self._windows)

    def execution_order(self) -> List[List[Window]]:
        A, B = self.layers()
        return [A, B]

    def debug_summary(self) -> str:
        A, B = self.layers()
        def fmt(ws: List[Window]) -> str:
            return ", ".join(
                f"{w.kind}{w.k}: span[{w.t0}, {w.t1}) commit[{w.c0},{w.c1})"
                for w in ws
            )
        return f"A-layer: {fmt(A)}\nB-layer: {fmt(B)}"


@dataclass(frozen=True)
class DecodeResult:
    k: int
    kind: str
    tspan: Tuple[int, int]
    cspan: Tuple[int, int]
    z_hat: Optional[np.ndarray] = None
    boundary_meta: Optional[dict] = None


class ParallelDecoder:
    def __init__(
        self,
        window_size: Tuple[int, int],
        dem: stim.DetectorErrorModel,
        matching: pm.Matching,
        det_to_coords: Dict[int, List[Coord]],
        global_syndromes: np.ndarray,
        max_workers: Optional[int] = None,
    ):
        self.dem = dem
        self.matching = matching
        self.d2c = det_to_coords
        self.S = global_syndromes
        self.max_workers = max_workers
        self.Z_global = global_syndromes.astype(dtype=bool, copy=True)

        n_commit, n_buffer = window_size
        T = det_to_coords[max(det_to_coords.keys())].t
        self.scheduler = ParallelWindowScheduler(T,n_commit,n_buffer,self.d2c)
        self.windows: List = self.scheduler.all_windows()
        self.A_windows, self.B_windows = self.scheduler.layers()
        
        self.by_k: Dict[int, object] = {w.k: w for w in self.windows}
        self._A_done: Dict[int, threading.Event] = {w.k: threading.Event() for w in self.windows}
        self._A_boundary_meta: Dict[int, dict] = {}
        self._zg_lock = threading.Lock()

        # Build all window views (maps detectors to windows)
        for w in self.windows:
            self.scheduler.build_window_view(self.d2c, w)

        for w in self.A_windows:
            nb = self.by_k.get(w.k+1)
            if nb and nb.kind == "B":
                a_xy = sorted((int(self.d2c[d].x), int(self.d2c[d].y)) for d in w.boundary_dets["forward"])
                b_xy = sorted((int(self.d2c[d].x), int(self.d2c[d].y)) for d in w.boundary_dets["back"])
                assert a_xy == b_xy, f"Face XY mismatch A{w.k}.forward vs B{w.k+1}.back"


    def _decode_A_window(self, w) -> DecodeResult:
        t0, t1 = w.t0, w.t1
        c0, c1 = w.c0, w.c1

        # global-shaped masked copy
        win_syndrome = make_window_view(self.S, np.array(w.dets, dtype=int))

        # decode syndrome data for these detectors.
        pairs_per_shot = decode_window(win_syndrome, self.matching)

        # use a lock for thread safety.
        with self._zg_lock:
            self.Z_global = write_commit(self.Z_global, set(w.commit_dets), pairs_per_shot)

        # boundary meta i.e., per-detector parities from pairs.
        boundary_meta = self._compute_A_boundary_meta_from_pairs(w, pairs_per_shot)

        return DecodeResult(k=w.k, kind="A", tspan=(t0, t1), cspan=(c0, c1),
                            z_hat=None, boundary_meta=boundary_meta)

    def _compute_A_boundary_meta_from_pairs(self, w, pairs_per_shot) -> dict:
        """
        Produce per-detector boundary parity matrices with explicit global columns.
        """
        back_cols = np.asarray(w.boundary_dets["back"], dtype=int)
        fwd_cols = np.asarray(w.boundary_dets["forward"], dtype=int)
        n_back = back_cols.size
        n_fwd = fwd_cols.size

        back_pos = {int(c): j for j, c in enumerate(back_cols)}
        fwd_pos = {int(c): j for j, c in enumerate(fwd_cols)}

        shots = len(pairs_per_shot)
        back_mat = np.zeros((shots, n_back), dtype=bool) if n_back else np.zeros((shots, 0), dtype=bool)
        fwd_mat = np.zeros((shots, n_fwd), dtype=bool) if n_fwd else np.zeros((shots, 0), dtype=bool)

        # j index, b / f is back / forward
        for s, pairs in enumerate(pairs_per_shot):
            if n_back:
                row_b = back_mat[s]
                for (u, v) in pairs:
                    jb = back_pos.get(u)
                    if jb is not None: row_b[jb] ^= True
                    jb = back_pos.get(v)
                    if jb is not None: row_b[jb] ^= True
            if n_fwd:
                row_f = fwd_mat[s]
                for (u, v) in pairs:
                    jf = fwd_pos.get(u)
                    if jf is not None: row_f[jf] ^= True
                    jf = fwd_pos.get(v)
                    if jf is not None: row_f[jf] ^= True

        return {
            "k": w.k,
            "backward_edge": {"t": w.t0, "cols": back_cols, "parity": back_mat},
            "forward_edge":  {"t": max(w.c0, w.c1 - 1), "cols": fwd_cols, "parity": fwd_mat},
        }
    
    def _build_face_xy(self, face_cols, d2c):
        out = []
        for d in face_cols:
            c = d2c[d]
            out.append((d, (c.x, c.y)))
        return out

    def _align_by_xy_and_time(self, face_cols_B, meta_cols_A, meta_parity_A, d2c):
        B_xy = self._build_face_xy(face_cols_B, d2c)
        A_xy = dict(self._build_face_xy(meta_cols_A, d2c))
        shots, nB = meta_parity_A.shape[0], len(B_xy)
        out = np.zeros((shots, nB), dtype=bool)
        posA = {int(d): j for j, d in enumerate(meta_cols_A)}
        for jB, (dB, xy) in enumerate(B_xy):
            dA = A_xy.get(xy)
            if dA is None:
                continue
            jA = posA.get(dA)
            if jA is not None:
                out[:, jB] = meta_parity_A[:, jA]
        return out
    
    def derive_boundary_from_A(self, left_meta, right_meta, w: Window, d2c):
        left_cols = np.asarray(w.boundary_dets["back"], dtype=int)
        right_cols = np.asarray(w.boundary_dets["forward"], dtype=int)

        left_parity_mat = None
        right_parity_mat = None

        if left_meta is not None:
            # Use A_{k-1}'s forward edge 
            # A_{k -1} c1-1 face for B's left face at t0
            A_cols = np.asarray(left_meta["forward_edge"]["cols"], dtype=int)
            A_par = np.asarray(left_meta["forward_edge"]["parity"], dtype=bool)  # (shots, n_meta)
            left_parity_mat = self._align_by_xy_and_time(left_cols, A_cols, A_par, d2c)

        if right_meta is not None:
            # Use A_{k+1}'s backward edge 
            # A_{k+1} at its c0 face for B's right face at t1-1
            A_cols = np.asarray(right_meta["backward_edge"]["cols"], dtype=int)
            A_par = np.asarray(right_meta["backward_edge"]["parity"], dtype=bool)  # (shots, n_meta)
            right_parity_mat = self._align_by_xy_and_time(right_cols, A_cols, A_par, d2c)

        left_mask = None if left_cols.size == 0 or left_parity_mat is None else {"cols": left_cols, "parity": left_parity_mat}
        right_mask = None if right_cols.size == 0 or right_parity_mat is None else {"cols": right_cols, "parity": right_parity_mat}
        return left_mask, right_mask

    def _decode_B_window(self, w) -> DecodeResult:
        t0, t1 = w.t0, w.t1
        c0, c1 = w.c0, w.c1

        # global-shaped masked copy for B window
        win_syndrome = make_window_view(self.S, np.array(w.dets, dtype=int))

        # get boundary meta from neighbors A_{k-1}, A_{k+1} and inject per-detector parity.
        left_meta = self._A_boundary_meta.get(w.k - 1)   # from A_{k-1}.forward
        right_meta = self._A_boundary_meta.get(w.k + 1)  # from A_{k+1}.back
        left_mask, right_mask = self.derive_boundary_from_A(left_meta, right_meta, w, self.d2c)
        
        apply_time_boundary_mask(win_syndrome, left_mask, right_mask)

        # Decode to pairs-per-shot (global indices)
        # pairs_per_shot = decode_window(win_syndrome, self.matching)
        # Thread-safe commit inside [c0,c1)
        # with self._zg_lock:
            # self.Z_global = write_commit(self.Z_global, set(w.commit_dets), pairs_per_shot)

        return DecodeResult(k=w.k, kind="B", tspan=(t0, t1), cspan=(c0, c1),
                            z_hat=None, boundary_meta=None)

    def decode(self, mode: str = "barrier", return_results = False) -> List | Dict[str, List[DecodeResult]]:
        """
        mode in {"barrier", "streaming"}
        barrier := runs all A first, then B
        streaming := once B's neighbors are complete, run B.
        """
        assert mode in ("barrier", "streaming")
        if mode == "streaming":
            print("Warning: Streaming is broken, switching to `barrier`")
            mode = "barrier"
        

        results_A: List[DecodeResult] = []
        results_B: List[DecodeResult] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futs = {pool.submit(self._decode_A_window, w): w for w in self.A_windows}
            for fut in as_completed(futs):
                res = fut.result()
                results_A.append(res)
                self._A_boundary_meta[res.k] = res.boundary_meta or {}
                ev = self._A_done.get(res.k)
                if ev is not None:
                    ev.set()

        if mode == "barrier":
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futs = {pool.submit(self._decode_B_window, w): w for w in self.B_windows}
                for fut in as_completed(futs):
                    results_B.append(fut.result())
        else:  # streaming
            def _launch_B(w):
                # wait for A_{k-1} and A_{k+1} if they exist
                if (w.k - 1) in self._A_done:
                    self._A_done[w.k - 1].wait()
                if (w.k + 1) in self._A_done:
                    self._A_done[w.k + 1].wait()
                return self._decode_B_window(w)

            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futs = {pool.submit(_launch_B, w): w for w in self.B_windows}
                for fut in as_completed(futs):
                    results_B.append(fut.result())

        if return_results:
            self.Z_global, {"A": sorted(results_A, key=lambda r: r.k),
            "B": sorted(results_B, key=lambda r: r.k)}
        else:
            return self.Z_global
