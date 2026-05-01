import math
from typing import Dict, List, Optional, Set
import torch
import tvm
from torch_geometric.data import Data
from tvm import tir


# mapping tables for categorical features
BIND_TYPES = {
    "blockIdx.x": 1, "blockIdx.y": 2, "blockIdx.z": 3,
    "threadIdx.x": 4, "threadIdx.y": 5, "threadIdx.z": 6,
}

SCOPE_MAP = {"global": 0, "shared": 1, "local": 2, "other": 3}

DTYPE_MAP = {
    "float32": 0, "float16": 1, "int32": 2, "int8": 3,
    "uint8": 4, "int64": 5, "float64": 6,
}


def _log2(n):
    return math.log2(max(int(n), 1) + 1.0)


def _try_int(expr, default=1):

    if isinstance(expr, int):
        return expr
    if isinstance(expr, tir.IntImm):
        return int(expr.value)
    if hasattr(expr, "value"):
        try:
            return int(expr.value)
        except (TypeError, ValueError):
            pass
    return default


def _is_tir_type(obj, type_name):

    cls = getattr(tir, type_name, None)
    if cls is not None and isinstance(obj, cls):
        return True
    stmt_mod = getattr(tir, "stmt", None)
    if stmt_mod is not None:
        cls2 = getattr(stmt_mod, type_name, None)
        if cls2 is not None and isinstance(obj, cls2):
            return True
    return False


def _get_var_name(v):
    return v.name if hasattr(v, "name") else str(v)


def _get_buf_names(regions):
    if not regions:
        return []
    names = []
    for r in regions:
        if hasattr(r, "buffer") and hasattr(r.buffer, "name"):
            names.append(r.buffer.name)
    return names


def _dtype_bytes(dtype_str):
    if dtype_str.endswith("64"):
        return 8
    elif dtype_str.endswith("16"):
        return 2
    elif dtype_str.endswith("8"):
        return 1
    return 4  # default to 4 (float32, int32)


class GraphBuilder:
    def __init__(self, feat_dim=16):
        assert feat_dim >= 16, 
        self.feat_dim = feat_dim

    def build(self, ir_mod: tvm.IRModule) -> Data:
        #Build a PyG Data from an IRModule
        self._reset()

        func = self._find_prim_func(ir_mod)
        if func is None:
            return self._make_empty_graph()

        # check for col-major loads 
        try:
            self._has_col_major = "col_major" in str(ir_mod)
        except Exception:
            pass

        
        self._scan_reduction_vars(func.body)

        # create nodes for function parameters (input/output buffers)
        self._add_param_buffers(func)

        # main traversal
        self._walk(func.body, parent_id=None, depth=0)

        return self._build_pyg_data()

    def build_from_schedule(self, sch: tir.Schedule) -> Data:
        return self.build(sch.mod)

   

    def _reset(self):
        #Clear all state for a fresh graph build.
        self._nodes: List[List[float]] = []
        self._edges: List[List[int]] = []
        # self._edge_types: List[int] = []  # might want this later for heterogeneous edges
        self._next_id = 0
        self._buf_node_ids: Dict[str, int] = {}
        self._reduction_vars: Set[str] = set()

        # accumulators for global features (dims 0-15)
        self._n_for = 0
        self._n_block = 0
        self._n_buf = 0
        self._loop_prod = 1  # product of all loop extents
        self._max_depth = 0
        self._n_thread_bind = 0
        self._n_reduce = 0
        self._n_spatial = 0
        self._buf_sizes: List[int] = []
        self._buf_bytes_total = 0
        self._n_vec = 0
        self._n_unroll = 0
        self._n_parallel = 0

        # accumulators for global features (dims 16-31)
        self._bind_ext: Dict[str, int] = {
            "blockIdx.x": 1, "blockIdx.y": 1, "blockIdx.z": 1,
            "threadIdx.x": 1, "threadIdx.y": 1, "threadIdx.z": 1,
        }
        self._vec_widths: List[int] = []
        self._max_unroll_step = 0
        self._has_col_major = False

        # create an analyzer for simplifying TIR expressions
        self._analyzer = tvm.arith.Analyzer()

        # debug counter — how many stmts did we skip
        self._n_skipped = 0

    def _new_feat(self):
        return [0.0] * self.feat_dim

    def _add_node(self, feat):
        nid = self._next_id
        self._nodes.append(feat)
        self._next_id += 1
        return nid

    def _add_edge(self, src, dst):
        self._edges.append([src, dst])



    def _find_prim_func(self, mod):
        if "main" in mod.functions:
            f = mod["main"]
            if isinstance(f, tir.PrimFunc):
                return f
        for _, f in mod.functions.items():
            if isinstance(f, tir.PrimFunc):
                return f
        return None

    def _scan_reduction_vars(self, stmt):
        if stmt is None:
            return

        if _is_tir_type(stmt, "BlockRealize"):
            block = stmt.block
            if hasattr(block, "iter_vars") and hasattr(stmt, "iter_values"):
                for iv, val in zip(block.iter_vars, stmt.iter_values):
                    # check if this iter var is a reduction axis
                    it = getattr(iv, "iter_type", None)
                    is_red = False
                    if it is not None:
                        try:
                            is_red = (int(it) == 2)
                        except (TypeError, ValueError):
                            is_red = "reduce" in str(it).lower()
                    if is_red:
                        self._reduction_vars.update(
                            self._extract_var_names(val))
            self._scan_reduction_vars(block.body)
            return

        if _is_tir_type(stmt, "SeqStmt"):
            for s in stmt.seq:
                self._scan_reduction_vars(s)
            return

        # recurse into body/then/else
        for attr in ("body", "then_case", "else_case"):
            child = getattr(stmt, attr, None)
            if child is not None:
                self._scan_reduction_vars(child)

    def _extract_var_names(self, expr):
        #Pull out variable names from a simple expression
        out = []
        if expr is None:
            return out
        if isinstance(expr, tir.Var):
            out.append(_get_var_name(expr))
        for attr in ("a", "b", "value", "index"):
            child = getattr(expr, attr, None)
            if isinstance(child, tir.Var):
                out.append(_get_var_name(child))
        return out

    # buffer nodes -

    def _add_param_buffers(self, func):
        #Create buffer nodes 
        for param in func.params:
            if param not in func.buffer_map:
                continue
            buf = func.buffer_map[param]
            name = buf.name if hasattr(buf, "name") else str(param)
            shape = [_try_int(s) for s in buf.shape]
            dtype = str(buf.dtype)
            scope = str(getattr(buf, "scope", "global")) or "global"
            self._make_buf_node(name, shape, dtype, scope)

    def _make_buf_node(self, name, shape, dtype, scope):
        """Create a single buffer node with features."""
        total_elems = 1
        for s in shape:
            total_elems *= max(int(s), 1)

        # this shouldn't happen but just in case
        if total_elems <= 0:
            total_elems = 1

        did = DTYPE_MAP.get(dtype, 0)

        # figure out the scope category
        if "global" in scope:
            sid = SCOPE_MAP["global"]
        elif "shared" in scope:
            sid = SCOPE_MAP["shared"]
        elif "local" in scope:
            sid = SCOPE_MAP["local"]
        else:
            sid = SCOPE_MAP["other"]

        feat = self._new_feat()
        feat[0] = 1.0  # node type = buffer
        feat[10] = min(len(shape) / 6.0, 1.0)
        feat[11] = _log2(total_elems) / 24.0
        if len(shape) >= 1:
            feat[12] = _log2(shape[0]) / 16.0
        if len(shape) >= 2:
            feat[13] = _log2(shape[1]) / 16.0
        feat[14] = min(did / 6.0, 1.0)
        feat[15] = min(sid / 3.0, 1.0)

        nid = self._add_node(feat)
        self._buf_node_ids[name] = nid

        # update global stats
        self._n_buf += 1
        self._buf_sizes.append(total_elems)
        self._buf_bytes_total += total_elems * _dtype_bytes(dtype)



    def _walk(self, stmt, parent_id, depth):
        if stmt is None:
            return

        if _is_tir_type(stmt, "For"):
            self._handle_for(stmt, parent_id, depth)
        elif _is_tir_type(stmt, "Block"):
            self._handle_block(stmt, parent_id, depth)
        elif _is_tir_type(stmt, "BlockRealize"):
            # unwrap and process the inner block
            self._walk(stmt.block, parent_id, depth)
        elif _is_tir_type(stmt, "SeqStmt"):
            for s in stmt.seq:
                self._walk(s, parent_id, depth)
        elif _is_tir_type(stmt, "Allocate"):
            self._handle_allocate(stmt, parent_id, depth)
        elif _is_tir_type(stmt, "AttrStmt"):
            # might have unroll annotations
            self._check_for_unroll(stmt)
            self._walk(stmt.body, parent_id, depth)
        elif _is_tir_type(stmt, "IfThenElse"):
            self._walk(stmt.then_case, parent_id, depth)
            if stmt.else_case is not None:
                self._walk(stmt.else_case, parent_id, depth)
        elif _is_tir_type(stmt, "LetStmt"):
            self._walk(stmt.body, parent_id, depth)
        elif _is_tir_type(stmt, "While"):
            self._walk(stmt.body, parent_id, depth)

    def _handle_for(self, stmt, parent_id, depth):
        var_name = _get_var_name(stmt.loop_var)
        extent = _try_int(stmt.extent, 1)
        kind = self._get_loop_kind(stmt)
        has_bind, bind_id = self._get_thread_binding(stmt)

        # clamp extent just in case
        extent = max(extent, 1)

        
        if self._reduction_vars:
            is_red = 1.0 if var_name in self._reduction_vars else 0.0
        else:
            # fallback heuristic if we couldn't detect properly
            n = var_name.lower()
            is_red = 1.0 if (n == "k" or n.startswith("r")) else 0.0

        feat = self._new_feat()
        feat[0] = 0.0  # node type = loop
        feat[1] = min(depth / 16.0, 1.0)
        feat[2] = _log2(extent) / 16.0
        feat[3] = min(kind / 4.0, 1.0)
        feat[4] = 1.0 if has_bind else 0.0
        feat[5] = bind_id / 9.0
        feat[6] = is_red

        nid = self._add_node(feat)
        if parent_id is not None:
            self._add_edge(parent_id, nid)

        # update global stats
        self._n_for += 1
        self._loop_prod = min(self._loop_prod * max(extent, 1), 1 << 62)
        self._max_depth = max(self._max_depth, depth + 1)

        if has_bind:
            self._n_thread_bind += 1
        if is_red > 0.5:
            self._n_reduce += 1
        else:
            self._n_spatial += 1

        if kind == 2:
            self._n_vec += 1
        elif kind == 3:
            self._n_unroll += 1
        elif kind == 1:
            self._n_parallel += 1

        # track per-axis thread binding extents
        if has_bind:
            bind_name = self._get_bind_name(stmt)
            if bind_name and bind_name in self._bind_ext:
                self._bind_ext[bind_name] = max(
                    self._bind_ext[bind_name], extent)

        if kind == 2:  # vectorized loop
            self._vec_widths.append(extent)

        # check for unroll annotations on this loop
        annot = getattr(stmt, "annotations", None)
        if annot:
            try:
                for key in annot:
                    if "pragma_auto_unroll_max_step" in str(key):
                        val = _try_int(annot[key], 0)
                        self._max_unroll_step = max(
                            self._max_unroll_step, val)
            except Exception:
                pass

        self._walk(stmt.body, nid, depth + 1)

    def _handle_block(self, stmt, parent_id, depth):
        n_iter = len(stmt.iter_vars) if hasattr(stmt, "iter_vars") else 0
        reads = _get_buf_names(getattr(stmt, "reads", None))
        writes = _get_buf_names(getattr(stmt, "writes", None))

        feat = self._new_feat()
        feat[0] = 0.5  # node type = computation block
        feat[1] = min(depth / 16.0, 1.0)
        feat[7] = min(n_iter / 10.0, 1.0)
        feat[8] = min(len(reads) / 8.0, 1.0)
        feat[9] = min(len(writes) / 8.0, 1.0)

        nid = self._add_node(feat)
        if parent_id is not None:
            self._add_edge(parent_id, nid)

        # dataflow edges: buffer -> block (reads), block -> buffer (writes)
        for buf_name in reads:
            bid = self._buf_node_ids.get(buf_name)
            if bid is not None:
                self._add_edge(bid, nid)
        for buf_name in writes:
            bid = self._buf_node_ids.get(buf_name)
            if bid is not None:
                self._add_edge(nid, bid)

        self._n_block += 1
        self._walk(stmt.body, nid, depth + 1)

    def _handle_allocate(self, stmt, parent_id, depth):
        name = _get_var_name(stmt.buffer_var)
        if name not in self._buf_node_ids:
            shape = [_try_int(e) for e in stmt.extents]
            dtype = str(getattr(stmt, "dtype", "float32"))
            self._make_buf_node(name, shape, dtype, "local")
        self._walk(stmt.body, parent_id, depth + 1)

    

    def _get_loop_kind(self, stmt):
        k = getattr(stmt, "kind", None)
        try:
            return int(k)
        except (TypeError, ValueError):
            s = str(k).lower() if k is not None else ""
            if "parallel" in s: return 1
            if "vector" in s: return 2
            if "unroll" in s: return 3
            if "thread" in s: return 4
            return 0

    def _get_thread_binding(self, stmt):
        tb = getattr(stmt, "thread_binding", None)
        if tb is None:
            return False, 0
        s = str(tb)
        for name, idx in BIND_TYPES.items():
            if name in s or name.replace(".", "_") in s:
                return True, idx
        return True, 0

    def _get_bind_name(self, stmt):
        tb = getattr(stmt, "thread_binding", None)
        if tb is None:
            return None
        s = str(tb)
        for name in BIND_TYPES:
            if name in s or name.replace(".", "_") in s:
                return name
        return None

    def _check_for_unroll(self, stmt):
        try:
            key = str(getattr(stmt, "attr_key", ""))
            if "pragma_auto_unroll_max_step" in key:
                val = _try_int(stmt.value, 0)
                self._max_unroll_step = max(self._max_unroll_step, val)
        except Exception:
            pass

    # ---- output 

    def _build_pyg_data(self):
        if len(self._nodes) == 0:
            return self._make_empty_graph()

        x = torch.tensor(self._nodes, dtype=torch.float32)

        if self._edges:
            ei = torch.tensor(self._edges, dtype=torch.long).t().contiguous()
        else:
            ei = torch.empty((2, 0), dtype=torch.long)

        data = Data(x=x, edge_index=ei)
        data.global_feat = self._compute_global_features()
        return data

    def _compute_global_features(self):
        gf = [0.0] * 32
        gf[0] = min(self._n_for / 20.0, 1.0)
        gf[1] = min(self._n_block / 5.0, 1.0)
        gf[2] = min(self._n_buf / 10.0, 1.0)
        gf[3] = min(_log2(self._loop_prod) / 40.0, 1.0)
        gf[4] = min(self._max_depth / 16.0, 1.0)
        gf[5] = min(self._n_thread_bind / 6.0, 1.0)
        gf[6] = min(self._n_reduce / 10.0, 1.0)
        gf[7] = min(self._n_spatial / 10.0, 1.0)

        if self._buf_sizes:
            gf[8] = min(_log2(max(self._buf_sizes)) / 24.0, 1.0)
            gf[9] = min(_log2(min(self._buf_sizes)) / 24.0, 1.0)
            avg = sum(self._buf_sizes) / len(self._buf_sizes)
            gf[10] = min(_log2(avg) / 24.0, 1.0)

        gf[11] = min(self._n_vec / 5.0, 1.0)
        gf[12] = min(self._n_unroll / 5.0, 1.0)
        gf[13] = min(self._n_parallel / 5.0, 1.0)
        gf[14] = min(_log2(self._buf_bytes_total) / 30.0, 1.0)

        # arithmetic intensity estimate
        if self._buf_bytes_total > 0:
            ai = self._loop_prod / float(self._buf_bytes_total)
            gf[15] = min(math.log2(ai + 1.0) / 20.0, 1.0)

        bx = self._bind_ext["blockIdx.x"]
        by = self._bind_ext["blockIdx.y"]
        bz = self._bind_ext["blockIdx.z"]
        tx = self._bind_ext["threadIdx.x"]
        ty = self._bind_ext["threadIdx.y"]
        tz = self._bind_ext["threadIdx.z"]

        gf[16] = min(_log2(bx) / 10.0, 1.0)
        gf[17] = min(_log2(by) / 10.0, 1.0)
        gf[18] = min(_log2(bz) / 10.0, 1.0)
        gf[19] = min(_log2(tx) / 10.0, 1.0)
        gf[20] = min(_log2(ty) / 10.0, 1.0)
        gf[21] = min(_log2(tz) / 10.0, 1.0)

        tpb = tx * ty * tz  # threads per block
        n_blocks = bx * by * bz
        gf[22] = min(_log2(tpb) / 12.0, 1.0)
        gf[23] = min(_log2(n_blocks) / 16.0, 1.0)

        if self._vec_widths:
            gf[24] = min(_log2(max(self._vec_widths)) / 5.0, 1.0)
        gf[25] = min(len(self._vec_widths) / 5.0, 1.0)

        gf[26] = min(_log2(self._max_unroll_step) / 12.0, 1.0)

        # grid/thread aspect ratios
        if by > 0:
            gf[27] = min(math.log2(bx / by + 1.0) / 10.0, 1.0)
        if ty > 0:
            gf[28] = min(math.log2(tx / ty + 1.0) / 10.0, 1.0)

        # work per thread (rough estimate)
        if tpb > 0:
            work = self._loop_prod / float(tpb * n_blocks)
            gf[29] = min(_log2(max(int(work), 1)) / 30.0, 1.0)

        gf[30] = 1.0 if self._has_col_major else 0.0
        gf[31] = 0.0  # reserved for future use

        return torch.tensor([gf], dtype=torch.float32)

    def _make_empty_graph(self):
        #Return a minimal valid graph when parsing fails
        x = torch.zeros((1, self.feat_dim), dtype=torch.float32)
        ei = torch.empty((2, 0), dtype=torch.long)
        data = Data(x=x, edge_index=ei)
        data.global_feat = torch.zeros((1, 32), dtype=torch.float32)
        return data


# quick sanity check if running this file directly
if __name__ == "__main__":
    print("graph builder module loaded ok")
    builder = GraphBuilder()
    empty = builder._make_empty_graph()
    print(f"empty graph: {empty.x.shape}, {empty.edge_index.shape}")
    print(f"global feat: {empty.global_feat.shape}")