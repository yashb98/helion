from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch

from ..autotuner.config_spec import KernelGridFact
from ..autotuner.config_spec import LiveTile
from ..autotuner.config_spec import RootGridFact
from ..language import _tracing_ops

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .compile_environment import CompileEnvironment
    from .device_ir import DeviceIR
    from .device_ir import GraphInfo


def matmul_operand_positions() -> dict[object, tuple[int, int]]:
    """Matmul/dot FX targets mapped to their lhs/rhs argument positions."""
    from ..language import matmul_ops

    return {
        matmul_ops.dot_scaled: (0, 3),
        matmul_ops.dot: (0, 1),
        torch.ops.aten.mm.default: (0, 1),
        torch.ops.aten.bmm.default: (0, 1),
        torch.ops.aten.addmm.default: (1, 2),
        torch.ops.aten.baddbmm.default: (1, 2),
    }


def trace_back_to_load(arg: object, load_op: object) -> torch.fx.Node | None:
    """Follow a matmul operand through pass-through ops to one load node."""
    cur = arg
    for _ in range(8):
        if not isinstance(cur, torch.fx.Node):
            return None
        if cur.target is load_op:
            return cur
        tensor_inputs = [
            value
            for value in cur.args
            if isinstance(value, torch.fx.Node)
            and isinstance(value.meta.get("val"), torch.Tensor)
        ]
        if len(tensor_inputs) != 1:
            return None
        cur = tensor_inputs[0]
    return None


def tile_rank(dims: tuple[int | None, ...]) -> int:
    """Number of block dimensions spanned by a tile."""
    return sum(dim is not None for dim in dims)


def tile_set_rank_profile(
    tiles: Iterable[tuple[int | None, ...]],
    max_rank: int,
) -> tuple[int, ...]:
    """Block-size-free lexicographic footprint key, highest rank first."""
    by_rank: dict[int, int] = {}
    for tile in tiles:
        rank = tile_rank(tile)
        if rank:
            by_rank[rank] = by_rank.get(rank, 0) + 1
    return tuple(by_rank.get(rank, 0) for rank in range(max_rank, 0, -1))


def _live_tile_kind(node: torch.fx.Node, dot_targets: frozenset[object]) -> str:
    from ..language import memory_ops

    if node.op == "placeholder":
        return "carry"
    if node.op == "call_function":
        if node.target in dot_targets:
            return "dot_out"
        if node.target is memory_ops.load:
            return "load"
    return "other"


def _tile_from_tensor(
    value: object,
    env: CompileEnvironment,
    *,
    kind: str,
    stageable: bool | None = None,
) -> LiveTile | None:
    if not (isinstance(value, torch.Tensor) and value.shape):
        return None
    dims = tuple(env.resolve_block_id(size) for size in value.shape)
    static_dims: list[int | None] = []
    for size, block_id in zip(value.shape, dims, strict=False):
        if block_id is not None:
            static_dims.append(None)
            continue
        try:
            static_dims.append(int(env.size_hint(size)))
        except Exception:
            static_dims.append(None)
    return LiveTile(
        dim_block_ids=dims,
        static_dims=tuple(static_dims),
        itemsize=value.dtype.itemsize,
        kind=kind,
        stageable=stageable,
    )


def _index_depends_on_loop(
    obj: object,
    env: CompileEnvironment,
    loop_block_ids: frozenset[int],
    seen: set[torch.fx.Node] | None = None,
) -> bool | None:
    """Whether an FX index is proven to depend on an enclosing loop axis."""
    if isinstance(obj, (list, tuple)):
        states = [
            _index_depends_on_loop(item, env, loop_block_ids, seen) for item in obj
        ]
        if any(state is True for state in states):
            return True
        if any(state is None for state in states):
            return None
        return False
    if isinstance(obj, dict):
        return _index_depends_on_loop(tuple(obj.values()), env, loop_block_ids, seen)
    if isinstance(obj, torch.fx.Node):
        seen = set() if seen is None else seen
        if obj in seen:
            return False
        seen.add(obj)
        value = obj.meta.get("val")
        known_origin = False
        try:
            block_id = env.resolve_block_id(value)
        except Exception:
            block_id = None
        if block_id is not None:
            known_origin = True
            if block_id in loop_block_ids:
                return True
        if isinstance(value, torch.Tensor):
            for dim in value.shape:
                try:
                    dim_block_id = env.resolve_block_id(dim)
                except Exception:
                    dim_block_id = None
                if dim_block_id is not None:
                    known_origin = True
                    if dim_block_id in loop_block_ids:
                        return True
        if obj.op == "placeholder":
            return False if known_origin else None
        children = (*obj.args, *obj.kwargs.values())
        if children:
            state = _index_depends_on_loop(children, env, loop_block_ids, seen)
            if state is not False:
                return state
            return False
        if obj.op == "get_attr":
            return False
        return False if known_origin else None
    if isinstance(obj, torch.SymInt):
        try:
            block_id = env.resolve_block_id(obj)
        except Exception:
            return None
        return block_id in loop_block_ids if block_id is not None else None
    if isinstance(obj, torch.Tensor):
        return None
    return False


@dataclasses.dataclass
class GraphAnalysis:
    """Shared structural and liveness observations for one DeviceIR graph."""

    graph_id: int
    nodes: tuple[torch.fx.Node, ...]
    is_reduction_loop: bool
    block_ids: frozenset[int]
    live_tile_steps: tuple[tuple[LiveTile, ...], ...]
    peak_live_tiles: tuple[LiveTile, ...]
    peak_dot_output_tiles: tuple[LiveTile, ...]
    dot_nodes: tuple[torch.fx.Node, ...]
    reduction_occurrences: tuple[int, ...]
    reduction_axis_by_node_id: dict[int, int]
    reduction_input_itemsizes: tuple[tuple[int, int], ...]
    memory_tiles: tuple[tuple[torch.fx.Node, LiveTile], ...]
    _memory_tiles_by_loop_axes: dict[frozenset[int], tuple[LiveTile, ...]] = (
        dataclasses.field(
            default_factory=dict,
            repr=False,
        )
    )

    @classmethod
    def build(
        cls,
        graph_info: GraphInfo,
        env: CompileEnvironment,
        *,
        is_reduction_loop: bool,
        dot_targets: frozenset[object],
    ) -> GraphAnalysis:
        from ..language import memory_ops
        from .inductor_lowering import ReductionLowering

        graph = graph_info.graph
        nodes = tuple(graph.nodes)
        last_use: dict[torch.fx.Node, int] = {}
        tile_details: dict[torch.fx.Node, LiveTile] = {}
        dot_nodes: list[torch.fx.Node] = []
        reduction_occurrences: list[int] = []
        seen_reductions: set[int] = set()
        reduction_axis_by_node_id: dict[int, int] = {}
        reduction_input_itemsizes: list[tuple[int, int]] = []
        memory_tiles: list[tuple[torch.fx.Node, LiveTile]] = []

        for index, node in enumerate(nodes):
            for input_node in node.all_input_nodes:
                last_use[input_node] = index

            kind = _live_tile_kind(node, dot_targets)
            tile = _tile_from_tensor(node.meta.get("val"), env, kind=kind)
            if tile is not None and any(
                block_id is not None for block_id in tile.dim_block_ids
            ):
                tile_details[node] = tile

            if node.op == "call_function" and node.target in dot_targets:
                dot_nodes.append(node)

            lowering = node.meta.get("lowering")
            if isinstance(lowering, ReductionLowering):
                block_id = getattr(lowering, "block_index", None)
                if isinstance(block_id, int):
                    reduction_axis_by_node_id[id(node)] = block_id
                    if block_id not in seen_reductions:
                        seen_reductions.add(block_id)
                        reduction_occurrences.append(block_id)
                    for input_node in node.all_input_nodes:
                        input_value = input_node.meta.get("val")
                        if isinstance(input_value, torch.Tensor):
                            reduction_input_itemsizes.append(
                                (block_id, input_value.element_size())
                            )
                            break

            if node.op != "call_function":
                continue
            if node.target is memory_ops.load:
                memory_tile = _tile_from_tensor(
                    node.meta.get("val"),
                    env,
                    kind="load",
                )
            elif node.target is memory_ops.store:
                stored_value = None
                for arg in node.args:
                    if isinstance(arg, torch.fx.Node):
                        candidate = arg.meta.get("val")
                        if isinstance(candidate, torch.Tensor) and candidate.shape:
                            stored_value = candidate
                memory_tile = _tile_from_tensor(stored_value, env, kind="store")
            else:
                memory_tile = None
            if memory_tile is not None:
                memory_tiles.append((node, memory_tile))

        live_tile_steps: list[tuple[LiveTile, ...]] = []
        seen_steps: set[frozenset[int]] = set()
        max_rank = max(
            (tile_rank(tile.dim_block_ids) for tile in tile_details.values()),
            default=0,
        )
        best_key: tuple[int, ...] = ()
        peak_live_tiles: tuple[LiveTile, ...] = ()
        live: set[torch.fx.Node] = set()
        for index, node in enumerate(nodes):
            if node in tile_details:
                live.add(node)
            if live:
                step_key = frozenset(id(value) for value in live)
                step = tuple(tile_details[value] for value in live)
                if step_key not in seen_steps:
                    seen_steps.add(step_key)
                    live_tile_steps.append(step)
                key = tile_set_rank_profile(
                    (tile_details[value].dim_block_ids for value in live),
                    max_rank,
                )
                if key > best_key:
                    best_key = key
                    peak_live_tiles = step
            live = {value for value in live if last_use.get(value, -1) > index}

        dot_details = {
            node: tile
            for node in dot_nodes
            if (
                tile := _tile_from_tensor(
                    node.meta.get("val"),
                    env,
                    kind="dot_out",
                )
            )
            is not None
        }
        best_dot_key = (-1, -1)
        peak_dot_output_tiles: tuple[LiveTile, ...] = ()
        live_dots: set[torch.fx.Node] = set()
        for index, node in enumerate(nodes):
            if node in dot_details:
                live_dots.add(node)
            if live_dots:
                tiles = tuple(dot_details[value] for value in live_dots)
                key = (
                    sum(tile_rank(tile.dim_block_ids) for tile in tiles),
                    len(tiles),
                )
                if key > best_dot_key:
                    best_dot_key = key
                    peak_dot_output_tiles = tiles
            live_dots = {
                value for value in live_dots if last_use.get(value, len(nodes)) > index
            }

        return cls(
            graph_id=graph_info.graph_id,
            nodes=nodes,
            is_reduction_loop=is_reduction_loop,
            block_ids=frozenset(getattr(graph_info, "block_ids", ()) or ()),
            live_tile_steps=tuple(live_tile_steps),
            peak_live_tiles=peak_live_tiles,
            peak_dot_output_tiles=peak_dot_output_tiles,
            dot_nodes=tuple(dot_nodes),
            reduction_occurrences=tuple(reduction_occurrences),
            reduction_axis_by_node_id=reduction_axis_by_node_id,
            reduction_input_itemsizes=tuple(reduction_input_itemsizes),
            memory_tiles=tuple(memory_tiles),
        )

    def reaches_output(self, node: torch.fx.Node, limit: int = 64) -> bool:
        """Whether a value reaches the graph output under the legacy bounded walk."""
        frontier = [node]
        seen: set[torch.fx.Node] = {node}
        steps = 0
        while frontier and steps < limit:
            steps += 1
            current = frontier.pop()
            for user in current.users:
                if user.op == "output":
                    return True
                if user not in seen:
                    seen.add(user)
                    frontier.append(user)
        return False

    def memory_tiles_for_loop_axes(
        self,
        env: CompileEnvironment,
        loop_block_ids: frozenset[int] = frozenset(),
    ) -> tuple[LiveTile, ...]:
        """Memory tiles with load stageability resolved for the given loop axes."""
        cached = self._memory_tiles_by_loop_axes.get(loop_block_ids)
        if cached is not None:
            return cached
        out: list[LiveTile] = []
        for node, tile in self.memory_tiles:
            if tile.kind == "load":
                stageable = (
                    _index_depends_on_loop(node.args[1], env, loop_block_ids)
                    if len(node.args) > 1 and loop_block_ids
                    else False
                )
            else:
                stageable = False
            out.append(tile._replace(stageable=stageable))
        result = tuple(out)
        self._memory_tiles_by_loop_axes[loop_block_ids] = result
        return result


@dataclasses.dataclass
class DeviceIRAnalysis:
    """Shared graph interpretation for all fact builders of one DeviceIR."""

    graphs: tuple[GraphAnalysis, ...]
    by_id: dict[int, GraphAnalysis]
    child_loops: dict[int, tuple[tuple[int, frozenset[int]], ...]]
    parent_of: dict[int, int]
    loop_block_ids: dict[int, frozenset[int]]
    loop_calls: dict[int, tuple[torch.fx.Node, ...]]
    dot_nodes: tuple[tuple[int, torch.fx.Node], ...]
    kernel_grid_fact: KernelGridFact

    @classmethod
    def build(
        cls,
        device_ir: DeviceIR,
        env: CompileEnvironment,
    ) -> DeviceIRAnalysis:
        from .device_ir import ReductionLoopGraphInfo

        dot_targets = frozenset(matmul_operand_positions())
        graphs = tuple(
            GraphAnalysis.build(
                graph_info,
                env,
                is_reduction_loop=isinstance(
                    graph_info,
                    ReductionLoopGraphInfo,
                ),
                dot_targets=dot_targets,
            )
            for graph_info in device_ir.graphs
        )
        by_id = {graph.graph_id: graph for graph in graphs}
        child_loops: dict[int, tuple[tuple[int, frozenset[int]], ...]] = {}
        loop_calls: dict[int, list[torch.fx.Node]] = {}
        graph_parent_of: dict[int, int] = {}
        for graph in graphs:
            if graph.is_reduction_loop:
                continue
            edges: list[tuple[int, frozenset[int]]] = []
            for node in graph.nodes:
                if (
                    node.op != "call_function"
                    or not _tracing_ops.is_for_loop_target(node.target)
                    or not node.args
                    or not isinstance(node.args[0], int)
                ):
                    continue
                body_id = node.args[0]
                loop_calls.setdefault(body_id, []).append(node)
                body = by_id.get(body_id)
                if body is not None:
                    graph_parent_of[body_id] = graph.graph_id
                if body is not None and not body.is_reduction_loop:
                    edges.append((body_id, body.block_ids))
            if edges:
                child_loops[graph.graph_id] = tuple(edges)

        parent_of: dict[int, int] = {}
        loop_block_ids: dict[int, frozenset[int]] = {}
        for parent_id, loop_edges in child_loops.items():
            for body_id, block_ids in loop_edges:
                parent_of[body_id] = parent_id
                loop_block_ids[body_id] = block_ids

        dot_nodes = tuple(
            (graph.graph_id, node)
            for graph in graphs
            if not graph.is_reduction_loop
            for node in graph.dot_nodes
        )
        roots = tuple(
            RootGridFact(graph_id, tuple(block_ids))
            for graph_id, block_ids in zip(
                device_ir.root_ids,
                device_ir.grid_block_ids,
                strict=True,
            )
        )
        root_ids = {root.root_graph_id for root in roots}
        graph_to_root: list[tuple[int, int]] = []
        for graph in graphs:
            current = graph.graph_id
            seen: set[int] = set()
            while current not in root_ids and current not in seen:
                seen.add(current)
                current = graph_parent_of.get(current, -1)
            if current in root_ids:
                graph_to_root.append((graph.graph_id, current))
        kernel_grid_fact = KernelGridFact(roots, tuple(graph_to_root))
        return cls(
            graphs=graphs,
            by_id=by_id,
            child_loops=child_loops,
            parent_of=parent_of,
            loop_block_ids=loop_block_ids,
            loop_calls={
                graph_id: tuple(calls) for graph_id, calls in loop_calls.items()
            },
            dot_nodes=dot_nodes,
            kernel_grid_fact=kernel_grid_fact,
        )

    @property
    def non_reduction_graphs(self) -> tuple[GraphAnalysis, ...]:
        return tuple(graph for graph in self.graphs if not graph.is_reduction_loop)

    def original_reductions(self) -> tuple[tuple[int, int], ...]:
        """Ordered, deduplicated reduction occurrences in original graphs."""
        return tuple(
            (graph.graph_id, block_id)
            for graph in self.non_reduction_graphs
            for block_id in graph.reduction_occurrences
        )

    def reduction_input_itemsize(self, block_id: int) -> int:
        """Legacy last-occurrence input width for one reduction axis."""
        itemsize = 0
        for graph in self.graphs:
            for axis, width in graph.reduction_input_itemsizes:
                if axis == block_id:
                    itemsize = width
        return itemsize

    def kernel_peak_live_tiles(self) -> tuple[LiveTile, ...]:
        """Legacy max-by-rank-profile live set across original graphs."""
        best: tuple[LiveTile, ...] = ()
        best_key: tuple[int, ...] = ()
        for graph in self.non_reduction_graphs:
            tiles = graph.peak_live_tiles
            max_rank = max(
                (tile_rank(tile.dim_block_ids) for tile in tiles),
                default=0,
            )
            key = tile_set_rank_profile(
                (tile.dim_block_ids for tile in tiles),
                max_rank,
            )
            if key > best_key:
                best_key = key
                best = tiles
        return best

    def kernel_live_tile_steps(self) -> tuple[tuple[LiveTile, ...], ...]:
        return tuple(
            step
            for graph in self.non_reduction_graphs
            for step in graph.live_tile_steps
        )

    def kernel_peak_dot_outputs(self) -> tuple[LiveTile, ...]:
        """Peak dot-output set after adding accumulator ancestor chains."""
        accumulator_of = {
            graph.graph_id: graph.peak_dot_output_tiles
            for graph in self.non_reduction_graphs
        }
        best: tuple[LiveTile, ...] = ()
        best_key = (-1, -1)
        for graph_id, own in accumulator_of.items():
            chain = list(own)
            current = self.parent_of.get(graph_id, -1)
            seen = {graph_id}
            while current in accumulator_of and current not in seen:
                seen.add(current)
                chain.extend(accumulator_of[current])
                current = self.parent_of.get(current, -1)
            key = (
                sum(tile_rank(tile.dim_block_ids) for tile in chain),
                len(chain),
            )
            if key > best_key:
                best_key = key
                best = tuple(chain)
        return best

    def group_live_tiles(
        self,
        group_graph_ids: list[int],
    ) -> dict[int, list[tuple[int | None, ...]]]:
        """Resident peak-live tiles attributed to reduction co-residency groups."""
        group_axes = {
            graph_id: set(self.by_id[graph_id].reduction_occurrences)
            for graph_id in group_graph_ids
        }
        peak_of = {
            graph.graph_id: [tile.dim_block_ids for tile in graph.peak_live_tiles]
            for graph in self.non_reduction_graphs
        }

        def max_by_profile(
            lhs: list[tuple[int | None, ...]],
            rhs: list[tuple[int | None, ...]],
        ) -> list[tuple[int | None, ...]]:
            max_rank = max(
                (tile_rank(tile) for tile in lhs + rhs),
                default=0,
            )
            lhs_key = tile_set_rank_profile(lhs, max_rank)
            rhs_key = tile_set_rank_profile(rhs, max_rank)
            return lhs if lhs_key >= rhs_key else rhs

        group_keys = set(group_graph_ids)
        result: dict[int, list[tuple[int | None, ...]]] = {}
        for graph_id in group_graph_ids:
            axes = group_axes[graph_id]
            tiles = list(peak_of.get(graph_id, ()))
            seen_bodies = {graph_id}
            frontier = [graph_id]
            while frontier:
                current = frontier.pop()
                for body_id, block_ids in self.child_loops.get(current, ()):
                    if body_id in seen_bodies or body_id in group_keys:
                        continue
                    if not axes or (block_ids & axes):
                        seen_bodies.add(body_id)
                        tiles = max_by_profile(tiles, peak_of.get(body_id, []))
                        frontier.append(body_id)
            result[graph_id] = tiles
        return result
