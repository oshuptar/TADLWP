# draw_architecture_helper.py

import torch
import torch.nn as nn
import torch.fx as fx
from graphviz import Digraph


def draw_filtered_fx_graph(gm, output_file="fx_graph", fmt="svg"):
    kept_ops = {"call_module", "call_function"}
    dot = Digraph(name="FXGraph")
    dot.attr(rankdir="LR")
    dot.attr("node", shape="box", fontsize="10")

    kept_nodes = [n for n in gm.graph.nodes if n.op in kept_ops]
    kept_names = {n.name for n in kept_nodes}

    def find_kept_parents(node, visited=None):
        """
        Walk backward through input nodes until we reach kept nodes.
        This preserves connectivity even when placeholder/call_method/output
        nodes are omitted.
        """
        if visited is None:
            visited = set()
        if node in visited:
            return set()
        visited.add(node)

        if node.op in kept_ops:
            return {node}

        parents = set()
        for inp in node.all_input_nodes:
            parents |= find_kept_parents(inp, visited.copy())
        return parents

    def short_label(node):
        if node.op == "call_module":
            return f"module\\n{node.target}"
        if node.op == "call_function":
            name = getattr(node.target, "__name__", str(node.target))
            return f"function\\n{name}"
        return f"{node.op}\\n{node.target}"

    # add only kept nodes
    for node in kept_nodes:
        dot.node(node.name, short_label(node))

    # connect each kept node to the nearest previous kept nodes
    for node in kept_nodes:
        direct_inputs = list(node.all_input_nodes)
        parent_kept = set()

        for inp in direct_inputs:
            parent_kept |= find_kept_parents(inp)

        for parent in parent_kept:
            if parent.name in kept_names and parent.name != node.name:
                dot.edge(parent.name, node.name)

    dot.render(output_file, format=fmt, cleanup=True)
    return dot


def trace_model_with_fx(model: nn.Module) -> fx.GraphModule:
    """
    Convert a regular PyTorch model into a torch.fx GraphModule.
    """
    model.eval()
    gm = fx.symbolic_trace(model)
    return gm


def print_and_draw_model_structure(model: nn.Module, output_file="fx_graph", fmt="svg"):
    """
    1. Trace model with torch.fx
    2. Print readable layer structure
    3. Draw filtered graph with Graphviz
    """
    gm = trace_model_with_fx(model)

    print("=" * 60)
    print("FX GRAPH NODES")
    print("=" * 60)
    for node in gm.graph.nodes:
        print(f"name={node.name:20} op={node.op:15} target={node.target}")

    print("\n" + "=" * 60)
    print("MODEL LAYERS")
    print("=" * 60)
    for name, module in gm.named_modules():
        if name == "":
            continue
        print(f"{name}: {module.__class__.__name__}")

    print("\n" + "=" * 60)
    print("TABULAR FX GRAPH")
    print("=" * 60)
    try:
        gm.graph.print_tabular()
    except Exception:
        print(gm.graph)

    draw_filtered_fx_graph(gm, output_file=output_file, fmt=fmt)
    print(f"\nSaved graph to: {output_file}.{fmt}")

    return gm