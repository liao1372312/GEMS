"""Visualization exporters for retrieved GEMS subgraphs."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx

from .graph_memory import Edge, ExecutionGraphMemory, Node
from .retrieval import RetrievedEvidence


NODE_COLORS = {
    "request": "#4C78A8",
    "subtask": "#72B7B2",
    "api": "#F58518",
    "parameter": "#54A24B",
    "schema": "#B279A2",
    "qos": "#EECA3B",
    "execution": "#9D755D",
    "failure": "#E45756",
    "repair": "#59A14F",
    "outcome": "#8CD17D",
}

NODE_SHAPES = {
    "request": "o",
    "subtask": "s",
    "api": "D",
    "parameter": "v",
    "schema": "^",
    "qos": "h",
    "execution": "p",
    "failure": "X",
    "repair": "P",
    "outcome": "8",
}


def build_retrieved_nx_graph(
    memory: ExecutionGraphMemory,
    evidence: RetrievedEvidence,
    include_neighbor_edges: bool = True,
    context_per_node: int = 4,
) -> nx.DiGraph:
    graph = nx.DiGraph()
    selected = set(evidence.node_ids)
    if include_neighbor_edges:
        for node_id in list(selected):
            neighbors = list(memory.neighbors(node_id))
            neighbors.sort(key=lambda item: _context_priority(memory.nodes[item[0]], item[1]))
            for neighbor_id, _ in neighbors[:context_per_node]:
                if neighbor_id in memory.nodes:
                    selected.add(neighbor_id)

    for node_id in selected:
        node = memory.nodes[node_id]
        graph.add_node(
            node_id,
            label=node_label(node),
            node_type=node.node_type,
            reliability=round(node.reliability, 4),
            risk=round(node.risk, 4),
            conflict=round(node.conflict, 4),
            score=round(evidence.scores.get(node_id, 0.0), 4),
            description=node.desc[:600],
        )
    for edge in memory.edges:
        if edge.source in selected and edge.target in selected:
            graph.add_edge(
                edge.source,
                edge.target,
                relation=edge.edge_type,
                reliability=round(edge.reliability, 4),
                attrs=json.dumps(edge.attrs, ensure_ascii=False),
            )
    return graph


def _context_priority(node: Node, edge: Edge) -> tuple[int, float]:
    type_rank = {
        "qos": 0,
        "schema": 1,
        "execution": 2,
        "outcome": 3,
        "failure": 4,
        "parameter": 5,
        "subtask": 6,
        "request": 7,
        "api": 8,
        "repair": 9,
    }
    edge_rank = {
        "produces": 0,
        "depends_on": 1,
        "causes": 2,
        "requires": 3,
        "binds": 4,
        "selects": 5,
    }
    return (edge_rank.get(edge.edge_type, 9) + type_rank.get(node.node_type, 9), -node.reliability)


def node_label(node: Node) -> str:
    attrs = node.attrs
    for key in ("endpoint_name", "service_name", "name", "category"):
        if attrs.get(key):
            return str(attrs[key])[:42]
    if node.node_id.startswith("api:endpoint:") and attrs.get("endpoint_id"):
        return f"API {attrs['endpoint_id'][:8]}"
    return node.node_id.split(":", 1)[-1][:42]


def export_graphml(graph: nx.DiGraph, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, path)


def export_png(graph: nx.DiGraph, path: str | Path, title: str = "") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(14, 9))
    pos = nx.spring_layout(graph, seed=42, k=1.2)
    for node_type, color in NODE_COLORS.items():
        nodes = [node for node, data in graph.nodes(data=True) if data.get("node_type") == node_type]
        if not nodes:
            continue
        sizes = [
            450 + 850 * float(graph.nodes[node].get("reliability", 0.5))
            for node in nodes
        ]
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=nodes,
            node_color=color,
            node_shape=NODE_SHAPES.get(node_type, "o"),
            node_size=sizes,
            alpha=0.88,
            linewidths=0.8,
            edgecolors="#2F3437",
            label=node_type,
        )
    nx.draw_networkx_edges(
        graph,
        pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=12,
        edge_color="#6B7280",
        width=1.1,
        alpha=0.65,
        connectionstyle="arc3,rad=0.05",
    )
    labels = {node: str(data.get("label", node)) for node, data in graph.nodes(data=True)}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8, font_color="#111827")
    edge_labels = {
        (source, target): data.get("relation", "")
        for source, target, data in graph.edges(data=True)
    }
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=7, font_color="#374151")
    if title:
        plt.title(title, fontsize=13)
    plt.axis("off")
    plt.legend(scatterpoints=1, fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def export_html(graph: nx.DiGraph, path: str | Path, title: str, metadata: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    positions = _circle_layout(graph)
    svg_nodes = []
    svg_edges = []
    for source, target, data in graph.edges(data=True):
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        sx, sy, tx, ty = _shorten_line(x1, y1, x2, y2, 24)
        label_x = (sx + tx) / 2
        label_y = (sy + ty) / 2
        relation = html.escape(str(data.get("relation", "")))
        svg_edges.append(
            f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{tx:.1f}" y2="{ty:.1f}" '
            'stroke="#6B7280" stroke-width="1.4" marker-end="url(#arrow)" />'
            f'<text x="{label_x:.1f}" y="{label_y:.1f}" class="edge-label">{relation}</text>'
        )
    for node_id, data in graph.nodes(data=True):
        x, y = positions[node_id]
        node_type = str(data.get("node_type", "api"))
        color = NODE_COLORS.get(node_type, "#9CA3AF")
        radius = 16 + 13 * float(data.get("reliability", 0.5))
        label = html.escape(str(data.get("label", node_id)))
        tooltip = html.escape(
            f"{node_id}\n"
            f"type={node_type}\n"
            f"score={data.get('score')}\n"
            f"reliability={data.get('reliability')}\n"
            f"risk={data.get('risk')}\n"
            f"{data.get('description', '')}"
        )
        svg_nodes.append(
            f'<g class="node" data-type="{html.escape(node_type)}">'
            f'<title>{tooltip}</title>'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{color}" />'
            f'<text x="{x:.1f}" y="{y + radius + 14:.1f}" class="node-label">{label}</text>'
            f'<text x="{x:.1f}" y="{y + 4:.1f}" class="node-type">{html.escape(node_type)}</text>'
            '</g>'
        )

    legend = "\n".join(
        f'<span><i style="background:{color}"></i>{html.escape(node_type)}</span>'
        for node_type, color in NODE_COLORS.items()
        if any(data.get("node_type") == node_type for _, data in graph.nodes(data=True))
    )
    meta_rows = "\n".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in metadata.items()
    )
    content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Arial, Helvetica, sans-serif;
      background: #F8FAFC;
      color: #111827;
    }}
    body {{ margin: 0; }}
    header {{
      padding: 18px 24px 12px;
      border-bottom: 1px solid #D1D5DB;
      background: #FFFFFF;
    }}
    h1 {{ margin: 0 0 8px; font-size: 22px; font-weight: 700; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 10px 16px; font-size: 13px; color: #374151; }}
    .legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
    .legend i {{ width: 12px; height: 12px; border-radius: 50%; border: 1px solid #374151; }}
    main {{ display: grid; grid-template-columns: minmax(0, 1fr) 330px; min-height: calc(100vh - 86px); }}
    .canvas {{ overflow: auto; padding: 16px; }}
    aside {{
      border-left: 1px solid #D1D5DB;
      background: #FFFFFF;
      padding: 16px;
      overflow: auto;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 7px 8px; border-bottom: 1px solid #E5E7EB; text-align: left; vertical-align: top; }}
    th {{ width: 110px; color: #374151; }}
    svg {{
      width: 1200px;
      height: 780px;
      background: #FFFFFF;
      border: 1px solid #D1D5DB;
      box-shadow: 0 1px 4px rgba(15, 23, 42, 0.08);
    }}
    .node circle {{ stroke: #1F2937; stroke-width: 1.1; }}
    .node-label {{ text-anchor: middle; font-size: 12px; fill: #111827; }}
    .node-type {{ text-anchor: middle; font-size: 10px; fill: #111827; font-weight: 700; pointer-events: none; }}
    .edge-label {{
      text-anchor: middle;
      font-size: 10px;
      fill: #374151;
      paint-order: stroke;
      stroke: #FFFFFF;
      stroke-width: 3px;
      stroke-linejoin: round;
    }}
    @media (max-width: 900px) {{
      main {{ grid-template-columns: 1fr; }}
      aside {{ border-left: 0; border-top: 1px solid #D1D5DB; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <div class="legend">{legend}</div>
  </header>
  <main>
    <section class="canvas">
      <svg viewBox="0 0 1200 780" role="img" aria-label="{html.escape(title)}">
        <defs>
          <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L0,6 L9,3 z" fill="#6B7280" />
          </marker>
        </defs>
        {''.join(svg_edges)}
        {''.join(svg_nodes)}
      </svg>
    </section>
    <aside>
      <table>{meta_rows}</table>
    </aside>
  </main>
</body>
</html>
"""
    path.write_text(content, encoding="utf-8")


def _circle_layout(graph: nx.DiGraph) -> dict[str, tuple[float, float]]:
    nodes = list(graph.nodes())
    center_x, center_y = 600.0, 390.0
    if not nodes:
        return {}
    if len(nodes) == 1:
        return {nodes[0]: (center_x, center_y)}
    radius_x = 430.0
    radius_y = 260.0
    result = {}
    for idx, node in enumerate(nodes):
        angle = 2.0 * math.pi * idx / len(nodes) - math.pi / 2.0
        result[node] = (center_x + radius_x * math.cos(angle), center_y + radius_y * math.sin(angle))
    return result


def _shorten_line(x1: float, y1: float, x2: float, y2: float, margin: float) -> tuple[float, float, float, float]:
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        return x1, y1, x2, y2
    ux = dx / length
    uy = dy / length
    return x1 + ux * margin, y1 + uy * margin, x2 - ux * margin, y2 - uy * margin
