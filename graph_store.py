"""Persistent knowledge graph over NetworkX.

Cherry-picked from AYON-ARYAN/graph-rag (`GraphRAGEngine`): the typed-node
DiGraph, per-entity-type colour assignment, BFS traversal that renders
"A --[rel]--> B" context lines, the D3 serialisation, and the stats payload.

Changed from upstream: the graph there lives only in `self.graph` and is lost
on restart, which is fine when you upload a document then immediately query
it, but not for a corpus-scale graph that takes minutes to build. This version
persists to disk as JSON node-link data and reloads on construction.
"""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import networkx as nx

import config

# Upstream's palette, kept so the visualisation looks the same.
TYPE_COLORS = [
    "#00f0ff", "#a855f7", "#f43f5e", "#22d3ee",
    "#facc15", "#4ade80", "#f97316", "#818cf8",
    "#e879f9", "#2dd4bf", "#fb923c", "#38bdf8",
]


class GraphStore:
    """A typed, coloured, persistent directed knowledge graph."""

    def __init__(self, path: str | None = None, autoload: bool = True):
        self.path = Path(path or config.GRAPH_PATH)
        self.graph = nx.DiGraph()
        self.entity_types: Dict[str, str] = {}   # type -> colour
        self._type_index = 0
        self.documents: List[str] = []
        self._lock = threading.Lock()
        if autoload:
            self.load()

    # ----------------------------------------------------------- colours ---
    def _color_for(self, entity_type: str) -> str:
        if entity_type not in self.entity_types:
            self.entity_types[entity_type] = TYPE_COLORS[
                self._type_index % len(TYPE_COLORS)
            ]
            self._type_index += 1
        return self.entity_types[entity_type]

    # ------------------------------------------------------------- build ---
    def add_triples(self, triples: Iterable[Dict[str, str]],
                    source: str | None = None) -> int:
        """Add triples to the graph. Returns the number actually added.

        Repeated edges accumulate a `weight`, which doubles as a confidence
        signal: a relation asserted by many chunks is more trustworthy than
        one seen once.
        """
        added = 0
        with self._lock:
            for t in triples:
                head = (t.get("head") or "").strip()
                tail = (t.get("tail") or "").strip()
                relation = (t.get("relation") or "").strip()
                if not (head and tail and relation):
                    continue
                head_type = (t.get("head_type") or "Entity").strip() or "Entity"
                tail_type = (t.get("tail_type") or "Entity").strip() or "Entity"

                for name, ntype in ((head, head_type), (tail, tail_type)):
                    if not self.graph.has_node(name):
                        self.graph.add_node(name, type=ntype,
                                            color=self._color_for(ntype),
                                            sources=[])
                    if source:
                        srcs = self.graph.nodes[name].setdefault("sources", [])
                        if source not in srcs:
                            srcs.append(source)

                if self.graph.has_edge(head, tail):
                    e = self.graph.edges[head, tail]
                    e["weight"] = e.get("weight", 1) + 1
                    if relation not in e.get("relations", [e.get("relation")]):
                        e.setdefault("relations", [e.get("relation")]).append(relation)
                else:
                    self.graph.add_edge(head, tail, relation=relation,
                                        relations=[relation], weight=1)
                added += 1
        return added

    def note_document(self, name: str) -> None:
        with self._lock:
            if name not in self.documents:
                self.documents.append(name)

    # ---------------------------------------------------------- traverse ---
    def find_nodes(self, entity: str, limit: int = 8) -> List[str]:
        """Match a lookup key to graph nodes, exact matches first.

        Partial matches are word-boundary anchored, not bare substring: plain
        `in` made "grounds" match "Recreation Grounds Co Ltd vs Taylor", and a
        single generic word was enough to fill the highlight with unrelated
        cases. Very short keys are required to match exactly.
        """
        if not entity:
            return []
        low = entity.lower().strip()
        if not low:
            return []

        exact = [n for n in self.graph.nodes if n.lower() == low]
        if len(low) < 4:
            return exact[:limit]

        pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(low)}(?![A-Za-z0-9])")
        partial = [n for n in self.graph.nodes
                   if n.lower() != low and pattern.search(n.lower())]
        partial.sort(key=lambda n: -self.graph.degree(n))
        return (exact + partial)[:limit]

    def traverse(self, entities: Sequence[str], depth: int | None = None,
                 max_relations: int | None = None,
                 hub_degree: int | None = None) -> List[str]:
        """BFS out from matched entities, rendering edges as context lines.

        Two things matter at corpus scale and neither is handled by a naive
        BFS:

        * **Hubs.** "Constitution of India" has ~575 incident edges. Expanding
          *through* it from "Article 21" reaches every other article, so the
          returned context becomes a wall of `Article N --[part_of]-->
          Constitution of India` and the case law that answers the question is
          pushed out. Hub nodes are therefore included as endpoints but never
          expanded through.
        * **Ordering.** Edges one hop from a seed are far more relevant than
          edges two hops out, so results are emitted depth-first-by-level and
          truncation drops the distant ones rather than arbitrary ones.

        Upstream recomputed `shortest_path` per neighbour inside a loop over
        depths, re-walking the same edges repeatedly; this visits each edge
        once.
        """
        depth = depth or config.GRAPH_DEPTH
        max_relations = max_relations or config.GRAPH_MAX_RELATIONS
        hub_degree = hub_degree or config.GRAPH_HUB_DEGREE

        seeds: List[str] = []
        for entity in entities:
            for n in self.find_nodes(entity):
                if n not in seeds:
                    seeds.append(n)
        if not seeds:
            return []

        und = self.graph.to_undirected(as_view=True)
        by_level: List[List[str]] = [[] for _ in range(depth + 1)]
        seen_edges: set[tuple] = set()
        visited: set[str] = set(seeds)
        frontier = list(seeds)

        for level in range(1, depth + 1):
            nxt: List[str] = []
            for node in frontier:
                for nbr in und.neighbors(node):
                    for a, b in ((node, nbr), (nbr, node)):
                        if not self.graph.has_edge(a, b) or (a, b) in seen_edges:
                            continue
                        seen_edges.add((a, b))
                        rel = self.graph.edges[a, b].get("relation", "related to")
                        by_level[level].append(f"{a} --[{rel}]--> {b}")
                    # Include the hub as an endpoint, but do not walk past it.
                    if nbr not in visited and self.graph.degree(nbr) <= hub_degree:
                        visited.add(nbr)
                        nxt.append(nbr)
            frontier = nxt
            if not frontier:
                break

        out: List[str] = []
        for level in range(1, depth + 1):
            for line in by_level[level]:
                out.append(line)
                if len(out) >= max_relations:
                    return out
        return out

    def subgraph_nodes(self, entities: Sequence[str],
                       limit: int | None = None,
                       hub_degree: int | None = None) -> List[str]:
        """Nodes to highlight for a query: matches plus their neighbours.

        Hubs are included but not expanded, for the same reason traversal
        blocks them: pulling in every neighbour of "Constitution of India"
        highlighted 798 nodes, which lights up essentially the whole rendered
        graph and tells the viewer nothing. Capped so the highlight stays a
        readable subgraph.
        """
        limit = limit or config.GRAPH_HIGHLIGHT_MAX
        hub_degree = hub_degree or config.GRAPH_HUB_DEGREE
        hi: List[str] = []
        seen: set[str] = set()

        def push(n: str) -> bool:
            if n in seen:
                return True
            seen.add(n)
            hi.append(n)
            return len(hi) < limit

        matched = [n for e in entities for n in self.find_nodes(e)]
        for n in matched:
            if not push(n):
                return hi
        for n in matched:
            if self.graph.degree(n) > hub_degree:
                continue   # hub: keep it lit, but do not drag its whole star in
            for nbr in list(self.graph.successors(n)) + list(self.graph.predecessors(n)):
                if not push(nbr):
                    return hi
        return hi

    # --------------------------------------------------------- serialise ---
    def get_graph_data(self, max_nodes: int | None = None,
                       seeds: Sequence[str] | None = None) -> Dict:
        """D3-ready nodes/edges, capped to a *connected* subgraph.

        Taking simply the top-N nodes by degree produces a near-edgeless
        picture: the hubs are rarely adjacent to one another, and all their
        neighbours fall outside the cap, so the force layout renders scattered
        dots. Instead seed with the highest-degree nodes (or explicit `seeds`)
        and BFS-expand until the node budget is filled, which keeps the
        induced subgraph connected and actually readable.
        """
        max_nodes = max_nodes or config.GRAPH_VIZ_MAX_NODES
        und = self.graph.to_undirected(as_view=True)

        if seeds:
            frontier = [n for s in seeds for n in self.find_nodes(s)]
        else:
            frontier = sorted(self.graph.nodes,
                              key=lambda n: -self.graph.degree(n))[:max(1, max_nodes // 20)]

        keep: set[str] = set()
        queue = list(frontier)
        qi = 0
        while queue and len(keep) < max_nodes:
            node = queue[qi] if qi < len(queue) else None
            if node is None:
                break
            qi += 1
            if node not in keep:
                keep.add(node)
                # Pull in the best-connected neighbours first.
                nbrs = sorted(und.neighbors(node), key=lambda n: -self.graph.degree(n))
                queue.extend(n for n in nbrs[:12] if n not in keep)
            if qi >= len(queue):
                # Exhausted this component: start another from the next hub.
                remaining = [n for n in sorted(self.graph.nodes,
                                               key=lambda n: -self.graph.degree(n))
                             if n not in keep]
                if not remaining:
                    break
                queue.append(remaining[0])

        nodes = [
            {
                "id": n,
                "type": self.graph.nodes[n].get("type", "Entity"),
                "color": self.graph.nodes[n].get("color", "#00f0ff"),
                "connections": self.graph.degree(n),
            }
            for n in keep
        ]
        edges = [
            {"source": u, "target": v,
             "relation": d.get("relation", "related"),
             "weight": d.get("weight", 1)}
            for u, v, d in self.graph.edges(data=True)
            if u in keep and v in keep
        ]
        return {"nodes": nodes, "edges": edges,
                "truncated": self.graph.number_of_nodes() > len(keep),
                "total_nodes": self.graph.number_of_nodes()}

    def get_stats(self) -> Dict:
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "total_documents": len(self.documents),
            "entity_types": dict(self.entity_types),
            "documents": self.documents[:50],
            "type_counts": self._type_counts(),
        }

    def _type_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for _, data in self.graph.nodes(data=True):
            t = data.get("type", "Entity")
            counts[t] = counts.get(t, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    # --------------------------------------------------------- persistence --
    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "graph": nx.node_link_data(self.graph, edges="links"),
            "entity_types": self.entity_types,
            "type_index": self._type_index,
            "documents": self.documents,
        }
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
        os.replace(tmp, self.path)   # atomic: never leave a half-written graph

    def load(self) -> bool:
        if not self.path.is_file():
            return False
        try:
            with self.path.open(encoding="utf-8") as fh:
                payload = json.load(fh)
            self.graph = nx.node_link_graph(
                payload["graph"], directed=True, edges="links")
            self.entity_types = payload.get("entity_types", {})
            self._type_index = payload.get("type_index", len(self.entity_types))
            self.documents = payload.get("documents", [])
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[graph_store] could not load {self.path}: {exc}")
            return False

    def clear(self) -> None:
        with self._lock:
            self.graph = nx.DiGraph()
            self.entity_types = {}
            self._type_index = 0
            self.documents = []
