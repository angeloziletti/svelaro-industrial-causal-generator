"""Variable-level ground truth for the benchmark.

The SCM graph holds scalar nodes (static batch conditions, generative
parameters, confounders). The variables a method sees are different: sensor
trajectories, (optionally) observed batch attributes, and the target
trajectory. This module builds the fine-grained generative graph over both,
groups it into variables, projects it onto the observed variables, and labels
every observed input by its causal role for the target.

A sensor VARIABLE is a GROUP of nodes: its generative parameters plus its
trajectory -- intervening on a sensor means setting its parameters or its
curve. The graph is kept fine-grained (parameters and trajectories as separate
nodes), so ancestry is exact: a sensor whose influence enters a truth through
a parameter the target never reads is NOT an ancestor of the target. (A first
version merged parameters into their sensor; the interventional audit showed
it labelled such sensors as causes with zero effect.)

Fine-grained graph:
    SCM graph (statics, parameters, confounders Z_k, ...)
    + parameter -> its sensor's trajectory   (kind: the parameter's kind)
    + truth trajectory -> target             (the target reads the truths' trajectories)
    + parent trajectory -> child trajectory  (dynamic edges)
    + Z_k(t) -> truth / shadow trajectories  (temporal confounder)
    + control loops: disturbance, actuator -> controlled; controlled -> actuator

Observable graph (projection onto observed variables, i.e. observed groups):
    u -> v        a path from any node of u to any node of v whose inner nodes
                  are hidden (in no observed group)
    u <-> v       one hidden node reaches both by hidden-only paths
                  (latent confounding; every node has its own noise)
Every directed edge carries `acts_on`, read from the edge(s) by which the path
enters v: "mean" (v's shape parameters), "kernel" (its GP texture
parameters), "both", "dynamic" (v follows u over time; combined e.g.
"mean+dynamic"), or "trajectory" (a function of u's trajectory, as truth ->
target or a control loop).

Target labels, for every observed input X:
    direct_parent      X -> target in the observable graph
    indirect_ancestor  X is an ancestor of the target, but not a direct parent
    non_cause          otherwise; `confounded_with_target` marks those sharing
                       an ancestor with the target (correlated non-causes)

Candidates. Control-loop actuators (valves) are known a priori -- controller
outputs on the P&ID -- so they stay in the data but are not candidate causes
(`candidate: false`). Scoring: direct parents among the candidates are
required; a `control_cancelled` ancestor (a disturbance the loop rejects) is
neutral -- a true cause that observational data need not reveal.
"""
from typing import Dict, List, Optional, Set

import networkx as nx

from advanced_scm.observability import hidden_scm_columns


def temporal_confounder_node(truth_sensor: str) -> str:
    return f"temporal_confounder_z_{truth_sensor}"


def _add_edge(G, u, v, kind=None):
    if not G.has_edge(u, v):
        G.add_edge(u, v, kinds=set())
    if kind:
        G.edges[u, v]["kinds"].add(kind)


def generative_graph(
    scm_graph: nx.DiGraph,
    node_types: Dict[str, str],
    gp_parameter_nodes: Dict[str, List[str]],
    causal_sensors: List[str],
    target: Optional[str] = "defect_ts",
    temporal_confounder_audit: Optional[Dict] = None,
    control_loops: Optional[List[Dict]] = None,
    dynamic_edges: Optional[List[Dict]] = None,
    consequences: Optional[List[Dict]] = None,
    campaign: Optional[Dict] = None,
) -> nx.DiGraph:
    """The fine-grained generative graph. Node attributes: `kind`, and `group`
    (the variable a node belongs to: a sensor's parameters and trajectory share
    the sensor's name)."""
    G = nx.DiGraph()
    owner = {p: s for s, params in gp_parameter_nodes.items() for p in params}
    for n, attrs in scm_graph.nodes(data=True):
        G.add_node(
            n, kind=node_types.get(n, "unknown"), group=owner.get(n, n),
            param_kind=attrs.get("param_kind"),
        )
    for a, b in scm_graph.edges:
        _add_edge(G, a, b)
    # a sensor's parameters shape its trajectory
    for sensor, params in gp_parameter_nodes.items():
        G.add_node(sensor, kind="sensor", group=sensor)
        for p in params:
            _add_edge(G, p, sensor, scm_graph.nodes[p].get("param_kind"))
    if target and causal_sensors:
        G.add_node(target, kind="target", group=target)
        for s in causal_sensors:
            _add_edge(G, s, target, "trajectory")
    audit = temporal_confounder_audit or {}
    if audit.get("enabled"):
        for truth in audit.get("per_truth", {}):
            z = temporal_confounder_node(truth)
            G.add_node(z, kind="temporal_confounder", group=z)
            _add_edge(G, z, truth, "trajectory")
        for shadow, info in audit.get("per_shadow", {}).items():
            _add_edge(G, temporal_confounder_node(info["shadow_of"]), shadow, "trajectory")
    for e in dynamic_edges or []:
        _add_edge(G, e["source"], e["target"], "dynamic")
    # raw-material lots and campaign drift (campaign.py): hidden, a parent of every sensor
    for name in ("lot", "drift"):
        if (campaign or {}).get(name):
            node = f"campaign_{name}"
            G.add_node(node, kind="campaign", group=node)
            for sensor in gp_parameter_nodes:
                _add_edge(G, node, sensor, "trajectory")
    if target and target in G:
        for c in consequences or []:
            _add_edge(G, target, c["sensor"], "trajectory")
    for loop in control_loops or []:
        c, d, a = loop["controlled"], loop["disturbance"], loop["actuator"]
        for u, v in [(d, c), (a, c)] + ([(c, a)] if loop["tightness"] > 0 else []):
            _add_edge(G, u, v, "trajectory")
    return G


def hidden_nodes(G: nx.DiGraph, observability: Optional[Dict] = None) -> Set[str]:
    """Nodes in no observed variable. Observed variables are the sensors, the
    target, and whatever SCM nodes the observability rule keeps visible. A
    sensor's (or the target's) own parameters belong to its observed group."""
    kinds = {n: d.get("kind") for n, d in G.nodes(data=True)}
    hidden_scm = set(hidden_scm_columns(kinds, observability))
    if not (observability or {}).get("confounders", False):
        hidden_scm |= {n for n, k in kinds.items() if k == "temporal_confounder"}
    hidden_scm |= {n for n, k in kinds.items() if k == "campaign"}  # lots and drift are never recorded
    hidden = set()
    for n, d in G.nodes(data=True):
        group = d.get("group", n)
        if group == n and n in hidden_scm:
            hidden.add(n)  # an unowned hidden node (Z_k, hidden static, ...)
    return hidden


def _group(G, n):
    return G.nodes[n].get("group", n)


def _walk(G: nx.DiGraph, starts, hidden: Set[str], own: Optional[str]):
    """From `starts`, follow edges through hidden nodes (and through nodes of
    `own`, the starting variable's own group). Returns (observed groups reached,
    entry edges: list of (x, y) whose head y is in a reached group)."""
    reached, entries, seen = set(), [], set()
    stack = [(s, c) for s in starts for c in G.successors(s)]
    while stack:
        x, y = stack.pop()
        if (x, y) in seen:
            continue
        seen.add((x, y))
        gy = _group(G, y)
        if y in hidden or (own is not None and gy == own):
            stack.extend((y, c) for c in G.successors(y))
        else:
            reached.add(gy)
            entries.append((x, y))
    return reached, entries


def _members(G) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for n in G.nodes:
        out.setdefault(_group(G, n), []).append(n)
    return out


def _acts_on(G, entries, v) -> str:
    kinds = set()
    for x, y in entries:
        if _group(G, y) != v:
            continue
        k = G.edges[x, y].get("kinds", set())
        if G.nodes[y].get("param_kind"):  # entered through one of v's parameters
            kinds.add(G.nodes[y]["param_kind"])
        else:
            kinds |= k
    names = []
    if "shape" in kinds and "kernel" in kinds:
        names.append("both")
    elif "shape" in kinds:
        names.append("mean")
    elif "kernel" in kinds:
        names.append("kernel")
    if "dynamic" in kinds:
        names.append("dynamic")
    return "+".join(names) or "trajectory"




def observed_groups(G: nx.DiGraph, hidden: Set[str]) -> List[str]:
    return sorted({_group(G, n) for n in G.nodes if n not in hidden})


def _projected(G: nx.DiGraph, hidden: Set[str]) -> Dict[str, Dict[str, str]]:
    members = _members(G)
    out = {}
    for u in observed_groups(G, hidden):
        reached, entries = _walk(G, members[u], hidden, own=u)
        out[u] = {v: _acts_on(G, entries, v) for v in reached if v != u}
    return out


def observable_projection(G: nx.DiGraph, hidden: Set[str]) -> Dict:
    proj = _projected(G, hidden)
    directed = sorted((u, v) for u, vs in proj.items() for v in vs)
    bidirected: Set[tuple] = set()
    for h in hidden:
        reached, _ = _walk(G, [h], hidden, own=None)
        reached = sorted(reached)
        for i, a in enumerate(reached):
            for b in reached[i + 1:]:
                bidirected.add((a, b))
    return {
        "nodes": observed_groups(G, hidden),
        "edges": [{"source": u, "target": v, "acts_on": proj[u][v]} for u, v in directed],
        "bidirected": [list(p) for p in sorted(bidirected)],
        "hidden_nodes": sorted(hidden),
        "n_nodes": len(proj),
        "n_edges": len(directed),
        "n_bidirected": len(bidirected),
        "note": (
            "Projection of the fine-grained generative graph onto the observed "
            "variables (a sensor = its parameters + its trajectory): u->v when a "
            "path from u's nodes to v's has only hidden nodes inside, u<->v when "
            "one hidden node reaches both by hidden-only paths."
        ),
    }


def acyclify(G: nx.DiGraph) -> nx.DiGraph:
    """The acyclification of a directed graph with cycles (Forre & Mooij 2017): d-separation
    on the result is sigma-separation on G, the separation criterion that holds for cyclic
    SCMs (e.g. a closed control loop). Within each strongly connected component (SCC) the
    directed edges are replaced by a shared latent parent for every pair of members; an
    edge i -> j entering an SCC becomes i -> k for every member k. Latent nodes are named
    __scc_<k>_<l> and are never conditioned on."""
    comp = {}
    for i, c in enumerate(nx.strongly_connected_components(G)):
        for n in c:
            comp[n] = i
    members = {}
    for n, i in comp.items():
        members.setdefault(i, []).append(n)
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes(data=True))
    for i, j in G.edges:
        if comp[i] == comp[j]:
            continue
        for k in members[comp[j]]:
            H.add_edge(i, k)
    for ms in members.values():
        ms = sorted(ms)
        for a in range(len(ms)):
            for b in range(a + 1, len(ms)):
                lat = f"__scc_{ms[a]}_{ms[b]}"
                H.add_edge(lat, ms[a])
                H.add_edge(lat, ms[b])
    return H


def with_measurement_layer(G: nx.DiGraph, noisy) -> nx.DiGraph:
    """The graph a method's conditioning actually sees when the sensors in `noisy` are read
    with measurement error: each such sensor's physical node becomes a hidden `<s>__true`
    (all its edges and its own parameters move to it) and the observed `<s>` is its noisy
    child. Conditioning on the reading then no longer blocks paths through the true value."""
    H = G.copy()
    for s in noisy:
        if s not in H:
            continue
        t = f"{s}__true"
        H.add_node(t, kind="sensor_true", group=t)
        for u, _, d in list(H.in_edges(s, data=True)):
            H.add_edge(u, t, **d)
            H.remove_edge(u, s)
        for _, v, d in list(H.out_edges(s, data=True)):
            H.add_edge(t, v, **d)
            H.remove_edge(s, v)
        for n, d in H.nodes(data=True):
            if n not in (s, t) and d.get("group") == s:
                d["group"] = t
        H.add_edge(t, s, kind="measurement")
    return H


def target_driver_labels(
    G: nx.DiGraph,
    hidden: Set[str],
    target: str,
    shadow_of: Optional[Dict] = None,
    control_loops: Optional[List[Dict]] = None,
    measurement_noisy=None,
    operator_reaction=None,
) -> Dict:
    """Role of every observed variable for `target`, and for every wrong answer
    its identification status (`identifiable`) from the observed sensors:

      conditioning   d-separated from the target given the true causes'
                     observed trajectories: a method can rule it out
      time_order     a consequence that lags the recorded target
      no             an open hidden path remains (a hidden confounder no observed
                     cause blocks): no method on the observed sensors can rule
                     it out; scored as a degradation test
    With cycles (a closed control loop) the separation test is sigma-separation, via
    acyclify(); `in_feedback_loop` marks variables inside a cycle. With `measurement_noisy`
    (sensors read with error) roles are unchanged, but the separation test runs on
    with_measurement_layer(): conditioning sees only the noisy readings. With
    `operator_reaction` ({"sensor": s}) the operator closes a loop target -> s. The roles
    read that edge (what s moves after it follows the target with a delay), but the
    separation test does not: the operator reacts to the target's past, so in time the loop
    has no cycle, and a method that conditions on the causes and on the target's own past
    blocks every path through it. The summary-graph cycle would instead tie every ancestor
    and sibling of s to the target (sigma-separation is conservative for delayed loops).
    """
    if target not in G:
        return {}
    G_sep = G  # the separation test runs without the operator edge (see the docstring)
    if operator_reaction and operator_reaction.get("sensor") in G:
        G = G.copy()
        G.add_edge(target, operator_reaction["sensor"], kind="operator")
    actuators = {loop["actuator"] for loop in control_loops or []}
    proj = _projected(G, hidden)
    members = _members(G)
    # Ancestry on the FINE-GRAINED graph: X causes the target iff one of X's
    # nodes has a directed path to one of the target's nodes. Chaining
    # group-level edges (u -> t, t -> target) would label u a cause even when
    # u only touches a part of t the target never reads.
    tgt_anc = set().union(*(nx.ancestors(G, n) for n in members[target]))
    ancestors = {g for g in proj if g != target and any(n in tgt_anc for n in members[g])}
    tgt_desc = set().union(*(nx.descendants(G, n) for n in members[target]))
    labels = {}
    for x in sorted(g for g in proj if g != target):
        if target in proj[x]:
            label = "direct_parent"
        elif x in ancestors:
            label = "indirect_ancestor"
        elif any(n in tgt_desc for n in members[x]):
            # downstream of the target: moves with it, causes nothing
            label = "consequence"
        else:
            label = "non_cause"
        entry = {"label": label, "kind": G.nodes[x].get("kind"), "candidate": x not in actuators}
        if label == "non_cause":
            x_anc = set().union(*(nx.ancestors(G, n) | {n} for n in members[x]))
            entry["confounded_with_target"] = bool(x_anc & tgt_anc)
            if shadow_of and x in shadow_of:
                entry["shadow_of"] = shadow_of[x]
        for loop in control_loops or []:
            for role in ("controlled", "disturbance", "actuator"):
                if loop[role] == x:
                    entry["control_loop"] = {"role": role, "tightness": loop["tightness"]}
                    if role == "disturbance" and loop["tightness"] > 0:
                        entry["control_cancelled"] = True
        labels[x] = entry
    # Identification status of every wrong answer (see the docstring).
    causes = [x for x, e in labels.items() if e["label"] == "direct_parent"]
    Gs = with_measurement_layer(G_sep, set(measurement_noisy or [])) if measurement_noisy else G_sep
    members_s = _members(Gs)
    observed_causes = {n for c in causes for n in members_s[c] if n not in hidden and Gs.nodes[n].get("kind") == "sensor"}
    dsep = getattr(nx, "is_d_separator", None) or getattr(nx, "d_separated")
    H = Gs if nx.is_directed_acyclic_graph(Gs) else acyclify(Gs)
    if measurement_noisy:
        members_0 = _members(G_sep)
        observed_exact = {n for c in causes for n in members_0[c] if n not in hidden and G_sep.nodes[n].get("kind") == "sensor"}
        H0 = G_sep if nx.is_directed_acyclic_graph(G_sep) else acyclify(G_sep)
    in_cycle = {n for c in nx.strongly_connected_components(G) if len(c) > 1 for n in c}
    for x, entry in labels.items():
        if entry["label"] == "direct_parent":
            continue
        if any(n in in_cycle for n in members[x]):
            entry["in_feedback_loop"] = True
        if entry["label"] == "consequence":
            entry["identifiable"] = "time_order"
        else:
            xs = {n for n in members_s[x] if n not in observed_causes}
            separated = bool(xs) and dsep(H, xs, {target}, observed_causes - xs)
            entry["identifiable"] = "conditioning" if separated else "no"
            if measurement_noisy:
                # the same question with exact readings, so a score can separate what the
                # measurement error alone makes unanswerable
                xs0 = {n for n in members_0[x] if n not in observed_exact}
                sep0 = bool(xs0) and dsep(H0, xs0, {target}, observed_exact - xs0)
                entry["identifiable_if_exact"] = "conditioning" if sep0 else "no"
    counts: Dict[str, int] = {}
    for e in labels.values():
        if e.get("identifiable"):
            k = f"identifiable_{e['identifiable']}"
            counts[k] = counts.get(k, 0) + 1
        key = e["label"]
        if key == "non_cause" and e["confounded_with_target"]:
            key = "non_cause_confounded"
        counts[key] = counts.get(key, 0) + 1
    return {
        "target": target,
        "labels": labels,
        "counts": counts,
        "candidate_causes": sorted(x for x, e in labels.items() if e["candidate"]),
    }
