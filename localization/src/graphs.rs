//! Stable graphs for the boundary strata of `Mbar_{g,n}`.
//!
//! These are the abstract stable-curve graphs used by the Givental graph
//! expansion, not stable-map localization graphs.  Vertices carry genera, legs
//! are labelled markings, and automorphism orders include both vertex
//! symmetries and bijections of repeated edges.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Mutex, OnceLock};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StableVertex {
    pub genus: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StableEdge {
    pub a: usize,
    pub b: usize,
}

impl StableEdge {
    pub fn new(a: usize, b: usize) -> Self {
        if a <= b {
            Self { a, b }
        } else {
            Self { a: b, b: a }
        }
    }

    pub fn is_loop(&self) -> bool {
        self.a == self.b
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StableGraph {
    pub vertices: Vec<StableVertex>,
    pub edges: Vec<StableEdge>,
    /// `legs[marking] = vertex`.
    pub legs: Vec<usize>,
}

impl StableGraph {
    pub fn first_betti(&self) -> usize {
        if self.vertices.is_empty() {
            0
        } else {
            self.edges.len() + 1 - self.vertices.len()
        }
    }

    pub fn genus(&self) -> usize {
        self.vertices.iter().map(|v| v.genus).sum::<usize>() + self.first_betti()
    }

    pub fn valence(&self, vertex: usize) -> usize {
        let leg_count = self.legs.iter().filter(|&&v| v == vertex).count();
        let edge_count = self
            .edges
            .iter()
            .map(|edge| {
                if edge.a == vertex && edge.b == vertex {
                    2
                } else if edge.a == vertex || edge.b == vertex {
                    1
                } else {
                    0
                }
            })
            .sum::<usize>();
        leg_count + edge_count
    }

    pub fn is_connected(&self) -> bool {
        if self.vertices.is_empty() {
            return false;
        }
        if self.vertices.len() == 1 {
            return true;
        }
        let mut dsu = DisjointSet::new(self.vertices.len());
        for edge in &self.edges {
            if !edge.is_loop() {
                dsu.union(edge.a, edge.b);
            }
        }
        let root = dsu.find(0);
        (1..self.vertices.len()).all(|v| dsu.find(v) == root)
    }

    pub fn is_stable(&self) -> bool {
        self.vertices
            .iter()
            .enumerate()
            .all(|(idx, vertex)| 2 * vertex.genus + self.valence(idx) > 2)
    }

    pub fn canonical_label(&self) -> String {
        let vertex_count = self.vertices.len();
        let mut best = None::<String>;
        for permutation in permutations(vertex_count) {
            let label = self.label_with_permutation(&permutation);
            if best.as_ref().map_or(true, |current| label < *current) {
                best = Some(label);
            }
        }
        best.unwrap_or_default()
    }

    pub fn automorphism_order(&self) -> usize {
        let vertex_count = self.vertices.len();
        let base_label = self.label_with_permutation(&(0..vertex_count).collect::<Vec<_>>());
        let mut total = 0usize;
        for permutation in permutations(vertex_count) {
            if self.label_with_permutation(&permutation) == base_label {
                total += edge_bijection_count_after_permutation(self, &permutation);
            }
        }
        total
    }

    pub fn vertex_automorphism_permutations(&self) -> Vec<Vec<usize>> {
        let vertex_count = self.vertices.len();
        let base_label = self.label_with_permutation(&(0..vertex_count).collect::<Vec<_>>());
        permutations(vertex_count)
            .into_iter()
            .filter(|permutation| self.label_with_permutation(permutation) == base_label)
            .collect()
    }

    fn label_with_permutation(&self, permutation: &[usize]) -> String {
        let mut inverse = vec![0usize; permutation.len()];
        for (new, &old) in permutation.iter().enumerate() {
            inverse[old] = new;
        }

        let genera = permutation
            .iter()
            .map(|&old| self.vertices[old].genus.to_string())
            .collect::<Vec<_>>()
            .join(",");

        let mut edges = self
            .edges
            .iter()
            .map(|edge| StableEdge::new(inverse[edge.a], inverse[edge.b]))
            .collect::<Vec<_>>();
        edges.sort_by_key(|edge| (edge.a, edge.b));
        let edge_label = edges
            .iter()
            .map(|edge| format!("{}-{}", edge.a, edge.b))
            .collect::<Vec<_>>()
            .join(",");

        let legs = self
            .legs
            .iter()
            .map(|&old| inverse[old].to_string())
            .collect::<Vec<_>>()
            .join(",");

        format!("g:{genera}|e:{edge_label}|l:{legs}")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StableGraphBounds {
    pub max_vertices: usize,
}

pub fn stable_graphs(genus: usize, legs: usize) -> Vec<StableGraph> {
    // The universal Givental sum only needs stable graphs for fixed (g,n), so
    // cache them independently of target, degree, and insertions.
    static CACHE: OnceLock<Mutex<BTreeMap<(usize, usize), Vec<StableGraph>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(BTreeMap::new()));
    if let Some(graphs) = cache.lock().unwrap().get(&(genus, legs)).cloned() {
        return graphs;
    }

    let stable_complexity = 2isize * genus as isize - 2 + legs as isize;
    let max_vertices = if stable_complexity > 0 {
        stable_complexity as usize
    } else {
        1
    };
    let graphs = stable_graphs_with_bounds(
        genus,
        legs,
        StableGraphBounds {
            max_vertices: max_vertices.max(1),
        },
    );
    cache.lock().unwrap().insert((genus, legs), graphs.clone());
    graphs
}

pub fn stable_graphs_with_bounds(
    genus: usize,
    legs: usize,
    bounds: StableGraphBounds,
) -> Vec<StableGraph> {
    // Generate connected multigraphs, assign legs and vertex genera, then
    // canonicalize to quotient by graph isomorphism.  This is exact and simple;
    // performance-sensitive callers cache the result or precompute color orbits.
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();

    for vertex_count in 1..=bounds.max_vertices.max(1) {
        let max_edges = genus + vertex_count - 1;
        let pair_types = edge_pair_types(vertex_count);
        let leg_assignments = labelled_leg_assignments(legs, vertex_count);
        for edge_count in 0..=max_edges {
            if edge_count + 1 < vertex_count {
                continue;
            }
            let h1 = edge_count + 1 - vertex_count;
            if h1 > genus {
                continue;
            }
            for_each_edge_multiset(&pair_types, edge_count, |edges| {
                for leg_assignment in &leg_assignments {
                    let base = StableGraph {
                        vertices: vec![StableVertex { genus: 0 }; vertex_count],
                        edges: edges.to_vec(),
                        legs: leg_assignment.clone(),
                    };
                    if !base.is_connected() {
                        continue;
                    }
                    let genus_sum = genus - h1;
                    for genera in compositions(genus_sum, vertex_count) {
                        let mut graph = base.clone();
                        for (vertex, gv) in graph.vertices.iter_mut().zip(genera) {
                            vertex.genus = gv;
                        }
                        if graph.is_stable() {
                            let label = graph.canonical_label();
                            if seen.insert(label) {
                                out.push(graph);
                            }
                        }
                    }
                }
            });
        }
    }

    out
}

fn edge_pair_types(vertex_count: usize) -> Vec<StableEdge> {
    let mut pairs = Vec::new();
    for a in 0..vertex_count {
        for b in a..vertex_count {
            pairs.push(StableEdge::new(a, b));
        }
    }
    pairs
}

fn for_each_edge_multiset(
    pair_types: &[StableEdge],
    edge_count: usize,
    mut visit: impl FnMut(&[StableEdge]),
) {
    fn rec(
        pair_types: &[StableEdge],
        edge_count: usize,
        start: usize,
        current: &mut Vec<StableEdge>,
        visit: &mut impl FnMut(&[StableEdge]),
    ) {
        if current.len() == edge_count {
            visit(current);
            return;
        }
        for idx in start..pair_types.len() {
            current.push(pair_types[idx].clone());
            rec(pair_types, edge_count, idx, current, visit);
            current.pop();
        }
    }

    rec(pair_types, edge_count, 0, &mut Vec::new(), &mut visit);
}

fn labelled_leg_assignments(legs: usize, vertex_count: usize) -> Vec<Vec<usize>> {
    fn rec(legs: usize, vertex_count: usize, current: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if current.len() == legs {
            out.push(current.clone());
            return;
        }
        for vertex in 0..vertex_count {
            current.push(vertex);
            rec(legs, vertex_count, current, out);
            current.pop();
        }
    }
    let mut out = Vec::new();
    rec(legs, vertex_count, &mut Vec::new(), &mut out);
    out
}

fn compositions(total: usize, parts: usize) -> Vec<Vec<usize>> {
    fn rec(total: usize, parts: usize, current: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if current.len() + 1 == parts {
            current.push(total);
            out.push(current.clone());
            current.pop();
            return;
        }
        for value in 0..=total {
            current.push(value);
            rec(total - value, parts, current, out);
            current.pop();
        }
    }
    if parts == 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    rec(total, parts, &mut Vec::new(), &mut out);
    out
}

fn permutations(n: usize) -> Vec<Vec<usize>> {
    fn rec(start: usize, values: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if start == values.len() {
            out.push(values.clone());
            return;
        }
        for idx in start..values.len() {
            values.swap(start, idx);
            rec(start + 1, values, out);
            values.swap(start, idx);
        }
    }
    let mut values = (0..n).collect::<Vec<_>>();
    let mut out = Vec::new();
    rec(0, &mut values, &mut out);
    out
}

fn edge_bijection_count_after_permutation(graph: &StableGraph, permutation: &[usize]) -> usize {
    let mut inverse = vec![0usize; permutation.len()];
    for (new, &old) in permutation.iter().enumerate() {
        inverse[old] = new;
    }
    let mut counts = BTreeMap::<(usize, usize), usize>::new();
    for edge in &graph.edges {
        let mapped = StableEdge::new(inverse[edge.a], inverse[edge.b]);
        *counts.entry((mapped.a, mapped.b)).or_default() += 1;
    }
    counts
        .into_iter()
        .map(|((a, b), count)| {
            let loop_flips = if a == b { 1usize << count } else { 1 };
            factorial(count) * loop_flips
        })
        .product()
}

fn factorial(n: usize) -> usize {
    (1..=n).product::<usize>().max(1)
}

#[derive(Debug, Clone)]
struct DisjointSet {
    parent: Vec<usize>,
}

impl DisjointSet {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            let root = self.find(self.parent[x]);
            self.parent[x] = root;
        }
        self.parent[x]
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            self.parent[rb] = ra;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_vertex_genus_zero_three_markings() {
        let graphs = stable_graphs(0, 3);
        assert_eq!(graphs.len(), 1);
        assert_eq!(graphs[0].genus(), 0);
        assert_eq!(graphs[0].automorphism_order(), 1);
    }

    #[test]
    fn genus_one_one_marking_has_irreducible_graph() {
        let graphs = stable_graphs(1, 1);
        assert!(graphs
            .iter()
            .any(|g| { g.vertices.len() == 1 && g.vertices[0].genus == 1 && g.edges.is_empty() }));
    }

    #[test]
    fn loop_contributes_to_betti_number() {
        let graph = StableGraph {
            vertices: vec![StableVertex { genus: 0 }],
            edges: vec![StableEdge::new(0, 0)],
            legs: vec![0, 0],
        };
        assert_eq!(graph.first_betti(), 1);
        assert_eq!(graph.valence(0), 4);
        assert_eq!(graph.automorphism_order(), 2);
        assert!(graph.is_stable());
    }
}
