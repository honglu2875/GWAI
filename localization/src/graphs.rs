//! Stable graphs for the boundary strata of `Mbar_{g,n}`.
//!
//! These are the abstract stable-curve graphs used by the Givental graph
//! expansion, not stable-map localization graphs.  Vertices carry genera, legs
//! are labelled markings, and automorphism orders include both vertex
//! symmetries and bijections of repeated edges.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{Condvar, Mutex, OnceLock};
use std::thread;

use crate::error::GwError;

/// Maximum `2g - 2 + n` accepted by the built-in stable-graph generator.
/// Complexity eight retains the documented frontier probes while preventing
/// untrusted bounds from entering an effectively unbounded enumeration.
pub const MAX_STABLE_GRAPH_COMPLEXITY: usize = 8;
/// Independent cap on labelled legs, whose assignments multiply graph counts.
pub const MAX_STABLE_GRAPH_MARKINGS: usize = 8;

/// Stability of a vertex (or complete connected curve) without evaluating
/// the potentially overflowing expression `2g + n > 2`.
pub(crate) fn is_stable_moduli_range(genus: usize, markings: usize) -> bool {
    match genus {
        0 => markings >= 3,
        1 => markings >= 1,
        _ => true,
    }
}

/// Complex dimension `3g - 3 + n` of a stable moduli space.
pub(crate) fn stable_graph_dimension(genus: usize, markings: usize) -> Result<usize, GwError> {
    if !is_stable_moduli_range(genus, markings) {
        return Err(GwError::UnsupportedInvariant(
            "stable-graph dimension is defined here only for stable (g,n) ranges".to_string(),
        ));
    }
    genus
        .checked_mul(3)
        .and_then(|value| value.checked_add(markings))
        .and_then(|value| value.checked_sub(3))
        .ok_or_else(|| {
            GwError::UnsupportedInvariant(format!(
                "stable-graph dimension is not representable for (g,n)=({genus},{markings})"
            ))
        })
}

/// Validate the finite work envelope and derive the maximum vertex count for
/// the built-in stable-graph enumeration.
pub fn stable_graph_generation_bounds(
    genus: usize,
    markings: usize,
) -> Result<StableGraphBounds, GwError> {
    if !is_stable_moduli_range(genus, markings) {
        return Err(GwError::UnsupportedInvariant(
            "stable-graph generation requires a stable (g,n) range".to_string(),
        ));
    }
    let complexity = genus
        .checked_mul(2)
        .and_then(|value| value.checked_add(markings))
        .and_then(|value| value.checked_sub(2))
        .ok_or_else(|| {
            GwError::UnsupportedInvariant(format!(
                "stable-graph complexity is not representable for (g,n)=({genus},{markings})"
            ))
        })?;
    if markings > MAX_STABLE_GRAPH_MARKINGS {
        return Err(GwError::ResourceLimit {
            operation: "stable-graph markings".to_string(),
            requested: markings,
            limit: MAX_STABLE_GRAPH_MARKINGS,
        });
    }
    if complexity > MAX_STABLE_GRAPH_COMPLEXITY {
        return Err(GwError::ResourceLimit {
            operation: "stable-graph complexity 2g-2+n".to_string(),
            requested: complexity,
            limit: MAX_STABLE_GRAPH_COMPLEXITY,
        });
    }
    Ok(StableGraphBounds {
        max_vertices: complexity.max(1),
    })
}

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

/// Complete isomorphism invariant of a stable graph: the lexicographically
/// smallest `(genera, sorted edges, legs)` tuple over the discrete leaf
/// labelings of the individualization-refinement search.
#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct CanonicalGraphKey {
    genera: Vec<usize>,
    edges: Vec<(usize, usize)>,
    legs: Vec<usize>,
}

impl StableGraph {
    pub fn first_betti(&self) -> usize {
        // b1 = E - V + C.  Counting components keeps this correct (and free of
        // usize underflow) for disconnected graphs, e.g. forests with more
        // than one tree, where the connected-only formula E - V + 1 fails.
        if self.vertices.is_empty() {
            return 0;
        }
        let mut dsu = DisjointSet::new(self.vertices.len());
        for edge in &self.edges {
            if !edge.is_loop() {
                dsu.union(edge.a, edge.b);
            }
        }
        let components = (0..self.vertices.len())
            .filter(|&vertex| dsu.find(vertex) == vertex)
            .count();
        self.edges.len() + components - self.vertices.len()
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
            .all(|(idx, vertex)| is_stable_moduli_range(vertex.genus, self.valence(idx)))
    }

    pub fn canonical_label(&self) -> String {
        let key = self.canonical_key();
        let genera = key
            .genera
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let edges = key
            .edges
            .iter()
            .map(|(a, b)| format!("{a}-{b}"))
            .collect::<Vec<_>>()
            .join(",");
        let legs = key
            .legs
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        format!("g:{genera}|e:{edges}|l:{legs}")
    }

    /// Canonical form by individualization-refinement.
    pub(crate) fn canonical_key(&self) -> CanonicalGraphKey {
        self.canonical_data().key
    }

    pub fn automorphism_order(&self) -> usize {
        self.canonical_data()
            .automorphisms
            .iter()
            .map(|permutation| edge_bijection_count_after_permutation(self, permutation))
            .sum()
    }

    pub fn vertex_automorphism_permutations(&self) -> Vec<Vec<usize>> {
        self.canonical_data().automorphisms
    }

    /// Canonical key and the complete vertex-automorphism list, both from one
    /// individualization-refinement search.
    ///
    /// Weisfeiler-Leman refinement alone stalls on regular structures (every
    /// vertex identical), where enumerating all class-respecting relabelings
    /// costs a product of class factorials.  Instead, whenever a class of size
    /// two or more remains, branch by distinguishing each of its members in
    /// turn and re-refining; distinguishing one vertex usually cascades, so
    /// the leaf count tracks the automorphism count rather than the factorial.
    ///
    /// The canonical key is the minimum over the leaf labelings.  Because the
    /// branching cell and refinement depend only on isomorphism-invariant
    /// data, the leaf key multiset — hence the minimum — is an isomorphism
    /// invariant.  The automorphism group acts freely and transitively on the
    /// minimal-key leaves, so pairing one fixed minimal leaf with every
    /// minimal leaf yields each vertex automorphism exactly once.
    fn canonical_data(&self) -> GraphCanonicalData {
        let vertex_count = self.vertices.len();
        if vertex_count == 0 {
            return GraphCanonicalData {
                key: CanonicalGraphKey::default(),
                automorphisms: vec![Vec::new()],
            };
        }

        let leaves = RefinementSearch::new(self).leaves();
        let mut best: Option<CanonicalGraphKey> = None;
        let mut minimal_leaves = Vec::new();
        for leaf in leaves {
            let key = self.key_with_permutation(&leaf);
            match best.as_ref().map(|current| key.cmp(current)) {
                None | Some(Ordering::Less) => {
                    best = Some(key);
                    minimal_leaves.clear();
                    minimal_leaves.push(leaf);
                }
                Some(Ordering::Equal) => minimal_leaves.push(leaf),
                Some(Ordering::Greater) => {}
            }
        }

        let reference = minimal_leaves
            .first()
            .cloned()
            .expect("individualization-refinement produces at least one leaf");
        let automorphisms = minimal_leaves
            .into_iter()
            .map(|leaf| {
                let mut automorphism = vec![0usize; vertex_count];
                for (reference_old, &leaf_old) in reference.iter().zip(leaf.iter()) {
                    automorphism[*reference_old] = leaf_old;
                }
                automorphism
            })
            .collect::<Vec<_>>();
        debug_assert!({
            let identity_key = self.key_with_permutation(&(0..vertex_count).collect::<Vec<_>>());
            automorphisms
                .iter()
                .all(|automorphism| self.key_with_permutation(automorphism) == identity_key)
        });

        GraphCanonicalData {
            key: best.expect("at least one leaf"),
            automorphisms,
        }
    }

    fn key_with_permutation(&self, permutation: &[usize]) -> CanonicalGraphKey {
        let mut inverse = vec![0usize; permutation.len()];
        for (new, &old) in permutation.iter().enumerate() {
            inverse[old] = new;
        }

        let genera = permutation
            .iter()
            .map(|&old| self.vertices[old].genus)
            .collect();
        let mut edges = self
            .edges
            .iter()
            .map(|edge| {
                let a = inverse[edge.a];
                let b = inverse[edge.b];
                if a <= b {
                    (a, b)
                } else {
                    (b, a)
                }
            })
            .collect::<Vec<_>>();
        edges.sort_unstable();
        let legs = self.legs.iter().map(|&old| inverse[old]).collect();

        CanonicalGraphKey {
            genera,
            edges,
            legs,
        }
    }
}

struct GraphCanonicalData {
    key: CanonicalGraphKey,
    /// Complete vertex-automorphism list, encoded like the permutations of
    /// `key_with_permutation`: relabeling by each entry reproduces the
    /// identity-labelled graph.
    automorphisms: Vec<Vec<usize>>,
}

/// Individualization-refinement search over vertex colorings.
///
/// Colors start from the local invariant `(genus, loop count, marking labels,
/// non-loop multiplicity multiset)` and are refined by neighbor colors until
/// stable (Weisfeiler-Leman).  When refinement stalls with a class of two or
/// more vertices, the search branches: each member of the first such class is
/// individualized in turn and refinement restarts.  Every branching choice
/// (the class picked, the refinement itself) depends only on
/// isomorphism-invariant data, so the discrete leaf colorings of isomorphic
/// graphs correspond under any isomorphism.
struct RefinementSearch<'g> {
    graph: &'g StableGraph,
    /// `adjacency[v]`: sorted `(neighbor, multiplicity)` pairs over non-loop
    /// edges; loops and legs are folded into the initial colors.
    adjacency: Vec<Vec<(usize, usize)>>,
}

impl<'g> RefinementSearch<'g> {
    fn new(graph: &'g StableGraph) -> Self {
        let vertex_count = graph.vertices.len();
        let mut adjacency_maps = vec![BTreeMap::<usize, usize>::new(); vertex_count];
        for edge in &graph.edges {
            if !edge.is_loop() {
                *adjacency_maps[edge.a].entry(edge.b).or_default() += 1;
                *adjacency_maps[edge.b].entry(edge.a).or_default() += 1;
            }
        }
        Self {
            graph,
            adjacency: adjacency_maps
                .into_iter()
                .map(|map| map.into_iter().collect())
                .collect(),
        }
    }

    fn initial_colors(&self) -> Vec<usize> {
        let vertex_count = self.graph.vertices.len();
        let mut loops = vec![0usize; vertex_count];
        for edge in &self.graph.edges {
            if edge.is_loop() {
                loops[edge.a] += 1;
            }
        }
        let mut leg_labels = vec![Vec::<usize>::new(); vertex_count];
        for (marking, &vertex) in self.graph.legs.iter().enumerate() {
            leg_labels[vertex].push(marking);
        }
        let signatures = (0..vertex_count)
            .map(|vertex| {
                let mut multiplicities = self.adjacency[vertex]
                    .iter()
                    .map(|&(_, multiplicity)| multiplicity)
                    .collect::<Vec<_>>();
                multiplicities.sort_unstable();
                (
                    self.graph.vertices[vertex].genus,
                    loops[vertex],
                    leg_labels[vertex].clone(),
                    multiplicities,
                )
            })
            .collect::<Vec<_>>();
        colors_from_signatures(&signatures)
    }

    fn refine(&self, colors: &mut Vec<usize>) {
        loop {
            let signatures = (0..colors.len())
                .map(|vertex| {
                    let mut neighbor_colors = self.adjacency[vertex]
                        .iter()
                        .map(|&(neighbor, multiplicity)| (colors[neighbor], multiplicity))
                        .collect::<Vec<_>>();
                    neighbor_colors.sort_unstable();
                    (colors[vertex], neighbor_colors)
                })
                .collect::<Vec<_>>();
            let next = colors_from_signatures(&signatures);
            if next == *colors {
                return;
            }
            *colors = next;
        }
    }

    /// Discrete leaf colorings as permutations `permutation[color] = vertex`.
    fn leaves(&self) -> Vec<Vec<usize>> {
        let mut colors = self.initial_colors();
        self.refine(&mut colors);
        let mut out = Vec::new();
        self.branch(colors, &mut out);
        out
    }

    fn branch(&self, colors: Vec<usize>, out: &mut Vec<Vec<usize>>) {
        let vertex_count = colors.len();
        let mut counts = vec![0usize; vertex_count];
        for &color in &colors {
            counts[color] += 1;
        }
        let Some(target_color) = counts.iter().position(|&count| count >= 2) else {
            // Discrete: color ids are dense ranks, so they are the positions.
            let mut permutation = vec![0usize; vertex_count];
            for (vertex, &color) in colors.iter().enumerate() {
                permutation[color] = vertex;
            }
            out.push(permutation);
            return;
        };

        for vertex in 0..vertex_count {
            if colors[vertex] != target_color {
                continue;
            }
            let signatures = (0..vertex_count)
                .map(|other| (colors[other], usize::from(other == vertex)))
                .collect::<Vec<_>>();
            let mut next = colors_from_signatures(&signatures);
            self.refine(&mut next);
            self.branch(next, out);
        }
    }
}

fn colors_from_signatures<S: Ord>(signatures: &[S]) -> Vec<usize> {
    let mut sorted = signatures.iter().collect::<Vec<_>>();
    sorted.sort();
    sorted.dedup();
    signatures
        .iter()
        .map(|signature| {
            sorted
                .binary_search(&signature)
                .expect("signature present in its own sorted list")
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StableGraphBounds {
    pub max_vertices: usize,
}

pub fn stable_graphs(genus: usize, legs: usize) -> Vec<StableGraph> {
    try_stable_graphs(genus, legs)
        .expect("stable_graphs input exceeds the built-in finite generation envelope")
}

/// Fallible stable-graph enumeration for untrusted genus and marking bounds.
pub fn try_stable_graphs(genus: usize, legs: usize) -> Result<Vec<StableGraph>, GwError> {
    let bounds = stable_graph_generation_bounds(genus, legs)?;
    // The universal Givental sum only needs stable graphs for fixed (g,n), so
    // cache them independently of target, degree, and insertions — in memory
    // for this process, and on disk for expensive tables so high-genus runs
    // pay generation once per machine rather than once per invocation.
    static CACHE: OnceLock<StableGraphCache> = OnceLock::new();
    let cache = CACHE.get_or_init(StableGraphCache::new);
    Ok(cache.get_or_generate(genus, legs, bounds))
}

struct StableGraphCache {
    entries: Mutex<BTreeMap<(usize, usize), StableGraphCacheEntry>>,
    ready: Condvar,
}

#[derive(Clone)]
enum StableGraphCacheEntry {
    Ready(Vec<StableGraph>),
    Generating,
}

impl StableGraphCache {
    fn new() -> Self {
        Self {
            entries: Mutex::new(BTreeMap::new()),
            ready: Condvar::new(),
        }
    }

    fn get_or_generate(
        &self,
        genus: usize,
        legs: usize,
        bounds: StableGraphBounds,
    ) -> Vec<StableGraph> {
        let key = (genus, legs);
        let mut entries = self.entries.lock().unwrap();
        loop {
            match entries.get(&key) {
                Some(StableGraphCacheEntry::Ready(graphs)) => return graphs.clone(),
                Some(StableGraphCacheEntry::Generating) => {
                    entries = self.ready.wait(entries).unwrap();
                }
                None => {
                    entries.insert(key, StableGraphCacheEntry::Generating);
                    break;
                }
            }
        }
        drop(entries);

        let graphs = generate_or_load_stable_graphs(genus, legs, bounds);

        let mut entries = self.entries.lock().unwrap();
        entries.insert(key, StableGraphCacheEntry::Ready(graphs.clone()));
        self.ready.notify_all();
        graphs
    }
}

fn generate_or_load_stable_graphs(
    genus: usize,
    legs: usize,
    bounds: StableGraphBounds,
) -> Vec<StableGraph> {
    if let Some(graphs) = load_stable_graphs_from_disk(genus, legs) {
        return graphs;
    }

    let started = std::time::Instant::now();
    let graphs = stable_graphs_with_bounds(genus, legs, bounds);
    if started.elapsed() >= DISK_CACHE_MIN_GENERATION_TIME {
        store_stable_graphs_to_disk(genus, legs, &graphs);
    }
    graphs
}

/// Only tables that took real work to generate are worth a disk file.
const DISK_CACHE_MIN_GENERATION_TIME: std::time::Duration = std::time::Duration::from_millis(100);

/// Bump when the generator, canonical representatives, or encoding change.
const DISK_CACHE_FORMAT: &str = "v1";

fn stable_graph_cache_path(genus: usize, legs: usize) -> Option<std::path::PathBuf> {
    if crate::env_flag("GWAI_DISABLE_GRAPH_CACHE") {
        return None;
    }
    let base = std::env::var_os("GWAI_GRAPH_CACHE_DIR")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var_os("XDG_CACHE_HOME")
                .map(std::path::PathBuf::from)
                .or_else(|| {
                    std::env::var_os("HOME")
                        .map(|home| std::path::PathBuf::from(home).join(".cache"))
                })
                .map(|cache| cache.join("gw-pn"))
        })?;
    Some(base.join(format!(
        "stable-graphs-{DISK_CACHE_FORMAT}-g{genus}-n{legs}.txt"
    )))
}

fn load_stable_graphs_from_disk(genus: usize, legs: usize) -> Option<Vec<StableGraph>> {
    let path = stable_graph_cache_path(genus, legs)?;
    let contents = std::fs::read_to_string(path).ok()?;
    let graphs = decode_stable_graphs(&contents, genus, legs)?;
    // Cheap structural audit so a corrupt or stale file regenerates instead
    // of silently poisoning every computation built on these tables.
    graphs
        .iter()
        .all(|graph| {
            graph.legs.len() == legs
                && graph.genus() == genus
                && graph.is_connected()
                && graph.is_stable()
        })
        .then_some(graphs)
}

fn store_stable_graphs_to_disk(genus: usize, legs: usize, graphs: &[StableGraph]) {
    let Some(path) = stable_graph_cache_path(genus, legs) else {
        return;
    };
    let Some(parent) = path.parent() else {
        return;
    };
    if std::fs::create_dir_all(parent).is_err() {
        return;
    }
    let temporary = path.with_extension("txt.tmp");
    if std::fs::write(&temporary, encode_stable_graphs(genus, legs, graphs)).is_ok() {
        let _ = std::fs::rename(&temporary, &path);
    }
}

fn encode_stable_graphs(genus: usize, legs: usize, graphs: &[StableGraph]) -> String {
    let mut out = format!(
        "gw-pn stable graphs {DISK_CACHE_FORMAT}\ngenus {genus} legs {legs} count {}\n",
        graphs.len()
    );
    for graph in graphs {
        let genera = graph
            .vertices
            .iter()
            .map(|vertex| vertex.genus.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let edges = graph
            .edges
            .iter()
            .map(|edge| format!("{}-{}", edge.a, edge.b))
            .collect::<Vec<_>>()
            .join(",");
        let leg_list = graph
            .legs
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        out.push_str(&format!("{genera}|{edges}|{leg_list}\n"));
    }
    out
}

fn decode_stable_graphs(contents: &str, genus: usize, legs: usize) -> Option<Vec<StableGraph>> {
    let mut lines = contents.lines();
    if lines.next()? != format!("gw-pn stable graphs {DISK_CACHE_FORMAT}") {
        return None;
    }
    let header = lines.next()?;
    let mut header_parts = header.split_whitespace();
    if header_parts.next()? != "genus" || header_parts.next()?.parse::<usize>().ok()? != genus {
        return None;
    }
    if header_parts.next()? != "legs" || header_parts.next()?.parse::<usize>().ok()? != legs {
        return None;
    }
    if header_parts.next()? != "count" {
        return None;
    }
    let count = header_parts.next()?.parse::<usize>().ok()?;

    let mut graphs = Vec::with_capacity(count);
    for line in lines {
        let mut sections = line.split('|');
        let genera = sections.next()?;
        let edges = sections.next()?;
        let leg_list = sections.next()?;
        if sections.next().is_some() {
            return None;
        }
        let vertices = split_csv(genera)?
            .into_iter()
            .map(|genus| StableVertex { genus })
            .collect::<Vec<_>>();
        let vertex_count = vertices.len();
        let mut parsed_edges = Vec::new();
        if !edges.is_empty() {
            for pair in edges.split(',') {
                let (a, b) = pair.split_once('-')?;
                let edge = StableEdge::new(a.parse().ok()?, b.parse().ok()?);
                if edge.b >= vertex_count {
                    return None;
                }
                parsed_edges.push(edge);
            }
        }
        let parsed_legs = split_csv(leg_list)?;
        if parsed_legs.iter().any(|&vertex| vertex >= vertex_count) {
            return None;
        }
        graphs.push(StableGraph {
            vertices,
            edges: parsed_edges,
            legs: parsed_legs,
        });
    }
    (graphs.len() == count).then_some(graphs)
}

fn split_csv(section: &str) -> Option<Vec<usize>> {
    if section.is_empty() {
        return Some(Vec::new());
    }
    section
        .split(',')
        .map(|value| value.parse::<usize>().ok())
        .collect()
}

pub fn stable_graphs_with_bounds(
    genus: usize,
    legs: usize,
    bounds: StableGraphBounds,
) -> Vec<StableGraph> {
    // Two-stage quotient by graph isomorphism.  Stage one: enumerate connected
    // edge multisets (connectivity enforced inside the recursion) and keep
    // only those that are canonical as unlabelled multigraphs, computing their
    // vertex automorphisms in the same sweep.  Stage two: decorate each
    // canonical skeleton with legs and genera — stability enforced by giving
    // each vertex its minimum admissible genus up front — and keep exactly the
    // decorations that are lexicographically minimal in their orbit under the
    // skeleton automorphisms.  Isomorphic decorated graphs share an isomorphic
    // skeleton, and canonical skeletons of isomorphic multigraphs are equal as
    // multisets, so every isomorphism class survives exactly once without any
    // per-decoration canonical form.
    let mut tasks = Vec::new();
    for vertex_count in 1..=bounds.max_vertices.max(1) {
        let min_edges = vertex_count - 1;
        let max_edges = genus + vertex_count - 1;
        for edge_count in min_edges..=max_edges {
            tasks.extend(stable_graph_tasks_for_bucket(vertex_count, edge_count));
        }
    }

    let worker_count = stable_graph_generation_worker_count(tasks.len());
    let out = if worker_count <= 1 {
        tasks
            .iter()
            .flat_map(|task| stable_graph_task(genus, legs, task.clone()))
            .collect::<Vec<_>>()
    } else {
        let next_task = AtomicUsize::new(0);
        let task_results = thread::scope(|scope| {
            let mut handles = Vec::new();
            for _ in 0..worker_count {
                let next_task = &next_task;
                let tasks = &tasks;
                handles.push(scope.spawn(move || {
                    let mut out = Vec::new();
                    loop {
                        let task_index = next_task.fetch_add(1, AtomicOrdering::Relaxed);
                        let Some(task) = tasks.get(task_index).cloned() else {
                            break;
                        };
                        out.push((task_index, stable_graph_task(genus, legs, task)));
                    }
                    out
                }));
            }
            handles
                .into_iter()
                .flat_map(|handle| handle.join().expect("stable graph worker panicked"))
                .collect::<Vec<_>>()
        });
        let mut ordered = vec![Vec::<StableGraph>::new(); tasks.len()];
        for (task_index, graphs) in task_results {
            ordered[task_index] = graphs;
        }
        ordered.into_iter().flatten().collect::<Vec<_>>()
    };

    debug_assert_eq!(
        out.iter()
            .map(StableGraph::canonical_key)
            .collect::<BTreeSet<_>>()
            .len(),
        out.len(),
        "two-stage quotient produced duplicate isomorphism classes"
    );
    out
}

#[derive(Debug, Clone)]
struct StableGraphTask {
    vertex_count: usize,
    edge_count: usize,
    prefix_pair_indices: Vec<usize>,
}

const STABLE_GRAPH_PARALLEL_PREFIX_DEPTH: usize = 2;

fn stable_graph_tasks_for_bucket(vertex_count: usize, edge_count: usize) -> Vec<StableGraphTask> {
    if edge_count == 0 {
        return vec![StableGraphTask {
            vertex_count,
            edge_count,
            prefix_pair_indices: Vec::new(),
        }];
    }
    let pair_types = edge_pair_types(vertex_count);
    let last_touch = edge_pair_last_touch(vertex_count, &pair_types);
    let prefix_depth = STABLE_GRAPH_PARALLEL_PREFIX_DEPTH.min(edge_count);
    let mut tasks = Vec::new();
    let mut prefix = Vec::with_capacity(prefix_depth);
    let mut degree = vec![0usize; vertex_count];
    let mut dsu = RollbackDisjointSet::new(vertex_count);
    stable_graph_task_prefix_rec(
        vertex_count,
        edge_count,
        prefix_depth,
        &pair_types,
        &last_touch,
        0,
        &mut prefix,
        &mut degree,
        &mut dsu,
        vertex_count,
        &mut tasks,
    );
    tasks
}

fn stable_graph_task_prefix_rec(
    vertex_count: usize,
    edge_count: usize,
    prefix_depth: usize,
    pair_types: &[StableEdge],
    last_touch: &[usize],
    start: usize,
    prefix: &mut Vec<usize>,
    degree: &mut [usize],
    dsu: &mut RollbackDisjointSet,
    components: usize,
    tasks: &mut Vec<StableGraphTask>,
) {
    if prefix.len() == prefix_depth || prefix.len() == edge_count {
        tasks.push(StableGraphTask {
            vertex_count,
            edge_count,
            prefix_pair_indices: prefix.clone(),
        });
        return;
    }
    let remaining = edge_count - prefix.len();
    if components > remaining + 1 {
        return;
    }
    for idx in start..pair_types.len() {
        if degree
            .iter()
            .enumerate()
            .any(|(vertex, &d)| d == 0 && last_touch[vertex] < idx)
        {
            break;
        }
        let pair = &pair_types[idx];
        let checkpoint = dsu.checkpoint();
        let merged = !pair.is_loop() && dsu.union_merged(pair.a, pair.b);
        let next_components = components - usize::from(merged);
        degree[pair.a] += 1;
        degree[pair.b] += 1;
        prefix.push(idx);
        stable_graph_task_prefix_rec(
            vertex_count,
            edge_count,
            prefix_depth,
            pair_types,
            last_touch,
            idx,
            prefix,
            degree,
            dsu,
            next_components,
            tasks,
        );
        prefix.pop();
        degree[pair.b] -= 1;
        degree[pair.a] -= 1;
        dsu.rollback_to(checkpoint);
    }
}

fn stable_graph_task(genus: usize, legs: usize, task: StableGraphTask) -> Vec<StableGraph> {
    let vertex_count = task.vertex_count;
    let edge_count = task.edge_count;
    let pair_types = edge_pair_types(vertex_count);
    let leg_assignments = labelled_leg_assignments(legs, vertex_count);
    let h1 = edge_count + 1 - vertex_count;
    let genus_sum = genus - h1;
    let mut out = Vec::new();
    let last_touch = edge_pair_last_touch(vertex_count, &pair_types);
    let mut current = Vec::with_capacity(edge_count);
    let mut degree = vec![0usize; vertex_count];
    let mut dsu = RollbackDisjointSet::new(vertex_count);
    let mut components = vertex_count;
    for &pair_idx in &task.prefix_pair_indices {
        let pair = &pair_types[pair_idx];
        let merged = !pair.is_loop() && dsu.union_merged(pair.a, pair.b);
        components -= usize::from(merged);
        degree[pair.a] += 1;
        degree[pair.b] += 1;
        current.push(pair.clone());
    }
    let mut visit = |edges: &[StableEdge]| {
        decorate_connected_edge_multiset(
            genus_sum,
            vertex_count,
            &leg_assignments,
            edges,
            &mut out,
        );
    };
    if edge_count == 0 {
        visit(&[]);
    } else {
        let start = task.prefix_pair_indices.last().copied().unwrap_or(0);
        connected_edge_multiset_rec(
            &pair_types,
            &last_touch,
            edge_count,
            start,
            &mut current,
            &mut degree,
            &mut dsu,
            components,
            &mut visit,
        );
    }
    out
}

fn decorate_connected_edge_multiset(
    genus_sum: usize,
    vertex_count: usize,
    leg_assignments: &[Vec<usize>],
    edges: &[StableEdge],
    out: &mut Vec<StableGraph>,
) {
    // Computed lazily: multisets whose valences admit no stable decoration
    // never pay for a canonicality sweep.
    let mut quotient_cache: Option<SkeletonQuotient> = None;

    let mut edge_valence = vec![0usize; vertex_count];
    for edge in edges {
        edge_valence[edge.a] += 1;
        edge_valence[edge.b] += 1;
    }
    for leg_assignment in leg_assignments {
        let mut valence = edge_valence.clone();
        for &vertex in leg_assignment {
            valence[vertex] += 1;
        }
        // Stability 2g_v + val_v > 2 gives a per-vertex genus minimum;
        // distributing only the remaining genus over the vertices enumerates
        // exactly the stable assignments.
        let genus_minimums = valence
            .iter()
            .map(|&val| match val {
                0 => 2,
                1 | 2 => 1,
                _ => 0,
            })
            .collect::<Vec<_>>();
        let required = genus_minimums.iter().sum::<usize>();
        if required > genus_sum {
            continue;
        }
        for extra in compositions(genus_sum - required, vertex_count) {
            let quotient =
                quotient_cache.get_or_insert_with(|| skeleton_quotient(vertex_count, edges));
            let automorphisms = match quotient {
                SkeletonQuotient::Canonical(automorphisms) => automorphisms,
                // A relabelled isomorph of a canonical multiset elsewhere in
                // the enumeration: skip it entirely.
                SkeletonQuotient::NotCanonical => return,
            };
            let genera = genus_minimums
                .iter()
                .zip(extra.iter())
                .map(|(&minimum, &extra_genus)| minimum + extra_genus)
                .collect::<Vec<_>>();
            if !decoration_is_orbit_minimal(automorphisms, &genera, leg_assignment) {
                continue;
            }
            let graph = StableGraph {
                vertices: genera
                    .into_iter()
                    .map(|genus| StableVertex { genus })
                    .collect(),
                edges: edges.to_vec(),
                legs: leg_assignment.clone(),
            };
            debug_assert!(graph.is_connected() && graph.is_stable());
            out.push(graph);
        }
    }
}

fn stable_graph_generation_worker_count(work_items: usize) -> usize {
    const MIN_PARALLEL_BUCKETS: usize = 8;
    if work_items < MIN_PARALLEL_BUCKETS {
        return 1;
    }
    let available = thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    let requested = std::env::var("GW_THREADS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|count| *count > 0)
        .unwrap_or(available);
    requested.min(work_items).max(1)
}

enum SkeletonQuotient {
    /// Canonical representative; carries the skeleton vertex automorphisms
    /// (encoded as `permutation[new] = old`).
    Canonical(Vec<Vec<usize>>),
    /// A relabelled isomorph of a canonical multiset elsewhere in the
    /// enumeration.
    NotCanonical,
}

fn skeleton_quotient(vertex_count: usize, edges: &[StableEdge]) -> SkeletonQuotient {
    let skeleton = StableGraph {
        vertices: vec![StableVertex { genus: 0 }; vertex_count],
        edges: edges.to_vec(),
        legs: Vec::new(),
    };
    let identity_key = skeleton.key_with_permutation(&(0..vertex_count).collect::<Vec<_>>());
    let data = skeleton.canonical_data();
    if data.key == identity_key {
        SkeletonQuotient::Canonical(data.automorphisms)
    } else {
        SkeletonQuotient::NotCanonical
    }
}

/// Whether `(genera, legs)` is lexicographically minimal in its orbit under
/// the skeleton automorphisms, so that each decorated isomorphism class keeps
/// exactly one representative.
fn decoration_is_orbit_minimal(
    automorphisms: &[Vec<usize>],
    genera: &[usize],
    legs: &[usize],
) -> bool {
    let mut inverse = vec![0usize; genera.len()];
    for permutation in automorphisms {
        for (new, &old) in permutation.iter().enumerate() {
            inverse[old] = new;
        }
        let permuted_genera = permutation.iter().map(|&old| genera[old]);
        match permuted_genera.cmp(genera.iter().copied()) {
            Ordering::Less => return false,
            Ordering::Greater => continue,
            Ordering::Equal => {}
        }
        let permuted_legs = legs.iter().map(|&old_vertex| inverse[old_vertex]);
        if permuted_legs.cmp(legs.iter().copied()) == Ordering::Less {
            return false;
        }
    }
    true
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

/// Visits every nondecreasing multiset of `edge_count` vertex pairs whose
/// non-loop edges connect all `vertex_count` vertices.
///
/// The recursion prunes branches that can no longer become connected: when the
/// component count exceeds the remaining edge budget plus one, and when a
/// still-isolated vertex has no incident pair left at or beyond the current
/// multiset position.
#[cfg(test)]
fn for_each_connected_edge_multiset(
    vertex_count: usize,
    pair_types: &[StableEdge],
    edge_count: usize,
    visit: &mut impl FnMut(&[StableEdge]),
) {
    if edge_count == 0 {
        if vertex_count == 1 {
            visit(&[]);
        }
        return;
    }
    let last_touch = edge_pair_last_touch(vertex_count, pair_types);
    connected_edge_multiset_rec(
        pair_types,
        &last_touch,
        edge_count,
        0,
        &mut Vec::with_capacity(edge_count),
        &mut vec![0usize; vertex_count],
        &mut RollbackDisjointSet::new(vertex_count),
        vertex_count,
        visit,
    );
}

fn connected_edge_multiset_rec(
    pair_types: &[StableEdge],
    last_touch: &[usize],
    edge_count: usize,
    start: usize,
    current: &mut Vec<StableEdge>,
    degree: &mut [usize],
    dsu: &mut RollbackDisjointSet,
    components: usize,
    visit: &mut impl FnMut(&[StableEdge]),
) {
    if current.len() == edge_count {
        if components == 1 {
            visit(current);
        }
        return;
    }
    let remaining = edge_count - current.len();
    if components > remaining + 1 {
        return;
    }
    for idx in start..pair_types.len() {
        // Once an untouched vertex has no incident pair at or beyond idx, no
        // extension from here can connect it.
        if degree
            .iter()
            .enumerate()
            .any(|(vertex, &d)| d == 0 && last_touch[vertex] < idx)
        {
            break;
        }
        let pair = &pair_types[idx];
        let checkpoint = dsu.checkpoint();
        let merged = !pair.is_loop() && dsu.union_merged(pair.a, pair.b);
        let next_components = components - usize::from(merged);
        if !pair.is_loop() && merged {
            debug_assert!(next_components + 1 == components);
        }
        degree[pair.a] += 1;
        degree[pair.b] += 1;
        current.push(pair.clone());
        connected_edge_multiset_rec(
            pair_types,
            last_touch,
            edge_count,
            idx,
            current,
            degree,
            dsu,
            next_components,
            visit,
        );
        current.pop();
        degree[pair.b] -= 1;
        degree[pair.a] -= 1;
        dsu.rollback_to(checkpoint);
    }
}

fn edge_pair_last_touch(vertex_count: usize, pair_types: &[StableEdge]) -> Vec<usize> {
    let mut last_touch = vec![0usize; vertex_count];
    for (idx, pair) in pair_types.iter().enumerate() {
        last_touch[pair.a] = idx;
        last_touch[pair.b] = idx;
    }
    last_touch
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
        self.union_merged(a, b);
    }

    /// Union returning whether two distinct components were merged.
    fn union_merged(&mut self, a: usize, b: usize) -> bool {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            self.parent[rb] = ra;
            true
        } else {
            false
        }
    }
}

#[derive(Debug, Clone)]
struct RollbackDisjointSet {
    parent: Vec<usize>,
    size: Vec<usize>,
    history: Vec<(usize, usize, usize)>,
}

impl RollbackDisjointSet {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            size: vec![1; n],
            history: Vec::new(),
        }
    }

    fn find(&self, mut x: usize) -> usize {
        while self.parent[x] != x {
            x = self.parent[x];
        }
        x
    }

    fn union_merged(&mut self, a: usize, b: usize) -> bool {
        let mut ra = self.find(a);
        let mut rb = self.find(b);
        if ra == rb {
            return false;
        }
        if self.size[ra] < self.size[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.history.push((rb, ra, self.size[ra]));
        self.parent[rb] = ra;
        self.size[ra] += self.size[rb];
        true
    }

    fn checkpoint(&self) -> usize {
        self.history.len()
    }

    fn rollback_to(&mut self, checkpoint: usize) {
        while self.history.len() > checkpoint {
            let (child, parent, parent_size) =
                self.history.pop().expect("history length checked above");
            self.parent[child] = child;
            self.size[parent] = parent_size;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stability_check_does_not_overflow_for_extreme_vertex_genus() {
        let graph = StableGraph {
            vertices: vec![StableVertex { genus: usize::MAX }],
            edges: Vec::new(),
            legs: Vec::new(),
        };
        assert!(graph.is_stable());
    }

    #[test]
    fn fallible_generation_enforces_the_finite_work_envelope() {
        assert!(stable_graph_generation_bounds(4, 2).is_ok());
        assert!(matches!(
            stable_graph_generation_bounds(5, 1),
            Err(GwError::ResourceLimit {
                operation,
                requested: 9,
                limit: MAX_STABLE_GRAPH_COMPLEXITY,
            }) if operation == "stable-graph complexity 2g-2+n"
        ));
        assert!(matches!(
            stable_graph_generation_bounds(0, 9),
            Err(GwError::ResourceLimit {
                operation,
                requested: 9,
                limit: MAX_STABLE_GRAPH_MARKINGS,
            }) if operation == "stable-graph markings"
        ));
        assert!(matches!(
            try_stable_graphs(usize::MAX, 0),
            Err(GwError::UnsupportedInvariant(_))
        ));
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

    /// Reference implementation: full `V!` sweep, as used before the
    /// class-refined fast path.
    fn automorphism_order_bruteforce(graph: &StableGraph) -> usize {
        let vertex_count = graph.vertices.len();
        let identity = (0..vertex_count).collect::<Vec<_>>();
        let base = graph.key_with_permutation(&identity);
        permutations(vertex_count)
            .into_iter()
            .filter(|permutation| graph.key_with_permutation(permutation) == base)
            .map(|permutation| edge_bijection_count_after_permutation(graph, &permutation))
            .sum()
    }

    fn canonical_key_bruteforce(graph: &StableGraph) -> CanonicalGraphKey {
        permutations(graph.vertices.len())
            .into_iter()
            .map(|permutation| graph.key_with_permutation(&permutation))
            .min()
            .unwrap_or_default()
    }

    fn vertex_automorphisms_bruteforce(graph: &StableGraph) -> Vec<Vec<usize>> {
        let vertex_count = graph.vertices.len();
        let identity = (0..vertex_count).collect::<Vec<_>>();
        let base = graph.key_with_permutation(&identity);
        permutations(vertex_count)
            .into_iter()
            .filter(|permutation| graph.key_with_permutation(permutation) == base)
            .collect()
    }

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
    fn first_betti_handles_disconnected_graphs() {
        let forest = StableGraph {
            vertices: vec![StableVertex { genus: 1 }, StableVertex { genus: 2 }],
            edges: vec![],
            legs: vec![0, 1],
        };
        assert_eq!(forest.first_betti(), 0);
        assert_eq!(forest.genus(), 3);

        let mixed = StableGraph {
            vertices: vec![
                StableVertex { genus: 0 },
                StableVertex { genus: 0 },
                StableVertex { genus: 1 },
            ],
            edges: vec![StableEdge::new(0, 1), StableEdge::new(0, 1)],
            legs: vec![2, 2],
        };
        assert_eq!(mixed.first_betti(), 1);
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

    #[test]
    fn stable_graph_counts_match_reference() {
        // Counts produced by the pre-optimization brute-force generator
        // (labelled enumeration, string canonical labels, full V! sweeps).
        for (genus, markings, expected) in [
            (0usize, 3usize, 1usize),
            (0, 4, 4),
            (1, 1, 2),
            (1, 2, 5),
            (1, 3, 23),
            (1, 4, 163),
            (2, 0, 7),
            (2, 1, 16),
            (2, 2, 75),
            (2, 3, 555),
            (3, 1, 181),
        ] {
            assert_eq!(
                stable_graphs(genus, markings).len(),
                expected,
                "graph count mismatch at (g,n)=({genus},{markings})"
            );
        }
    }

    #[test]
    #[ignore = "takes about a minute; run with cargo test -- --ignored"]
    fn stable_graph_counts_match_reference_high_genus() {
        // (3,0), (3,2), (4,1) from the pre-optimization generator; (4,1)
        // took 100 minutes there and about a minute here.
        assert_eq!(stable_graphs(3, 0).len(), 42);
        assert_eq!(stable_graphs(3, 2).len(), 1355);
        assert_eq!(stable_graphs(4, 1).len(), 2666);
    }

    #[test]
    fn refined_canonicalization_matches_bruteforce() {
        for (genus, markings) in [
            (0, 3),
            (0, 4),
            (0, 5),
            (1, 1),
            (1, 2),
            (2, 0),
            (2, 1),
            (2, 2),
        ] {
            let graphs = stable_graphs(genus, markings);
            for graph in &graphs {
                assert_eq!(
                    graph.automorphism_order(),
                    automorphism_order_bruteforce(graph),
                    "automorphism order mismatch for {graph:?}"
                );
                let mut fast = graph.vertex_automorphism_permutations();
                let mut brute = vertex_automorphisms_bruteforce(graph);
                fast.sort();
                brute.sort();
                assert_eq!(fast, brute, "automorphism set mismatch for {graph:?}");
            }
            // The refined canonical key must induce the same isomorphism
            // classes as the brute-force minimum over all permutations.
            for left in &graphs {
                for right in &graphs {
                    assert_eq!(
                        left.canonical_key() == right.canonical_key(),
                        canonical_key_bruteforce(left) == canonical_key_bruteforce(right),
                        "isomorphism classification mismatch:\n{left:?}\n{right:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn disk_cache_encoding_round_trips() {
        for (genus, legs) in [(1usize, 2usize), (2, 1), (0, 4)] {
            let graphs = stable_graphs(genus, legs);
            let encoded = encode_stable_graphs(genus, legs, &graphs);
            let decoded =
                decode_stable_graphs(&encoded, genus, legs).expect("encoded table must decode");
            assert_eq!(decoded, graphs);
            // Header mismatches and truncation must be rejected, not
            // silently accepted.
            assert!(decode_stable_graphs(&encoded, genus + 1, legs).is_none());
            assert!(decode_stable_graphs(&encoded, genus, legs + 1).is_none());
            let mut truncated = encoded.lines().collect::<Vec<_>>();
            if truncated.len() > 2 {
                truncated.pop();
                assert!(decode_stable_graphs(&truncated.join("\n"), genus, legs).is_none());
            }
        }
    }

    #[test]
    fn connected_multiset_enumeration_matches_filtered_bruteforce() {
        // The pruned recursion must visit exactly the connected multisets the
        // old generate-then-filter enumeration produced, in the same order.
        fn bruteforce(
            pair_types: &[StableEdge],
            vertex_count: usize,
            edge_count: usize,
        ) -> Vec<Vec<StableEdge>> {
            fn rec(
                pair_types: &[StableEdge],
                edge_count: usize,
                start: usize,
                current: &mut Vec<StableEdge>,
                out: &mut Vec<Vec<StableEdge>>,
            ) {
                if current.len() == edge_count {
                    out.push(current.clone());
                    return;
                }
                for idx in start..pair_types.len() {
                    current.push(pair_types[idx].clone());
                    rec(pair_types, edge_count, idx, current, out);
                    current.pop();
                }
            }
            let mut all = Vec::new();
            rec(pair_types, edge_count, 0, &mut Vec::new(), &mut all);
            all.into_iter()
                .filter(|edges| {
                    StableGraph {
                        vertices: vec![StableVertex { genus: 0 }; vertex_count],
                        edges: edges.clone(),
                        legs: Vec::new(),
                    }
                    .is_connected()
                })
                .collect()
        }

        for vertex_count in 1..=4usize {
            let pair_types = edge_pair_types(vertex_count);
            for edge_count in 0..=5usize {
                let mut pruned = Vec::new();
                for_each_connected_edge_multiset(
                    vertex_count,
                    &pair_types,
                    edge_count,
                    &mut |edges| pruned.push(edges.to_vec()),
                );
                let expected = if vertex_count == 1 || edge_count + 1 >= vertex_count {
                    bruteforce(&pair_types, vertex_count, edge_count)
                } else {
                    Vec::new()
                };
                if vertex_count > 1 && edge_count + 1 < vertex_count {
                    assert!(pruned.is_empty());
                } else {
                    assert_eq!(
                        pruned, expected,
                        "multiset mismatch at V={vertex_count} E={edge_count}"
                    );
                }
            }
        }
    }
}
