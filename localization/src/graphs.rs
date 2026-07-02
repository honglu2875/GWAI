//! Stable graphs for the boundary strata of `Mbar_{g,n}`.
//!
//! These are the abstract stable-curve graphs used by the Givental graph
//! expansion, not stable-map localization graphs.  Vertices carry genera, legs
//! are labelled markings, and automorphism orders include both vertex
//! symmetries and bijections of repeated edges.

use std::cmp::Ordering;
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

/// Complete isomorphism invariant of a stable graph: the lexicographically
/// smallest `(genera, sorted edges, legs)` tuple over the vertex relabelings
/// allowed by [`refined_vertex_classes`].
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
            .all(|(idx, vertex)| 2 * vertex.genus + self.valence(idx) > 2)
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

    /// Canonical form over class-respecting relabelings.
    ///
    /// Only permutations preserving the refined vertex classes can realize an
    /// isomorphism, so restricting the minimization to them yields the same
    /// equivalence classes as a full `V!` sweep at a fraction of the cost.
    pub(crate) fn canonical_key(&self) -> CanonicalGraphKey {
        let vertex_count = self.vertices.len();
        if vertex_count == 0 {
            return CanonicalGraphKey::default();
        }
        let classes = refined_vertex_classes(self);
        let mut targets = Vec::with_capacity(classes.len());
        let mut next_position = 0usize;
        for class in &classes {
            targets.push((next_position..next_position + class.len()).collect::<Vec<_>>());
            next_position += class.len();
        }

        let mut best: Option<CanonicalGraphKey> = None;
        for_each_class_assignment(&classes, &targets, vertex_count, &mut |permutation| {
            let key = self.key_with_permutation(permutation);
            if best.as_ref().is_none_or(|current| key < *current) {
                best = Some(key);
            }
        });
        best.expect("at least one class-respecting permutation exists")
    }

    pub fn automorphism_order(&self) -> usize {
        let vertex_count = self.vertices.len();
        if vertex_count == 0 {
            return 1;
        }
        let classes = refined_vertex_classes(self);
        // Automorphisms map each vertex to a same-class vertex, so each class
        // permutes over its own original index positions.
        let targets = classes.clone();
        let base = self.key_with_permutation(&(0..vertex_count).collect::<Vec<_>>());
        let mut total = 0usize;
        for_each_class_assignment(&classes, &targets, vertex_count, &mut |permutation| {
            if self.key_with_permutation(permutation) == base {
                total += edge_bijection_count_after_permutation(self, permutation);
            }
        });
        total
    }

    pub fn vertex_automorphism_permutations(&self) -> Vec<Vec<usize>> {
        let vertex_count = self.vertices.len();
        if vertex_count == 0 {
            return vec![Vec::new()];
        }
        let classes = refined_vertex_classes(self);
        let targets = classes.clone();
        let base = self.key_with_permutation(&(0..vertex_count).collect::<Vec<_>>());
        let mut out = Vec::new();
        for_each_class_assignment(&classes, &targets, vertex_count, &mut |permutation| {
            if self.key_with_permutation(permutation) == base {
                out.push(permutation.to_vec());
            }
        });
        out
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

/// Partition of the vertices into isomorphism-invariant classes.
///
/// Starts from the local invariant `(genus, loop count, marking labels,
/// non-loop multiplicity multiset)` and refines by neighbor classes until
/// stable (Weisfeiler-Leman color refinement).  Class ids — and therefore the
/// order of the returned classes — depend only on invariant data, never on
/// vertex labels, so isomorphic graphs produce corresponding partitions.
fn refined_vertex_classes(graph: &StableGraph) -> Vec<Vec<usize>> {
    let vertex_count = graph.vertices.len();
    if vertex_count == 0 {
        return Vec::new();
    }

    let mut adjacency = vec![BTreeMap::<usize, usize>::new(); vertex_count];
    let mut loops = vec![0usize; vertex_count];
    for edge in &graph.edges {
        if edge.is_loop() {
            loops[edge.a] += 1;
        } else {
            *adjacency[edge.a].entry(edge.b).or_default() += 1;
            *adjacency[edge.b].entry(edge.a).or_default() += 1;
        }
    }
    let mut leg_labels = vec![Vec::<usize>::new(); vertex_count];
    for (marking, &vertex) in graph.legs.iter().enumerate() {
        leg_labels[vertex].push(marking);
    }

    let initial = (0..vertex_count)
        .map(|vertex| {
            let mut multiplicities = adjacency[vertex].values().copied().collect::<Vec<_>>();
            multiplicities.sort_unstable();
            (
                graph.vertices[vertex].genus,
                loops[vertex],
                leg_labels[vertex].clone(),
                multiplicities,
            )
        })
        .collect::<Vec<_>>();
    let mut colors = colors_from_signatures(&initial);

    loop {
        let signatures = (0..vertex_count)
            .map(|vertex| {
                let mut neighbor_colors = adjacency[vertex]
                    .iter()
                    .map(|(&neighbor, &multiplicity)| (colors[neighbor], multiplicity))
                    .collect::<Vec<_>>();
                neighbor_colors.sort_unstable();
                (colors[vertex], neighbor_colors)
            })
            .collect::<Vec<_>>();
        let next = colors_from_signatures(&signatures);
        if next == colors {
            break;
        }
        colors = next;
    }

    let class_count = colors.iter().max().map(|max| max + 1).unwrap_or(0);
    let mut classes = vec![Vec::new(); class_count];
    for (vertex, &color) in colors.iter().enumerate() {
        classes[color].push(vertex);
    }
    classes
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

/// Visits every permutation (`permutation[new] = old`) that assigns each
/// class's vertices onto that class's target positions, in every order.
fn for_each_class_assignment(
    classes: &[Vec<usize>],
    targets: &[Vec<usize>],
    vertex_count: usize,
    visit: &mut impl FnMut(&[usize]),
) {
    fn rec(
        classes: &mut [Vec<usize>],
        targets: &[Vec<usize>],
        class_index: usize,
        position: usize,
        permutation: &mut [usize],
        visit: &mut impl FnMut(&[usize]),
    ) {
        if class_index == classes.len() {
            visit(permutation);
            return;
        }
        let class_size = classes[class_index].len();
        if position == class_size {
            rec(classes, targets, class_index + 1, 0, permutation, visit);
            return;
        }
        for swap in position..class_size {
            classes[class_index].swap(position, swap);
            permutation[targets[class_index][position]] = classes[class_index][position];
            rec(
                classes,
                targets,
                class_index,
                position + 1,
                permutation,
                visit,
            );
            classes[class_index].swap(position, swap);
        }
    }

    debug_assert_eq!(
        classes.iter().map(Vec::len).sum::<usize>(),
        vertex_count,
        "classes partition the vertices"
    );
    debug_assert!(classes
        .iter()
        .zip(targets.iter())
        .all(|(class, target)| class.len() == target.len()));
    let mut scratch = classes.to_vec();
    let mut permutation = vec![0usize; vertex_count];
    rec(&mut scratch, targets, 0, 0, &mut permutation, visit);
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
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();

    for vertex_count in 1..=bounds.max_vertices.max(1) {
        let pair_types = edge_pair_types(vertex_count);
        let leg_assignments = labelled_leg_assignments(legs, vertex_count);
        let min_edges = vertex_count - 1;
        let max_edges = genus + vertex_count - 1;
        for edge_count in min_edges..=max_edges {
            let h1 = edge_count + 1 - vertex_count;
            let genus_sum = genus - h1;
            // Upper bound on the decorations a multiset with this vertex and
            // edge count can carry; the skeleton-level quotient only pays off
            // when its sweep amortizes over enough decorations.  Both inputs
            // are isomorphism-invariant, so isomorphic multisets always take
            // the same dedupe path.
            let decoration_scale = leg_assignments
                .len()
                .saturating_mul(compositions_count(genus_sum, vertex_count));
            for_each_connected_edge_multiset(vertex_count, &pair_types, edge_count, &mut |edges| {
                // Computed lazily: multisets whose valences admit no stable
                // decoration never pay for a canonicality sweep.
                let mut quotient_cache: Option<SkeletonQuotient> = None;

                let mut edge_valence = vec![0usize; vertex_count];
                for edge in edges {
                    edge_valence[edge.a] += 1;
                    edge_valence[edge.b] += 1;
                }
                for leg_assignment in &leg_assignments {
                    let mut valence = edge_valence.clone();
                    for &vertex in leg_assignment {
                        valence[vertex] += 1;
                    }
                    // Stability 2g_v + val_v > 2 gives a per-vertex genus
                    // minimum; distributing only the remaining genus over the
                    // vertices enumerates exactly the stable assignments.
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
                        let quotient = quotient_cache.get_or_insert_with(|| {
                            skeleton_quotient(vertex_count, edges, decoration_scale)
                        });
                        let automorphisms = match quotient {
                            SkeletonQuotient::Canonical(automorphisms) => Some(&*automorphisms),
                            // A relabelled isomorph of a canonical multiset
                            // elsewhere in the enumeration: skip it entirely.
                            SkeletonQuotient::NotCanonical => return,
                            // Coarse skeleton classes (near-regular
                            // multigraphs) or too few decorations to amortize
                            // the sweep: dedupe per decorated graph, where
                            // genera and legs refine the classes.
                            SkeletonQuotient::PerDecoration => None,
                        };
                        let genera = genus_minimums
                            .iter()
                            .zip(extra.iter())
                            .map(|(&minimum, &extra_genus)| minimum + extra_genus)
                            .collect::<Vec<_>>();
                        if let Some(automorphisms) = automorphisms {
                            if !decoration_is_orbit_minimal(automorphisms, &genera, leg_assignment)
                            {
                                continue;
                            }
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
                        if automorphisms.is_none() && !seen.insert(graph.canonical_key()) {
                            continue;
                        }
                        out.push(graph);
                    }
                }
            });
        }
    }

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

enum SkeletonQuotient {
    /// Canonical representative; carries the skeleton vertex automorphisms
    /// (encoded as `permutation[new] = old`).
    Canonical(Vec<Vec<usize>>),
    /// A relabelled isomorph of a canonical multiset elsewhere in the
    /// enumeration.
    NotCanonical,
    /// Skeleton-level quotient not worthwhile here; dedupe per decoration.
    PerDecoration,
}

/// Cap on `prod_i k_i!` over the refined skeleton classes, above which the
/// skeleton-level quotient falls back to per-decoration canonical dedupe.
const SKELETON_ORBIT_SWEEP_LIMIT: usize = 720;

fn skeleton_quotient(
    vertex_count: usize,
    edges: &[StableEdge],
    decoration_scale: usize,
) -> SkeletonQuotient {
    let skeleton = StableGraph {
        vertices: vec![StableVertex { genus: 0 }; vertex_count],
        edges: edges.to_vec(),
        legs: Vec::new(),
    };
    let classes = refined_vertex_classes(&skeleton);
    let mut sweep_size = 1usize;
    for class in &classes {
        sweep_size = sweep_size.saturating_mul(factorial(class.len()));
    }
    // The two sweeps below cost about 2 * sweep_size key constructions; the
    // decorated path costs roughly one canonical key per stable decoration.
    // Prefer the skeleton quotient only when it can amortize.
    if sweep_size > SKELETON_ORBIT_SWEEP_LIMIT || 2 * sweep_size > decoration_scale {
        return SkeletonQuotient::PerDecoration;
    }
    let identity_key = skeleton.key_with_permutation(&(0..vertex_count).collect::<Vec<_>>());

    // Canonicality: the identity key must *equal* the minimal key over the
    // block-ordered class-respecting relabelings.  Comparing with `<` alone
    // would be wrong: the identity arrangement is generally not among the
    // block arrangements, so a merely differently-arranged labelling could
    // pass while a relabelled isomorph of it also passes.
    let mut identity_is_minimal = false;
    let mut smaller_exists = false;
    let mut targets = Vec::with_capacity(classes.len());
    let mut next_position = 0usize;
    for class in &classes {
        targets.push((next_position..next_position + class.len()).collect::<Vec<_>>());
        next_position += class.len();
    }
    for_each_class_assignment(&classes, &targets, vertex_count, &mut |permutation| {
        if smaller_exists {
            return;
        }
        let key = skeleton.key_with_permutation(permutation);
        match key.cmp(&identity_key) {
            Ordering::Less => smaller_exists = true,
            Ordering::Equal => identity_is_minimal = true,
            Ordering::Greater => {}
        }
    });
    if smaller_exists || !identity_is_minimal {
        return SkeletonQuotient::NotCanonical;
    }

    // Automorphisms: class members permute over their own original positions.
    let mut automorphisms = Vec::new();
    for_each_class_assignment(&classes, &classes, vertex_count, &mut |permutation| {
        if skeleton.key_with_permutation(permutation) == identity_key {
            automorphisms.push(permutation.to_vec());
        }
    });
    SkeletonQuotient::Canonical(automorphisms)
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
fn for_each_connected_edge_multiset(
    vertex_count: usize,
    pair_types: &[StableEdge],
    edge_count: usize,
    visit: &mut impl FnMut(&[StableEdge]),
) {
    fn rec(
        pair_types: &[StableEdge],
        last_touch: &[usize],
        edge_count: usize,
        start: usize,
        current: &mut Vec<StableEdge>,
        degree: &mut [usize],
        dsu: DisjointSet,
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
            // Once an untouched vertex has no incident pair at or beyond idx,
            // no extension from here can connect it.
            if degree
                .iter()
                .enumerate()
                .any(|(vertex, &d)| d == 0 && last_touch[vertex] < idx)
            {
                break;
            }
            let pair = &pair_types[idx];
            let mut next_dsu = dsu.clone();
            let mut next_components = components;
            if !pair.is_loop() && next_dsu.union_merged(pair.a, pair.b) {
                next_components -= 1;
            }
            degree[pair.a] += 1;
            degree[pair.b] += 1;
            current.push(pair.clone());
            rec(
                pair_types,
                last_touch,
                edge_count,
                idx,
                current,
                degree,
                next_dsu,
                next_components,
                visit,
            );
            current.pop();
            degree[pair.b] -= 1;
            degree[pair.a] -= 1;
        }
    }

    if edge_count == 0 {
        if vertex_count == 1 {
            visit(&[]);
        }
        return;
    }
    let mut last_touch = vec![0usize; vertex_count];
    for (idx, pair) in pair_types.iter().enumerate() {
        last_touch[pair.a] = idx;
        last_touch[pair.b] = idx;
    }
    rec(
        pair_types,
        &last_touch,
        edge_count,
        0,
        &mut Vec::with_capacity(edge_count),
        &mut vec![0usize; vertex_count],
        DisjointSet::new(vertex_count),
        vertex_count,
        visit,
    );
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

/// Number of compositions of `total` into `parts` nonnegative parts,
/// `C(total + parts - 1, parts - 1)`, saturating on overflow.
fn compositions_count(total: usize, parts: usize) -> usize {
    if parts == 0 {
        return 0;
    }
    let mut out = 1u128;
    for k in 1..parts {
        out = out.saturating_mul((total + k) as u128) / k as u128;
        if out > usize::MAX as u128 {
            return usize::MAX;
        }
    }
    out as usize
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

#[cfg(test)]
mod tests {
    use super::*;

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
