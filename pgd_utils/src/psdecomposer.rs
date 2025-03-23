use numpy::PyArray1;
use pyo3::{Bound, PyResult, Python, exceptions::PyValueError, pyclass, pyfunction, pymethods};
use rayon::iter::{
    IntoParallelIterator, ParallelBridge, ParallelIterator,
};
use std::{
    cmp::{max, min},
    collections::{HashMap, hash_map::Entry::Vacant},
};

use crate::pauli_string::PauliString;

fn permutations(k: usize) -> Box<dyn Iterator<Item = (usize, usize)>> {
    if k == 1 {
        Box::new(None.into_iter())
    } else {
        Box::new(permutations(k - 1).chain((0..(k - 1)).flat_map(move |i| {
            if k % 2 == 0 {
                Box::new(Some((i, k - 1)).into_iter())
            } else {
                Box::new(Some((0, k - 1)).into_iter())
            }
            .chain(permutations(k - 1))
        })))
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
struct Edge {
    lhs: usize,
    rhs: usize,
}

impl Edge {
    fn new(id1: usize, id2: usize) -> Self {
        let lhs = min(id1, id2);
        let rhs = max(id1, id2);
        Self { lhs, rhs }
    }
    #[inline(always)]
    fn swap_nodes(self, id1: usize, id2: usize) -> Self {
        let new_id1 = if self.lhs == id1 {
            id2
        } else if self.lhs == id2 {
            id1
        } else {
            self.lhs
        };
        let new_id2 = if self.rhs == id1 {
            id2
        } else if self.rhs == id2 {
            id1
        } else {
            self.rhs
        };
        let lhs = min(new_id1, new_id2);
        let rhs = max(new_id1, new_id2);
        Self { lhs, rhs }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub(super) struct PSDecomposer {
    edges: HashMap<Edge, f64>,
    nodes_number: usize,
}

#[inline(always)]
fn log_round(val: usize) -> usize {
    let mut log_val = 0;
    if val == 0 {
        panic!("log of 0")
    } else if val == 1 {
        return 1;
    };
    while val > 1 << log_val {
        log_val += 1;
    }
    log_val
}

impl PSDecomposer {
    fn swap(&mut self, id1: usize, id2: usize) {
        let edges = std::mem::take(&mut self.edges);
        self.edges = edges
            .into_iter()
            .map(|(edge, ampl)| (edge.swap_nodes(id1, id2), ampl))
            .collect()
    }
    fn pauli_overlap_iter(
        &self,
        order: usize,
        eps: f64,
    ) -> impl ParallelIterator<Item = (f64, PauliString)> {
        let n = self.rounded_size();
        let pauli_strings: Vec<_> = PauliString::fixed_order_pauli_strings(order, n).collect();
        pauli_strings
            .into_par_iter()
            .map(move |ps| {
                (
                    ps.iter_nonzero_elements()
                        .take(1 << n)
                        .filter_map(|(id1, id2, mul)| {
                            let edge = Edge::new(id1, id2);
                            let multiplicity = if id1 == id2 { 1f64 } else { 0.5f64 };
                            self.edges
                                .get(&edge)
                                .map(|ampl| multiplicity * ampl * (mul as f64))
                        })
                        .sum::<f64>()
                        / ((1 << n) as f64),
                    ps,
                )
            })
            .filter(move |x| x.0.abs() > eps)
    }
    fn l1(&self, eps: f64) -> f64 {
        let max_order = self.rounded_size();
        (0..=max_order)
            .par_bridge()
            .flat_map(|order| self.pauli_overlap_iter(order, eps))
            .map(|(ampl, _)| ampl.abs())
            .sum()
    }
}

#[pymethods]
impl PSDecomposer {
    #[new]
    fn new() -> Self {
        Self {
            edges: HashMap::new(),
            nodes_number: 0,
        }
    }
    fn add_edge(&mut self, id1: usize, id2: usize, ampl: f64) -> PyResult<()> {
        let new_edge = Edge::new(id1, id2);
        if let Vacant(e) = self.edges.entry(new_edge) {
            self.nodes_number = max(self.nodes_number, new_edge.rhs + 1);
            e.insert(ampl);
            Ok(())
        } else {
            Err(PyValueError::new_err(format!(
                "An edge connecting nodes {} and {} already exists",
                id1, id2
            )))
        }
    }
    fn size(&self) -> usize {
        self.nodes_number
    }
    #[pyo3(signature = (
        order,
        eps = 1e-10,
    ))]
    fn pauli_overlap<'a>(
        &self,
        order: usize,
        eps: f64,
        py: Python<'a>,
    ) -> (Vec<PauliString>, Bound<'a, PyArray1<f64>>) {
        let (ampls, pauli_strings): (Vec<_>, Vec<_>) = self.pauli_overlap_iter(order, eps).unzip();
        (pauli_strings, PyArray1::from_vec(py, ampls))
    }
    fn rounded_size(&self) -> usize {
        log_round(self.size())
    }
    #[pyo3(signature = (
        eps = 1e-10,
    ))]
    fn decompose<'a>(
        &self,
        eps: f64,
        py: Python<'a>,
    ) -> Vec<(Vec<PauliString>, Bound<'a, PyArray1<f64>>)> {
        let max_order = self.rounded_size();
        (0..=max_order)
            .map(|order| self.pauli_overlap(order, eps, py))
            .collect()
    }
    #[pyo3(signature = (
        eps = 1e-10,
    ))]
    fn pauli_optimize(&mut self, eps: f64) -> Vec<usize> {
        let mut best_self = self.clone();
        let mut optimal_order: Vec<_> = (0..self.nodes_number).collect();
        let mut current_order: Vec<_> = optimal_order.clone();
        let mut l1 = best_self.l1(eps);
        for (node1, node2) in permutations(self.nodes_number) {
            self.swap(node1, node2);
            current_order.swap(node1, node2);
            let new_l1 = self.l1(eps);
            if new_l1 < l1 {
                l1 = new_l1;
                best_self = self.clone();
                optimal_order = current_order.clone();
            }
        }
        *self = best_self;
        optimal_order
    }
}

#[pyfunction]
pub(super) fn parse_gset(data: &str) -> PyResult<PSDecomposer> {
    let mut lines_iter = data.lines();
    let mut header_iter = lines_iter
        .next()
        .ok_or(PyValueError::new_err("Empty GSet file"))?
        .split_whitespace();
    let _ = header_iter
        .next()
        .ok_or(PyValueError::new_err("Empty header"))?
        .parse::<usize>()?;
    let _ = header_iter
        .next()
        .ok_or(PyValueError::new_err("Empty edges number field"))?
        .parse::<usize>()?;
    let mut graph = PSDecomposer::new();
    for (num, line) in lines_iter.enumerate() {
        let mut line = line.split_whitespace();
        let id1 = line
            .next()
            .ok_or(PyValueError::new_err(format!("Empty line #{num}")))?
            .parse::<usize>()?;
        let id2 = line
            .next()
            .ok_or(PyValueError::new_err(format!(
                "Empty second id of line #{num}"
            )))?
            .parse::<usize>()?;
        let ampl = line
            .next()
            .ok_or(PyValueError::new_err(format!(
                "Empty amplitude of line #{num}"
            )))?
            .parse::<f64>()?;
        graph.add_edge(id1, id2, ampl)?;
    }
    Ok(graph)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::{log_round, permutations};

    #[test]
    fn test_log_round() {
        assert_eq!(1, log_round(1));
        assert_eq!(5, log_round(20));
        assert_eq!(5, log_round(32));
        assert_eq!(11, log_round(1025));
    }

    fn _fac(k: usize) -> usize {
        if k == 1 { 1 } else { k * _fac(k - 1) }
    }

    fn _test_permutations(k: usize) {
        let mut v: Vec<_> = (0..k).collect();
        let mut seen = HashSet::from([v.clone()]);
        for (lhs, rhs) in permutations(k) {
            v.swap(lhs, rhs);
            assert!(seen.insert(v.clone()));
        }
        assert_eq!(_fac(k), seen.len());
    }

    #[test]
    fn test_permutations() {
        _test_permutations(1);
        _test_permutations(2);
        _test_permutations(3);
        _test_permutations(4);
        _test_permutations(5);
        _test_permutations(6);
        _test_permutations(7);
        _test_permutations(8);
    }
}
