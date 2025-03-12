use numpy::PyArray1;
use pyo3::{Bound, PyResult, Python, exceptions::PyValueError, pyclass, pyfunction, pymethods};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::{
    cmp::{max, min},
    collections::{HashMap, hash_map::Entry::Vacant},
};

use crate::pauli_string::PauliString;

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
        let n = log_round(self.size());
        let pauli_strings: Vec<_> = PauliString::fixed_order_pauli_strings(order, n).collect();
        let (ampls, pauli_strings): (Vec<_>, Vec<_>) = pauli_strings
            .par_iter()
            .map(|ps| {
                (
                    ps.iter_nonzero_elements()
                        .take(1 << n)
                        .filter_map(|(id1, id2, mul)| {
                            let edge = Edge::new(id1, id2);
                            self.edges.get(&edge).map(|ampl| ampl * (mul as f64))
                        })
                        .sum::<f64>()
                        / ((1 << n) as f64),
                    ps,
                )
            })
            .filter(|x| x.0.abs() > eps)
            .unzip();
        (pauli_strings, PyArray1::from_vec(py, ampls))
    }
    fn rounded_size(&self) -> usize {
        log_round(self.size())
    }
    #[pyo3(signature = (
        eps = 1e-10,
    ))]
    fn decompose<'a>(&self,
        eps: f64,
        py: Python<'a>,
    ) -> Vec<(Vec<PauliString>, Bound<'a, PyArray1<f64>>)> {
        let max_order = self.rounded_size();
        (0..=max_order).map(|order| self.pauli_overlap(order, eps, py)).collect()
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
    use super::log_round;

    #[test]
    fn test_log_round() {
        assert_eq!(1, log_round(1));
        assert_eq!(5, log_round(20));
        assert_eq!(5, log_round(32));
        assert_eq!(11, log_round(1025));
    }
}
