use pauli_string::PauliString;
use pyo3::{Bound, PyResult, pymodule, types::PyModule, types::PyModuleMethods, wrap_pyfunction};

mod psdecomposer;
mod pauli_string;

use psdecomposer::{PSDecomposer, parse_gset};

#[pymodule]
fn pgd_utils(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_gset, m)?)?;
    m.add_class::<PSDecomposer>()?;
    m.add_class::<PauliString>()?;
    Ok(())
}
