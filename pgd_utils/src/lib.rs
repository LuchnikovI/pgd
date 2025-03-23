use pauli_string::PauliString;
use pyo3::{Bound, PyResult, pymodule, types::PyModule, types::PyModuleMethods, wrap_pyfunction};

mod pauli_string;
mod psdecomposer;

use psdecomposer::{PSDecomposer, parse_gset};

#[pymodule]
fn pgd_utils(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_gset, m)?)?;
    m.add_class::<PSDecomposer>()?;
    m.add_class::<PauliString>()?;
    Ok(())
}
