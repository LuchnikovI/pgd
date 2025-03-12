use PauliMatrix::{I, X, Y, Z};
use pyo3::{pyclass, pymethods};
use std::cmp::max;
use std::fmt::Display;

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PauliMatrix {
    X,
    Y,
    Z,
    I,
}

impl Display for PauliMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            X => write!(f, "X")?,
            Y => write!(f, "Y")?,
            Z => write!(f, "Z")?,
            I => write!(f, "I")?,
        }
        Ok(())
    }
}

impl<'a, T: IntoIterator<Item = &'a PauliMatrix>> From<T> for PauliString {
    fn from(matrices: T) -> Self {
        matrices
            .into_iter()
            .enumerate()
            .fold(PauliString::default(), |ps, (pos, m)| match m {
                X => ps.set_x(pos),
                Y => ps.set_y(pos),
                Z => ps.set_z(pos),
                I => ps.set_id(pos),
            })
    }
}

// -----------------------------------------------------------------------------

#[pyclass]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct PauliString {
    x: usize,
    z: usize,
    n: usize,
}

impl Display for PauliString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for pos in 0..self.n {
            match ((self.x >> pos) & 1, (self.z >> pos) & 1) {
                (0, 0) => write!(f, "I")?,
                (1, 0) => write!(f, "X")?,
                (1, 1) => write!(f, "Y")?,
                (0, 1) => write!(f, "Z")?,
                _ => unreachable!(),
            }
        }
        Ok(())
    }
}

impl PauliString {
    fn new(n: usize) -> Self {
        Self { x: 0, z: 0, n }
    }
    pub(super) fn fixed_order_pauli_strings(
        order: usize,
        n: usize,
    ) -> Box<dyn Iterator<Item = PauliString>> {
        Self::_fixed_order_pauli_strings(order, n, n)
    }
    fn _fixed_order_pauli_strings(
        order: usize,
        n: usize,
        initial_n: usize,
    ) -> Box<dyn Iterator<Item = PauliString>> {
        if order == 0 {
            Box::new(Some(Self::new(initial_n)).into_iter())
        } else if n == 1 {
            let ps = Self::new(initial_n);
            Box::new([ps.set_x(0), ps.set_y(0), ps.set_z(0)].into_iter())
        } else {
            Box::new(((order - 1)..n).flat_map(move |prev_n| {
                Self::_fixed_order_pauli_strings(order - 1, prev_n, initial_n).flat_map(move |ps| {
                    [ps.set_x(prev_n), ps.set_y(prev_n), ps.set_z(prev_n)].into_iter()
                })
            }))
        }
    }
    #[inline(always)]
    fn get_n_and_mask(&self, pos: usize) -> (usize, usize) {
        let n = max(pos + 1, self.n);
        assert!(n < usize::BITS as usize);
        let mask = 1 << pos;
        (n, mask)
    }
    #[inline(always)]
    fn flip_by_x(&self, elem: usize) -> usize {
        self.x ^ elem
    }
    #[inline(always)]
    fn sign_by_z(&self, elem: usize) -> i8 {
        1 - 2 * ((elem & self.z).count_ones() & 1) as i8
    }
    #[inline(always)]
    fn apply_to_basis_elem(&self, elem: usize) -> (i8, usize) {
        (self.sign_by_z(elem), self.flip_by_x(elem))
    }
    #[inline(always)]
    fn set_x(self, pos: usize) -> Self {
        let (n, mask) = self.get_n_and_mask(pos);
        Self {
            x: self.x | mask,
            z: self.z,
            n,
        }
    }
    #[inline(always)]
    fn set_y(self, pos: usize) -> Self {
        let (n, mask) = self.get_n_and_mask(pos);
        Self {
            x: self.x | mask,
            z: self.z | mask,
            n,
        }
    }
    #[inline(always)]
    fn set_z(self, pos: usize) -> Self {
        let (n, mask) = self.get_n_and_mask(pos);
        Self {
            x: self.x,
            z: self.z | mask,
            n,
        }
    }
    #[inline(always)]
    fn set_id(self, pos: usize) -> Self {
        let (n, _) = self.get_n_and_mask(pos);
        Self {
            x: self.x,
            z: self.z,
            n,
        }
    }
    pub(super) fn iter_nonzero_elements(&self) -> impl Iterator<Item = (usize, usize, i8)> {
        (0..usize::MAX).map(|rhs_idx| {
            let (sgn, lhs_idx) = self.apply_to_basis_elem(rhs_idx);
            (lhs_idx, rhs_idx, sgn)
        })
    }
}

#[pymethods]
impl PauliString {
    fn __str__(&self) -> String {
        format!("{}", self)
    }
    fn __repr__(&self) -> String {
        self.__str__()
    }
    #[inline(always)]
    pub(super) fn size(&self) -> usize {
        self.n
    }
    #[inline(always)]
    pub(super) fn get_order(&self) -> usize {
        (self.x | self.z).count_ones() as usize
    }
}

#[cfg(test)]
mod tests {
    use super::{I, PauliMatrix, PauliString, X, Y, Z};
    use lazy_static::lazy_static;
    use ndarray::{Array2, array, linalg::kron};
    use rand::distr::{Distribution, StandardUniform};
    use rand::{Rng, rng};
    use std::collections::HashSet;

    lazy_static! {
        static ref iden: Array2<i8> = array![[1, 0], [0, 1]];
        static ref pauli_x: Array2<i8> = array![[0, 1], [1, 0]];
        static ref pauli_y: Array2<i8> = array![[0, -1], [1, 0]];
        static ref pauli_z: Array2<i8> = array![[1, 0], [0, -1]];
    }

    impl Distribution<PauliMatrix> for StandardUniform {
        fn sample<R: Rng + ?Sized>(&self, prng: &mut R) -> PauliMatrix {
            match prng.random::<u8>() % 4 {
                0 => I,
                1 => X,
                2 => Y,
                3 => Z,
                _ => unreachable!(),
            }
        }
    }

    fn gen_random_pauli_matrices(prng: &mut impl Rng, size: usize) -> Vec<PauliMatrix> {
        (0..size).map(|_| prng.random()).collect()
    }

    impl From<PauliMatrix> for &'static Array2<i8> {
        fn from(pauli_matrix: PauliMatrix) -> Self {
            match pauli_matrix {
                X => &pauli_x,
                Y => &pauli_y,
                Z => &pauli_z,
                I => &iden,
            }
        }
    }

    impl From<PauliString> for Array2<i8> {
        fn from(pauli_string: PauliString) -> Self {
            let size = 1 << pauli_string.n;
            let mut arr = Array2::zeros([size, size]);
            for (lhs_idx, rhs_idx, val) in pauli_string.iter_nonzero_elements().take(size) {
                *arr.get_mut([lhs_idx, rhs_idx]).unwrap() = val;
            }
            arr
        }
    }

    fn get_correct_array(pauli_string: &[PauliMatrix]) -> Array2<i8> {
        pauli_string.iter().rev().fold(array![[1]], |acc, m| {
            kron(&acc, Into::<&'static Array2<i8>>::into(*m))
        })
    }

    fn get_array(pauli_string: &[PauliMatrix]) -> Array2<i8> {
        let pauli_string: PauliString = pauli_string.into();
        pauli_string.into()
    }

    #[test]
    fn test_non_zero_elements_iter() {
        for size in 0..5 {
            for _ in 0..10 {
                let mut prng = rng();
                let pauli_string = gen_random_pauli_matrices(&mut prng, size);
                {
                    let pauli_string: PauliString = (&pauli_string).into();
                    assert_eq!(size, pauli_string.size());
                }
                let correct_arr = get_correct_array(&pauli_string);
                let arr = get_array(&pauli_string);
                (arr - correct_arr).iter().all(|diff| *diff == 0);
            }
        }
    }

    fn fac(n: usize) -> usize {
        let mut acc = 1;
        for mul in 2..=n {
            acc *= mul;
        }
        acc
    }

    fn number_of_pauli_strings(order: usize, n: usize) -> usize {
        3usize.pow(order as u32) * fac(n) / (fac(n - order) * fac(order))
    }

    #[test]
    fn test_pauli_strings_iter() {
        for n in 1..=10 {
            for order in 0..=n {
                let strings_number = number_of_pauli_strings(order, n);
                let mut cache = HashSet::with_capacity(strings_number);
                for ps in PauliString::fixed_order_pauli_strings(order, n) {
                    assert!(ps.get_order() == order);
                    assert!(ps.size() == n);
                    assert!(cache.insert(ps));
                }
                assert!(cache.len() == strings_number)
            }
        }
    }
}
