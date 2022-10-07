//! Contains utilities for performing arithmetic over univariate polynomials in
//! various forms, including computing commitments to them and provably opening
//! the committed polynomials at arbitrary points.

use crate::arithmetic::{parallelize, Group};
use crate::plonk::Assigned;

use group::ff::{BatchInvert, Field};
use pasta_curves::arithmetic::FieldExt;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::io;
use std::marker::PhantomData;
use std::mem;
use std::ops::{Add, Deref, DerefMut, Index, IndexMut, Mul, RangeFrom, RangeFull};
use std::slice;

pub mod commitment;
mod domain;
mod evaluator;
pub mod multiopen;

pub use domain::*;
pub use evaluator::*;

/// This is an error that could occur during proving or circuit synthesis.
// TODO: these errors need to be cleaned up
#[derive(Debug)]
pub enum Error {
    /// OpeningProof is not well-formed
    OpeningError,
    /// Caller needs to re-sample a point
    SamplingError,
}

/// The basis over which a polynomial is described.
pub trait Basis: Copy + Debug + Send + Sync {}

/// The polynomial is defined as coefficients
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Coeff;
impl Basis for Coeff {}

/// The polynomial is defined as coefficients of Lagrange basis polynomials
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct LagrangeCoeff;
impl Basis for LagrangeCoeff {}

/// The polynomial is defined as coefficients of Lagrange basis polynomials in
/// an extended size domain which supports multiplication
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct ExtendedLagrangeCoeff;
impl Basis for ExtendedLagrangeCoeff {}

/// Represents a univariate polynomial defined over a field and a particular
/// basis.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct Polynomial<F, B> {
    values: Vec<F>,
    _marker: PhantomData<B>,
}

impl<F, B> Index<usize> for Polynomial<F, B> {
    type Output = F;

    fn index(&self, index: usize) -> &F {
        self.values.index(index)
    }
}

impl<F, B> IndexMut<usize> for Polynomial<F, B> {
    fn index_mut(&mut self, index: usize) -> &mut F {
        self.values.index_mut(index)
    }
}

impl<F, B> Index<RangeFrom<usize>> for Polynomial<F, B> {
    type Output = [F];

    fn index(&self, index: RangeFrom<usize>) -> &[F] {
        self.values.index(index)
    }
}

impl<F, B> IndexMut<RangeFrom<usize>> for Polynomial<F, B> {
    fn index_mut(&mut self, index: RangeFrom<usize>) -> &mut [F] {
        self.values.index_mut(index)
    }
}

impl<F, B> Index<RangeFull> for Polynomial<F, B> {
    type Output = [F];

    fn index(&self, index: RangeFull) -> &[F] {
        self.values.index(index)
    }
}

impl<F, B> IndexMut<RangeFull> for Polynomial<F, B> {
    fn index_mut(&mut self, index: RangeFull) -> &mut [F] {
        self.values.index_mut(index)
    }
}

impl<F, B> Deref for Polynomial<F, B> {
    type Target = [F];

    fn deref(&self) -> &[F] {
        &self.values[..]
    }
}

impl<F, B> DerefMut for Polynomial<F, B> {
    fn deref_mut(&mut self) -> &mut [F] {
        &mut self.values[..]
    }
}

impl<F, B> Polynomial<F, B> {
    /// Iterate over the values, which are either in coefficient or evaluation
    /// form depending on the basis `B`.
    pub fn iter(&self) -> impl Iterator<Item = &F> {
        self.values.iter()
    }

    /// Iterate over the values mutably, which are either in coefficient or
    /// evaluation form depending on the basis `B`.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut F> {
        self.values.iter_mut()
    }

    /// Gets the size of this polynomial in terms of the number of
    /// coefficients used to describe it.
    pub fn num_coeffs(&self) -> usize {
        self.values.len()
    }

    /// Transmutes the polynomial into bytes.
    #[allow(unsafe_code)]
    pub fn as_bytes(&self) -> &[u8] {
        let bytes_len = self.values.len() * mem::size_of::<F>();
        let bytes = unsafe { slice::from_raw_parts(self.values.as_ptr() as *const u8, bytes_len) };
        bytes
    }

    /// Writes polynomial to a buffer.
    pub fn write<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        let len =
            u32::try_from(self.values.len()).expect("Polynomials have less then 2^32 elements");
        writer.write_all(&len.to_le_bytes())?;

        let bytes = self.as_bytes();
        writer.write_all(&bytes)?;

        Ok(())
    }

    /// Transmutes the bytes into a polynomial.
    pub fn from_bytes(data: Vec<u8>) -> Self {
        let len = data.len() / mem::size_of::<F>();
        // Make sure the memory is not freed.
        let mut buffer_no_drop = mem::ManuallyDrop::new(data);
        let values =
            unsafe { Vec::from_raw_parts(buffer_no_drop.as_mut_ptr() as *mut F, len, len) };

        Polynomial {
            values,
            _marker: PhantomData,
        }
    }

    /// Reads polynomoal from a buffer.
    ///
    /// `len` is the number of elements.
    #[allow(unsafe_code)]
    pub fn read<R: io::Read>(reader: &mut R) -> io::Result<Self> {
        let mut buffer = [0u8; 4];
        reader.read_exact(&mut buffer[..])?;
        let len = usize::try_from(u32::from_le_bytes(buffer)).expect("Platform is at least 32-bit");

        // Create a bytes buffer the fulll polynomial is read in, which is the transmuted to the
        // final output value.
        let mut buffer = vec![0; len * mem::size_of::<F>()];
        reader.read_exact(&mut buffer[..])?;

        let polynomial = Self::from_bytes(buffer);
        Ok(polynomial)
    }
}

pub(crate) fn batch_invert_assigned<F: FieldExt>(
    assigned: Vec<Polynomial<Assigned<F>, LagrangeCoeff>>,
) -> Vec<Polynomial<F, LagrangeCoeff>> {
    let mut assigned_denominators: Vec<_> = assigned
        .iter()
        .map(|f| {
            f.iter()
                .map(|value| value.denominator())
                .collect::<Vec<_>>()
        })
        .collect();

    assigned_denominators
        .iter_mut()
        .flat_map(|f| {
            f.iter_mut()
                // If the denominator is trivial, we can skip it, reducing the
                // size of the batch inversion.
                .filter_map(|d| d.as_mut())
        })
        .batch_invert();

    assigned
        .iter()
        .zip(assigned_denominators.into_iter())
        .map(|(poly, inv_denoms)| {
            poly.invert(inv_denoms.into_iter().map(|d| d.unwrap_or_else(F::one)))
        })
        .collect()
}

impl<F: Field> Polynomial<Assigned<F>, LagrangeCoeff> {
    /// Creates a new empty polynomial of the given size.
    pub fn new(len: usize) -> Self
    where
        F: Clone + Group,
    {
        Self {
            values: vec![F::group_zero().into(); len],
            _marker: std::marker::PhantomData,
        }
    }

    pub(crate) fn invert(
        &self,
        inv_denoms: impl Iterator<Item = F> + ExactSizeIterator,
    ) -> Polynomial<F, LagrangeCoeff> {
        assert_eq!(inv_denoms.len(), self.values.len());
        Polynomial {
            values: self
                .values
                .iter()
                .zip(inv_denoms.into_iter())
                .map(|(a, inv_den)| a.numerator() * inv_den)
                .collect(),
            _marker: self._marker,
        }
    }
}

impl<'a, F: Field, B: Basis> Add<&'a Polynomial<F, B>> for Polynomial<F, B> {
    type Output = Polynomial<F, B>;

    fn add(mut self, rhs: &'a Polynomial<F, B>) -> Polynomial<F, B> {
        parallelize(&mut self.values, |lhs, start| {
            for (lhs, rhs) in lhs.iter_mut().zip(rhs.values[start..].iter()) {
                *lhs += *rhs;
            }
        });

        self
    }
}

impl<F: Field> Polynomial<F, LagrangeCoeff> {
    /// Rotates the values in a Lagrange basis polynomial by `Rotation`
    pub fn rotate(&self, rotation: Rotation) -> Polynomial<F, LagrangeCoeff> {
        let mut values = self.values.clone();
        if rotation.0 < 0 {
            values.rotate_right((-rotation.0) as usize);
        } else {
            values.rotate_left(rotation.0 as usize);
        }
        Polynomial {
            values,
            _marker: PhantomData,
        }
    }

    /// Gets the specified chunk of the rotated version of this polynomial.
    ///
    /// Equivalent to:
    /// ```ignore
    /// self.rotate(rotation)
    ///     .chunks(chunk_size)
    ///     .nth(chunk_index)
    ///     .unwrap()
    ///     .to_vec()
    /// ```
    pub(crate) fn get_chunk_of_rotated(
        &self,
        rotation: Rotation,
        chunk_size: usize,
        chunk_index: usize,
    ) -> Vec<F> {
        self.get_chunk_of_rotated_helper(
            rotation.0 < 0,
            rotation.0.unsigned_abs() as usize,
            chunk_size,
            chunk_index,
        )
    }
}

impl<F: Clone + Copy, B> Polynomial<F, B> {
    pub(crate) fn get_chunk_of_rotated_helper(
        &self,
        rotation_is_negative: bool,
        rotation_abs: usize,
        chunk_size: usize,
        chunk_index: usize,
    ) -> Vec<F> {
        //println!("vmx: get_chunk_of_rotated_helper: negative, abs, chunk size, chunk index: {} {} {} {}", rotation_is_negative, rotation_abs, chunk_size, chunk_index);
        // Compute the lengths such that when applying the rotation, the first `mid`
        // coefficients move to the end, and the last `k` coefficients move to the front.
        // The coefficient previously at `mid` will be the first coefficient in the
        // rotated polynomial, and the position from which chunk indexing begins.
        #[allow(clippy::branches_sharing_code)]
        let (mid, k) = if rotation_is_negative {
            let k = rotation_abs;
            assert!(k <= self.len());
            let mid = self.len() - k;
            (mid, k)
        } else {
            let mid = rotation_abs;
            assert!(mid <= self.len());
            let k = self.len() - mid;
            (mid, k)
        };
        //println!("vmx: get_chunk_of_rotated_helper: mid, k, len, abs: {} {} {} {}", mid, k, self.len(), rotation_abs);

        // Compute [chunk_start..chunk_end], the range of the chunk within the rotated
        // polynomial.
        let chunk_start = chunk_size * chunk_index;
        let chunk_end = self.len().min(chunk_size * (chunk_index + 1));
        //println!("vmx: get_chunk_of_rotated_helper: chunk_start, chunk_end: {} {}", chunk_start, chunk_end);

        if chunk_end < k {
            // The chunk is entirely in the last `k` coefficients of the unrotated
            // polynomial.
            //println!("vmx: get_chunk_of_rotated_helper: range1: {} {}", mid + chunk_start, mid + chunk_end);
            self.values[mid + chunk_start..mid + chunk_end].to_vec()
        } else if chunk_start >= k {
            // The chunk is entirely in the first `mid` coefficients of the unrotated
            // polynomial.
            //println!("vmx: get_chunk_of_rotated_helper: range2: {} {}", chunk_start - k, chunk_end - k);
            self.values[chunk_start - k..chunk_end - k].to_vec()
        } else {
            //println!("vmx: poly get of rotated helper: range3: {} {}", mid + chunk_start, chunk_end - k);
            // The chunk falls across the boundary between the last `k` and first `mid`
            // coefficients of the unrotated polynomial. Splice the halves together.
            let chunk = self.values[mid + chunk_start..]
                .iter()
                .chain(&self.values[..chunk_end - k])
                .copied()
                .collect::<Vec<_>>();
            assert!(chunk.len() <= chunk_size);
            chunk
        }
    }
}

impl<F: Field, B: Basis> Mul<F> for Polynomial<F, B> {
    type Output = Polynomial<F, B>;

    fn mul(mut self, rhs: F) -> Polynomial<F, B> {
        parallelize(&mut self.values, |lhs, _| {
            for lhs in lhs.iter_mut() {
                *lhs *= rhs;
            }
        });

        self
    }
}

/// Describes the relative rotation of a vector. Negative numbers represent
/// reverse (leftmost) rotations and positive numbers represent forward (rightmost)
/// rotations. Zero represents no rotation.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct Rotation(pub i32);

impl Rotation {
    /// The current location in the evaluation domain
    pub fn cur() -> Rotation {
        Rotation(0)
    }

    /// The previous location in the evaluation domain
    pub fn prev() -> Rotation {
        Rotation(-1)
    }

    /// The next location in the evaluation domain
    pub fn next() -> Rotation {
        Rotation(1)
    }
}

#[cfg(test)]
mod tests {
    use ff::Field;
    use pasta_curves::pallas;
    use rand_core::OsRng;

    use super::{EvaluationDomain, Rotation};

    #[test]
    fn test_get_chunk_of_rotated() {
        let k = 11;
        let domain = EvaluationDomain::<pallas::Base>::new(1, k);

        // Create a random polynomial.
        let mut poly = domain.empty_lagrange();
        for coefficient in poly.iter_mut() {
            *coefficient = pallas::Base::random(OsRng);
        }

        // Pick a chunk size that is guaranteed to not be a multiple of the polynomial
        // length.
        let chunk_size = 7;

        for rotation in [
            Rotation(-6),
            Rotation::prev(),
            Rotation::cur(),
            Rotation::next(),
            Rotation(12),
        ] {
            for (chunk_index, chunk) in poly.rotate(rotation).chunks(chunk_size).enumerate() {
                assert_eq!(
                    poly.get_chunk_of_rotated(rotation, chunk_size, chunk_index),
                    chunk
                );
            }
        }
    }
}
