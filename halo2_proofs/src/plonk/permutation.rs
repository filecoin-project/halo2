use super::{
    circuit::{Any, Column},
    AFFINE_SIZE,
};
use crate::{
    arithmetic::CurveAffine,
    poly::{Coeff, ExtendedLagrangeCoeff, LagrangeCoeff, Polynomial},
};

pub(crate) mod keygen;
pub(crate) mod prover;
pub(crate) mod verifier;

use std::io;

/// A permutation argument.
#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct Argument {
    /// A sequence of columns involved in the argument.
    columns: Vec<Column<Any>>,
}

impl Argument {
    pub(crate) fn new() -> Self {
        Argument { columns: vec![] }
    }

    /// Returns the minimum circuit degree required by the permutation argument.
    /// The argument may use larger degree gates depending on the actual
    /// circuit's degree and how many columns are involved in the permutation.
    pub(crate) fn required_degree(&self) -> usize {
        // degree 2:
        // l_0(X) * (1 - z(X)) = 0
        //
        // We will fit as many polynomials p_i(X) as possible
        // into the required degree of the circuit, so the
        // following will not affect the required degree of
        // this middleware.
        //
        // (1 - (l_last(X) + l_blind(X))) * (
        //   z(\omega X) \prod (p(X) + \beta s_i(X) + \gamma)
        // - z(X) \prod (p(X) + \delta^i \beta X + \gamma)
        // )
        //
        // On the first sets of columns, except the first
        // set, we will do
        //
        // l_0(X) * (z(X) - z'(\omega^(last) X)) = 0
        //
        // where z'(X) is the permutation for the previous set
        // of columns.
        //
        // On the final set of columns, we will do
        //
        // degree 3:
        // l_last(X) * (z'(X)^2 - z'(X)) = 0
        //
        // which will allow the last value to be zero to
        // ensure the argument is perfectly complete.

        // There are constraints of degree 3 regardless of the
        // number of columns involved.
        3
    }

    pub(crate) fn add_column(&mut self, column: Column<Any>) {
        if !self.columns.contains(&column) {
            self.columns.push(column);
        }
    }

    pub(crate) fn get_columns(&self) -> Vec<Column<Any>> {
        self.columns.clone()
    }

    /// Returns the number of columns.
    pub(crate) fn len(&self) -> usize {
        self.columns.len()
    }
}

/// The verifying key for a single permutation argument.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct VerifyingKey<C: CurveAffine> {
    commitments: Vec<C>,
}

impl<C: CurveAffine> VerifyingKey<C> {
    /// Writes verifying key to a buffer.
    #[allow(unsafe_code)]
    pub fn write<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        let byte_len = self.commitments.len() * AFFINE_SIZE;
        let bytes_ptr = self.commitments.as_ptr() as *const u8;
        let bytes = unsafe { std::slice::from_raw_parts(bytes_ptr, byte_len) };
        writer.write_all(&bytes)?;

        Ok(())
    }

    /// Reads verifying key from a buffer.
    #[allow(unsafe_code)]
    pub fn read<R: io::Read>(reader: &mut R, commitments_len: usize) -> io::Result<Self> {
        let byte_len = commitments_len * AFFINE_SIZE;
        let mut buf = vec![0u8; byte_len];
        reader.read_exact(&mut buf)?;
        let mut buf_no_drop = std::mem::ManuallyDrop::new(buf);
        let commitments = unsafe {
            Vec::from_raw_parts(
                buf_no_drop.as_mut_ptr() as *mut C,
                commitments_len,
                commitments_len,
            )
        };

        Ok(VerifyingKey { commitments })
    }
}

/// The proving key for a single permutation argument.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ProvingKey<C: CurveAffine> {
    permutations: Vec<Polynomial<C::Scalar, LagrangeCoeff>>,
    polys: Vec<Polynomial<C::Scalar, Coeff>>,
    pub(super) cosets: Vec<Polynomial<C::Scalar, ExtendedLagrangeCoeff>>,
}

impl<C: CurveAffine> ProvingKey<C> {
    /// Writes proving key to a buffer.
    pub fn write<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        let n = u8::try_from(self.permutations.len())
            .expect("Number of permutations must be less than 256");
        writer.write_all(&n.to_le_bytes())?;

        for permutation in &self.permutations {
            permutation.write(writer)?;
        }
        for poly in &self.polys {
            poly.write(writer)?;
        }
        for coset in &self.cosets {
            coset.write(writer)?;
        }

        Ok(())
    }

    /// Reads proving key from a buffer.
    pub fn read<R: io::Read>(reader: &mut R) -> io::Result<Self> {
        let mut buffer = [0u8; 1];
        reader.read_exact(&mut buffer[..])?;
        let n = u8::from_le_bytes(buffer);

        let permutations: Vec<_> = (0..n)
            .map(|_| Polynomial::read(reader))
            .collect::<Result<_, _>>()?;
        let polys: Vec<_> = (0..n)
            .map(|_| Polynomial::read(reader))
            .collect::<Result<_, _>>()?;
        let cosets: Vec<_> = (0..n)
            .map(|_| Polynomial::read(reader))
            .collect::<Result<_, _>>()?;

        Ok(ProvingKey {
            permutations,
            polys,
            cosets,
        })
    }
}
