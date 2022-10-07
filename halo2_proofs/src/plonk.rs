//! This module provides an implementation of a variant of (Turbo)[PLONK][plonk]
//! that is designed specifically for the polynomial commitment scheme described
//! in the [Halo][halo] paper.
//!
//! [halo]: https://eprint.iacr.org/2019/1021
//! [plonk]: https://eprint.iacr.org/2019/953

use blake2b_simd::Params as Blake2bParams;
use group::ff::Field;

use crate::arithmetic::{CurveAffine, FieldExt};
use crate::poly::{
    commitment::Params, Coeff, EvaluationDomain, ExtendedLagrangeCoeff, LagrangeCoeff,
    PinnedEvaluationDomain, Polynomial,
};
use crate::transcript::{ChallengeScalar, EncodedChallenge, Transcript};

mod assigned;
mod circuit;
mod error;
mod keygen;
mod lookup;
pub(crate) mod permutation;
mod vanishing;

mod prover;
mod verifier;

pub use assigned::*;
pub use circuit::*;
pub use error::*;
pub use keygen::*;
pub use prover::*;
pub use verifier::*;

use std::io;

pub(crate) const FIELD_SIZE: usize = 32;
pub(crate) const AFFINE_SIZE: usize = 2 * FIELD_SIZE;

/// This is a verifying key which allows for the verification of proofs for a
/// particular circuit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifyingKey<C: CurveAffine> {
    domain: EvaluationDomain<C::Scalar>,
    fixed_commitments: Vec<C>,
    permutation: permutation::VerifyingKey<C>,
    cs: ConstraintSystem<C::Scalar>,
    /// Cached maximum degree of `cs` (which doesn't change after construction).
    cs_degree: usize,
    /// The representative of this `VerifyingKey` in transcripts.
    transcript_repr: C::Scalar,
}

impl<C: CurveAffine> VerifyingKey<C> {
    fn from_parts(
        domain: EvaluationDomain<C::Scalar>,
        fixed_commitments: Vec<C>,
        permutation: permutation::VerifyingKey<C>,
        cs: ConstraintSystem<C::Scalar>,
    ) -> Self {
        // Compute cached values.
        let cs_degree = cs.degree();

        let mut vk = Self {
            domain,
            fixed_commitments,
            permutation,
            cs,
            cs_degree,
            // Temporary, this is not pinned.
            transcript_repr: C::Scalar::zero(),
        };

        let mut hasher = Blake2bParams::new()
            .hash_length(64)
            .personal(b"Halo2-Verify-Key")
            .to_state();

        let s = format!("{:?}", vk.pinned());

        hasher.update(&(s.len() as u64).to_le_bytes());
        hasher.update(s.as_bytes());

        // Hash in final Blake2bState
        vk.transcript_repr = C::Scalar::from_bytes_wide(hasher.finalize().as_array());

        vk
    }

    /// Hashes a verification key into a transcript.
    pub fn hash_into<E: EncodedChallenge<C>, T: Transcript<C, E>>(
        &self,
        transcript: &mut T,
    ) -> io::Result<()> {
        transcript.common_scalar(self.transcript_repr)?;

        Ok(())
    }

    /// Obtains a pinned representation of this verification key that contains
    /// the minimal information necessary to reconstruct the verification key.
    pub fn pinned(&self) -> PinnedVerificationKey<'_, C> {
        PinnedVerificationKey {
            base_modulus: C::Base::MODULUS,
            scalar_modulus: C::Scalar::MODULUS,
            domain: self.domain.pinned(),
            fixed_commitments: &self.fixed_commitments,
            permutation: &self.permutation,
            cs: self.cs.pinned(),
        }
    }

    /// Writes evaluation domain, fixed commitments, and permutation verification key to a buffer.
    #[allow(unsafe_code)]
    pub fn write<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        assert_eq!(std::mem::size_of::<C::Scalar>(), FIELD_SIZE);
        assert_eq!(std::mem::size_of::<C>(), AFFINE_SIZE);

        self.domain.write(writer)?;
        {
            let byte_len = self.fixed_commitments.len() * AFFINE_SIZE;
            let bytes_ptr = self.fixed_commitments.as_ptr() as *const u8;
            let bytes = unsafe { std::slice::from_raw_parts(bytes_ptr, byte_len) };
            writer.write_all(&bytes)?;
        }
        self.permutation.write(writer)?;
        writer.write_all(&self.cs_degree.to_le_bytes())?;
        {
            let bytes_ptr = (&self.transcript_repr as *const C::Scalar) as *const [u8; FIELD_SIZE];
            unsafe { writer.write_all(&*bytes_ptr)? };
        }

        Ok(())
    }

    /// Reads evaluation domain, fixed commitments, and permutation verification key from a buffer.
    #[allow(unsafe_code)]
    pub fn read<R: io::Read, ConcreteCircuit: Circuit<C::Scalar>>(
        reader: &mut R,
        params: &Params<C>,
        circuit: &ConcreteCircuit,
    ) -> io::Result<Self> {
        assert_eq!(std::mem::size_of::<C::Scalar>(), FIELD_SIZE);
        assert_eq!(std::mem::size_of::<C>(), AFFINE_SIZE);

        let mut cs = ConstraintSystem::<C::Scalar>::default();
        let config = ConcreteCircuit::configure(&mut cs);
        let empty_lagrange = Polynomial::new(params.n as usize);
        let mut assembly = Assembly::new(&params, &cs, empty_lagrange);
        ConcreteCircuit::FloorPlanner::synthesize(
            &mut assembly,
            circuit,
            config,
            cs.constants.clone(),
        )
        .expect("failed to synthesize circuit");
        let (cs, _selector_polys) = cs.compress_selectors(assembly.selectors);

        let domain = EvaluationDomain::read(reader, params, &cs)?;
        let fixed_commitments = {
            let len = cs.num_fixed_columns;
            let byte_len = len * AFFINE_SIZE;
            let mut buf = vec![0u8; byte_len];
            reader.read_exact(&mut buf)?;
            let mut buf_no_drop = std::mem::ManuallyDrop::new(buf);
            unsafe { Vec::from_raw_parts(buf_no_drop.as_mut_ptr() as *mut C, len, len) }
        };
        let permutation = permutation::VerifyingKey::read(reader, cs.permutation.len())?;
        let cs_degree = {
            let mut buf = [0u8; 8];
            reader.read_exact(&mut buf)?;
            usize::from_le_bytes(buf)
        };
        let transcript_repr = {
            let mut buf = [0u8; FIELD_SIZE];
            reader.read_exact(&mut buf)?;
            unsafe { *(buf.as_ptr() as *const C::Scalar) }
        };
        let vk = Self::from_parts(domain, fixed_commitments, permutation, cs);

        assert_eq!(cs_degree, vk.cs_degree);
        assert_eq!(transcript_repr, vk.transcript_repr);
        Ok(vk)
    }

    #[allow(missing_docs)]
    pub fn transcript_repr(&self) -> &C::Scalar {
        &self.transcript_repr
    }
}

/// Minimal representation of a verification key that can be used to identify
/// its active contents.
#[allow(dead_code)]
#[derive(Debug)]
pub struct PinnedVerificationKey<'a, C: CurveAffine> {
    base_modulus: &'static str,
    scalar_modulus: &'static str,
    domain: PinnedEvaluationDomain<'a, C::Scalar>,
    cs: PinnedConstraintSystem<'a, C::Scalar>,
    fixed_commitments: &'a Vec<C>,
    permutation: &'a permutation::VerifyingKey<C>,
}
/// This is a proving key which allows for the creation of proofs for a
/// particular circuit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProvingKey<C: CurveAffine> {
    vk: VerifyingKey<C>,
    l0: Polynomial<C::Scalar, ExtendedLagrangeCoeff>,
    l_blind: Polynomial<C::Scalar, ExtendedLagrangeCoeff>,
    l_last: Polynomial<C::Scalar, ExtendedLagrangeCoeff>,
    fixed_values: Vec<Polynomial<C::Scalar, LagrangeCoeff>>,
    fixed_polys: Vec<Polynomial<C::Scalar, Coeff>>,
    fixed_cosets: Vec<Polynomial<C::Scalar, ExtendedLagrangeCoeff>>,
    permutation: permutation::ProvingKey<C>,
}

impl<C: CurveAffine> ProvingKey<C> {
    /// Get the underlying [`VerifyingKey`].
    pub fn get_vk(&self) -> &VerifyingKey<C> {
        &self.vk
    }

    /// Writes proving key to a buffer.
    ///
    /// The verifying key is not part of this serialization, it's usually serialized separately.
    pub fn write<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        self.l0.write(writer)?;
        self.l_blind.write(writer)?;
        self.l_last.write(writer)?;

        let n =
            u8::try_from(self.fixed_values.len()).expect("Number of fixed values is less than 256");
        writer.write_all(&n.to_le_bytes())?;
        for fixed_value in &self.fixed_values {
            fixed_value.write(writer)?;
        }
        for fixed_poly in &self.fixed_polys {
            fixed_poly.write(writer)?;
        }
        for fixed_coset in &self.fixed_cosets {
            fixed_coset.write(writer)?;
        }

        self.permutation.write(writer)?;

        Ok(())
    }

    /// Reads proving key from a buffer.
    ///
    /// The verifying key is passed it as it is usually serialized separately.
    pub fn read<R: io::Read>(reader: &mut R, vk: VerifyingKey<C>) -> io::Result<Self> {
        let l0 = Polynomial::read(reader)?;
        let l_blind = Polynomial::read(reader)?;
        let l_last = Polynomial::read(reader)?;

        let mut buffer = [0u8; 1];
        reader.read_exact(&mut buffer[..])?;
        let n = u8::from_le_bytes(buffer);
        let fixed_values: Vec<_> = (0..n)
            .map(|_| Polynomial::read(reader))
            .collect::<Result<_, _>>()?;
        let fixed_polys: Vec<_> = (0..n)
            .map(|_| Polynomial::read(reader))
            .collect::<Result<_, _>>()?;
        let fixed_cosets: Vec<_> = (0..n)
            .map(|_| Polynomial::read(reader))
            .collect::<Result<_, _>>()?;

        let permutation = permutation::ProvingKey::read(reader)?;

        Ok(ProvingKey {
            vk,
            l0,
            l_blind,
            l_last,
            fixed_values,
            fixed_polys,
            fixed_cosets,
            permutation,
        })
    }
}

impl<C: CurveAffine> VerifyingKey<C> {
    /// Get the underlying [`EvaluationDomain`].
    pub fn get_domain(&self) -> &EvaluationDomain<C::Scalar> {
        &self.domain
    }
}

#[derive(Clone, Copy, Debug)]
struct Theta;
type ChallengeTheta<F> = ChallengeScalar<F, Theta>;

#[derive(Clone, Copy, Debug)]
struct Beta;
type ChallengeBeta<F> = ChallengeScalar<F, Beta>;

#[derive(Clone, Copy, Debug)]
struct Gamma;
type ChallengeGamma<F> = ChallengeScalar<F, Gamma>;

#[derive(Clone, Copy, Debug)]
struct Y;
type ChallengeY<F> = ChallengeScalar<F, Y>;

#[derive(Clone, Copy, Debug)]
struct X;
type ChallengeX<F> = ChallengeScalar<F, X>;
