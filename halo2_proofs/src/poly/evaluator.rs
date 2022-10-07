use std::{
    cmp, fmt,
    hash::{Hash, Hasher},
    marker::PhantomData,
    ops::{Add, Mul, MulAssign, Neg, Sub},
    sync::Arc,
};

use group::ff::Field;
use pasta_curves::arithmetic::FieldExt;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use super::{
    Basis, Coeff, EvaluationDomain, ExtendedLagrangeCoeff, LagrangeCoeff, Polynomial, Rotation,
};
use crate::multicore;

/// Returns `(chunk_size, num_chunks)` suitable for processing the given polynomial length
/// in the current parallelization environment.
fn get_chunk_params(poly_len: usize) -> (usize, usize) {
    // Check the level of parallelization we have available.
    let num_threads = multicore::current_num_threads();
    // We scale the number of chunks by a constant factor, to ensure that if not all
    // threads are available, we can achieve more uniform throughput and don't end up
    // waiting on a couple of threads to process the last chunks.
    let num_chunks = num_threads * 4;
    // Calculate the ideal chunk size for the desired throughput. We use ceiling
    // division to ensure the minimum chunk size is 1.
    //     chunk_size = ceil(poly_len / num_chunks)
    let chunk_size = (poly_len + num_chunks - 1) / num_chunks;
    // Now re-calculate num_chunks from the actual chunk size.
    //     num_chunks = ceil(poly_len / chunk_size)
    let num_chunks = (poly_len + chunk_size - 1) / chunk_size;

    (chunk_size, num_chunks)
}

/// A reference to a polynomial registered with an [`Evaluator`].
#[derive(Clone, Copy, Deserialize, Serialize)]
pub struct AstLeaf<E, B: Basis> {
    pub index: usize,
    pub rotation: Rotation,
    _evaluator: PhantomData<(E, B)>,
}

impl<E, B: Basis> fmt::Debug for AstLeaf<E, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AstLeaf")
            .field("index", &self.index)
            .field("rotation", &self.rotation)
            .finish()
    }
}

impl<E, B: Basis> PartialEq for AstLeaf<E, B> {
    fn eq(&self, rhs: &Self) -> bool {
        // We compare rotations by offset, which doesn't account for equivalent rotations.
        self.index.eq(&rhs.index) && self.rotation.0.eq(&rhs.rotation.0)
    }
}

impl<E, B: Basis> Eq for AstLeaf<E, B> {}

impl<E, B: Basis> Hash for AstLeaf<E, B> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.index.hash(state);
        self.rotation.0.hash(state);
    }
}

impl<E, B: Basis> AstLeaf<E, B> {
    /// Produces a new `AstLeaf` node corresponding to the underlying polynomial at a
    /// _new_ rotation. Existing rotations applied to this leaf node are ignored and the
    /// returned polynomial is not rotated _relative_ to the previous structure.
    pub(crate) fn with_rotation(&self, rotation: Rotation) -> Self {
        AstLeaf {
            index: self.index,
            rotation,
            _evaluator: PhantomData::default(),
        }
    }
}

/// An evaluation context for polynomial operations.
///
/// This context enables us to de-duplicate queries of circuit columns (and the rotations
/// they might require), by storing a list of all the underlying polynomials involved in
/// any query (which are almost certainly column polynomials). We use the context like so:
///
/// - We register each underlying polynomial with the evaluator, which returns a reference
///   to it as a [`AstLeaf`].
/// - The references are then used to build up a [`Ast`] that represents the overall
///   operations to be applied to the polynomials.
/// - Finally, we call [`Evaluator::evaluate`] passing in the [`Ast`].
pub struct Evaluator<E, F: Field, B: Basis> {
    pub polys: Vec<Polynomial<F, B>>,
    pub _context: E,
}

/// Constructs a new `Evaluator`.
///
/// The `context` parameter is used to provide type safety for evaluators. It ensures that
/// an evaluator will only be used to evaluate [`Ast`]s containing [`AstLeaf`]s obtained
/// from itself. It should be set to the empty closure `|| {}`, because anonymous closures
/// all have unique types.
pub fn new_evaluator<E: Fn() + Clone, F: Field, B: Basis>(context: E) -> Evaluator<E, F, B> {
    Evaluator {
        polys: vec![],
        _context: context,
    }
}

fn get_of_rotated_pos<F: Field, B: Basis>(
    pos: usize,
    rotation_is_negative: bool,
    rotation_abs: usize,
    poly: &Polynomial<F, B>,
) -> usize {
    //let rotation_abs = rotation.0.unsigned_abs() as usize;
    //let rotation_abs= ((1 << (self.extended_k - self.k)) * rotation.0.abs()) as usize;

    let (mid, k) = if rotation_is_negative {
        (poly.len() - rotation_abs, rotation_abs)
    } else {
        (rotation_abs, poly.len() - rotation_abs)
    };
    //println!("vmx: get_of_rotated_helper:       mid, k, len, abs: {} {} {} {}", mid, k, poly.len(), rotation_abs);

    if pos < k {
        mid + pos
    } else {
        pos - k
    }
    //if pos + 1 < k {
    //   mid + pos
    //} else if pos >= k {
    //   pos - k
    //} else {
    //   panic!("vmx: oups, i thought this case doesn't happen")
    //}
}
//fn get_of_rotated<F: Field, B: Basis>(pos: usize, rotation_is_negative: bool, rotation_abs: usize, poly: &Polynomial<F, B>) -> F {
//    //let rotation_abs = rotation.0.unsigned_abs() as usize;
//    //let rotation_abs= ((1 << (self.extended_k - self.k)) * rotation.0.abs()) as usize;
//
//    let (mid, k) = if rotation_is_negative {
//        (poly.len() - rotation_abs, rotation_abs)
//    } else {
//        (rotation_abs, poly.len() - rotation_abs)
//    };
//        println!("vmx: get_of_rotated_helper:       mid, k, len, abs: {} {} {} {}", mid, k, poly.len(), rotation_abs);
//
//    if pos + 1 < k {
//       poly[mid + pos]
//    } else if pos >= k {
//       poly[pos - k]
//    } else {
//       panic!("vmx: oups, i thought this case doesn't happen")
//    }
////let chunk_start = pos;
////let chunk_end = pos + 1;
////
////if pos + 1 < k {
////  poly.values[mid + pos..mid + pos + 1].to_vec()
////} else if pos >= k {
////  poly.values[pos - k..pos + 1 - k].to_vec()
////    } else {
////       panic!("vmx: oups, i thought this case doesn't happen")
////    }
//}

impl<E, F: Field, B: Basis> Evaluator<E, F, B> {
    /// Registers the given polynomial for use in this evaluation context.
    ///
    /// This API treats each registered polynomial as unique, even if the same polynomial
    /// is added multiple times.
    pub(crate) fn register_poly(&mut self, poly: Polynomial<F, B>) -> AstLeaf<E, B> {
        let index = self.polys.len();
        self.polys.push(poly);

        AstLeaf {
            index,
            rotation: Rotation::cur(),
            _evaluator: PhantomData::default(),
        }
    }

    /// Evaluates the given polynomial operation against this context.
    pub fn evaluate(&self, ast: &Ast<E, F, B>, domain: &EvaluationDomain<F>) -> Polynomial<F, B>
    where
        E: Copy + Send + Sync,
        F: FieldExt + Serialize + DeserializeOwned,
        B: BasisOps + Serialize + DeserializeOwned,
    {
        log::debug!("vmx: halo2: poly: evalutator: evaluate: ast: {:?}", ast);
        // We're working in a single basis, so all polynomials are the same length.
        let poly_len = self.polys.first().unwrap().len();
        let (chunk_size, _num_chunks) = get_chunk_params(poly_len);
        log::debug!(
            "vmx: halo2: poly: evalutator: evaluate: poly_len, chunk size, num chunks: {} {} {}",
            poly_len,
            chunk_size,
            _num_chunks
        );
        //// NOTE vmx 2022-10-12: Don't chunk it and see what happens.
        //let chunk_size = poly_len;
        //log::debug!(
        //    "vmx: halo2: poly: evalutator: evaluate: num polys, poly_len: {} {}",
        //    self.polys.len(),
        //    poly_len
        //);

        struct AstContext<'a, F: FieldExt, B: Basis> {
            domain: &'a EvaluationDomain<F>,
            poly_len: usize,
            chunk_size: usize,
            chunk_index: usize,
            polys: &'a [Polynomial<F, B>],
        }

        fn recurse<E, F: FieldExt, B: BasisOps>(
            ast: &Ast<E, F, B>,
            ctx: &AstContext<'_, F, B>,
        ) -> Vec<F> {
            match ast {
                Ast::Poly(leaf) => {
                    let orig_result = B::get_chunk_of_rotated(
                        ctx.domain,
                        ctx.chunk_size,
                        ctx.chunk_index,
                        &ctx.polys[leaf.index],
                        leaf.rotation,
                    );
                    //println!("vmx: poly: orig: {:?}", orig_result[0]);
                    //let rotation_abs = ((1 << (ctx.domain.extended_k - ctx.domain.k)) * leaf.rotation.0.abs()) as usize;
                    //println!("vmx: poly: mine: roation_abs: {:?}", rotation_abs);
                    //let myrotatedpos = get_of_rotated_pos(1, leaf.rotation.0 < 0, rotation_abs, &ctx.polys[leaf.index]);
                    //////println!("vmx: rotated, my, direct: {:?} {:?} {:?}", rotated, ctx.polys[leaf.index][myrotatedpos], ctx.polys[leaf.index][0]);
                    //println!("vmx: poly: mine: pos: {}", myrotatedpos);
                    //println!("vmx: poly: mine: {:?}", ctx.polys[leaf.index][myrotatedpos]);
                    //////vec![ctx.polys[leaf.index][0]]
                    ////let result = vec![ctx.polys[leaf.index][myrotatedpos]];
                    ////println!("vmx: poly: {:?}", result);
                    ////result
                    orig_result
                }
                Ast::Add(a, b) => {
                    let mut lhs = recurse(a, ctx);
                    let rhs = recurse(b, ctx);
                    for (lhs, rhs) in lhs.iter_mut().zip(rhs.iter()) {
                        *lhs += *rhs;
                    }
                    //lhs[0] += rhs[0];
                    //println!("vmx: add: {:?}", lhs[0]);
                    lhs
                }
                Ast::Mul(AstMul(a, b)) => {
                    let mut lhs = recurse(a, ctx);
                    let rhs = recurse(b, ctx);
                    for (lhs, rhs) in lhs.iter_mut().zip(rhs.iter()) {
                        *lhs *= *rhs;
                    }
                    //println!("vmx: about to multiply {:?} with: {:?}", lhs[0], rhs[0]);
                    //lhs[0] *= rhs[0];
                    //println!("vmx: mul: {:?}", lhs[0]);
                    lhs
                }
                Ast::Scale(a, scalar) => {
                    let mut lhs = recurse(a, ctx);
                    for lhs in lhs.iter_mut() {
                        *lhs *= scalar;
                    }
                    //lhs[0] *= scalar;
                    //println!("vmx: scale: {:?}", lhs[0]);
                    lhs
                }
                Ast::DistributePowers(terms, base) => terms.iter().fold(
                    B::constant_term(ctx.poly_len, ctx.chunk_size, ctx.chunk_index, F::zero()),
                    |mut acc, term| {
                        let term = recurse(term, ctx);
                        for (acc, term) in acc.iter_mut().zip(term) {
                            //println!("vmx: debugging: acc: {:?}", acc);
                            *acc *= base;
                            //println!("vmx: debugging: acc: {:?}", acc);
                            *acc += term;
                            //println!("vmx: debugging: acc: {:?}", acc);
                        }
                        acc
                    },
                ),
                Ast::LinearTerm(scalar) => {
                    let term = B::linear_term(
                        ctx.domain,
                        ctx.poly_len,
                        ctx.chunk_size,
                        ctx.chunk_index,
                        *scalar,
                    );
                    //let omega = ctx.domain.get_extended_omega();
                    //let mut result = omega.pow_vartime(&[0 as u64]) * F::ZETA * scalar;
                    ////result *= omega;
                    ////let zeta_scalar_omega = F::ZETA * scalar * omega;
                    ////
                    ////source.push(format!("let omega{} = {};", ctx.counter, to_fp_from_raw(&omega)));
                    ////source.push(format!("let mut elem{} = omega{}.pow_vartime(&[pos as u64]) * {};", ctx.counter, ctx.counter, to_fp_from_raw(&zeta_scalar_omega)));
                    //         println!("vmx: linear: my result of first element: {:?}", result);

                    //println!("vmx: linear: {:?}", term[0]);
                    term
                    //vec![*scalar]
                }
                Ast::ConstantTerm(scalar) => {
                    let term =
                        B::constant_term(ctx.poly_len, ctx.chunk_size, ctx.chunk_index, *scalar);
                    //println!("vmx: const: {:?}", term[0]);
                    term
                    //vec![*scalar]
                }
            }
        }

        // Dump the AST
        if matches!(ast, Ast::DistributePowers(_, _)) {
            use std::io::Write;
            let k = domain.k;
            let j = domain.get_quotient_poly_degree() + 1;
            let field =
                std::any::type_name::<F>().replace(|c: char| !c.is_ascii_alphanumeric(), "_");
            let basis =
                std::any::type_name::<B>().replace(|c: char| !c.is_ascii_alphanumeric(), "_");
            let time = chrono::Local::now().format("%Y%m%d%H%M%S").to_string();

            let mut ast_file = std::fs::File::create(format!(
                "/tmp/halo2-{}-{}-{}-{}_{}.ast",
                j, k, field, basis, time
            ))
            .unwrap();
            let ast_bytes = bincode::serialize(&ast).expect("AST cannot be serialized");
            ast_file.write_all(&ast_bytes).expect("write failed");

            let mut polys_file = std::fs::File::create(format!(
                "/tmp/halo2-{}-{}-{}-{}_{}.polys",
                j, k, field, basis, time
            ))
            .unwrap();
            // Prefix the poly file with the number of polynomials and the number of elements of
            // the polynomials (they all have the same length).
            let num_polys =
                u32::try_from(self.polys.len()).expect("There are less then 2^32 polynomials");
            polys_file.write_all(&num_polys.to_le_bytes()).unwrap();
            let poly_len = u32::try_from(self.polys[0].len())
                .expect("There are less then 2^32 elements in a polynomial");
            polys_file.write_all(&poly_len.to_le_bytes()).unwrap();
            for poly in &self.polys {
                polys_file.write_all(poly.as_bytes()).unwrap();
            }
        }

        log::trace!("vmx: halo2: poly: evalutator: evaluate: apply ast: start");
        // Apply `ast` to each chunk in parallel, writing the result into an output
        // done.
        let mut result = B::empty_poly(domain);
        multicore::scope(|scope| {
            for (chunk_index, out) in result.chunks_mut(chunk_size).enumerate() {
                scope.spawn(move |_| {
                    let ctx = AstContext {
                        domain,
                        poly_len,
                        chunk_size,
                        chunk_index,
                        polys: &self.polys,
                    };
                    out.copy_from_slice(&recurse(ast, &ctx));
                });
            }
        });
        ////let mut result = Polynomial {
        ////    values: vec![F::group_zero(); 1],
        ////    _marker: PhantomData,
        ////};
        //let ctx = AstContext {
        //    domain,
        //    //poly_len: 1
        //    //chunk_size: 1,
        //    //chunk_index: 0,
        //    poly_len,
        //    chunk_size: poly_len,
        //    chunk_index: 0,
        //    polys: &self.polys,
        //};
        //result.copy_from_slice(&recurse(ast, &ctx));
        log::trace!("vmx: halo2: poly: evalutator: evaluate: apply ast: done");
        result
    }
}

/// Struct representing the [`Ast::Mul`] case.
///
/// This struct exists to make the internals of this case private so that we don't
/// accidentally construct this case directly, because it can only be implemented for the
/// [`ExtendedLagrangeCoeff`] basis.
#[derive(Clone, Deserialize, Serialize)]
#[serde(bound(
    deserialize = "F: Deserialize<'de>, B: Deserialize<'de>",
    serialize = "F: Serialize, B: Serialize",
))]
pub struct AstMul<E, F: Field, B: Basis>(pub Arc<Ast<E, F, B>>, pub Arc<Ast<E, F, B>>);

impl<E, F: Field, B: Basis> fmt::Debug for AstMul<E, F, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("AstMul")
            .field(&self.0)
            .field(&self.1)
            .finish()
    }
}

/// A polynomial operation backed by an [`Evaluator`].
#[derive(Clone, Deserialize, Serialize)]
#[serde(bound(
    deserialize = "F: Deserialize<'de>, B: Deserialize<'de>",
    serialize = "F: Serialize, B: Serialize",
))]
pub enum Ast<E, F: Field, B: Basis> {
    Poly(AstLeaf<E, B>),
    Add(Arc<Ast<E, F, B>>, Arc<Ast<E, F, B>>),
    Mul(AstMul<E, F, B>),
    Scale(Arc<Ast<E, F, B>>, F),
    /// Represents a linear combination of a vector of nodes and the powers of a
    /// field element, where the nodes are ordered from highest to lowest degree
    /// terms.
    DistributePowers(Arc<Vec<Ast<E, F, B>>>, F),
    /// The degree-1 term of a polynomial.
    ///
    /// The field element is the coefficient of the term in the standard basis, not the
    /// coefficient basis.
    LinearTerm(F),
    /// The degree-0 term of a polynomial.
    ///
    /// The field element is the same in both the standard and evaluation bases.
    ConstantTerm(F),
}

impl<E, F: Field + Serialize, B: Basis + Serialize> Ast<E, F, B> {
    pub fn distribute_powers<I: IntoIterator<Item = Self>>(i: I, base: F) -> Self {
        //Ast::DistributePowers(Arc::new(i.into_iter().collect()), base)
        let terms = i.into_iter().collect::<Vec<_>>();
        log::debug!(
            "vmx: halo2: poly: evalutator: distribute powers: num terms: {}",
            terms.len()
        );
        Ast::DistributePowers(Arc::new(terms), base)
    }
}

impl<E, F: Field, B: Basis> fmt::Debug for Ast<E, F, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Poly(leaf) => f.debug_tuple("Poly").field(leaf).finish(),
            Self::Add(lhs, rhs) => f.debug_tuple("Add").field(lhs).field(rhs).finish(),
            Self::Mul(x) => f.debug_tuple("Mul").field(x).finish(),
            Self::Scale(base, scalar) => f.debug_tuple("Scale").field(base).field(scalar).finish(),
            Self::DistributePowers(terms, base) => f
                .debug_tuple("DistributePowers")
                .field(terms)
                .field(base)
                .finish(),
            Self::LinearTerm(x) => f.debug_tuple("LinearTerm").field(x).finish(),
            Self::ConstantTerm(x) => f.debug_tuple("ConstantTerm").field(x).finish(),
        }
    }
}

impl<E, F: Field, B: Basis> From<AstLeaf<E, B>> for Ast<E, F, B> {
    fn from(leaf: AstLeaf<E, B>) -> Self {
        Ast::Poly(leaf)
    }
}

impl<E, F: Field, B: Basis> Ast<E, F, B> {
    pub(crate) fn one() -> Self {
        Self::ConstantTerm(F::one())
    }
}

impl<E, F: Field, B: Basis> Neg for Ast<E, F, B> {
    type Output = Ast<E, F, B>;

    fn neg(self) -> Self::Output {
        Ast::Scale(Arc::new(self), -F::one())
    }
}

impl<E: Clone, F: Field, B: Basis> Neg for &Ast<E, F, B> {
    type Output = Ast<E, F, B>;

    fn neg(self) -> Self::Output {
        -(self.clone())
    }
}

impl<E, F: Field, B: Basis> Add for Ast<E, F, B> {
    type Output = Ast<E, F, B>;

    fn add(self, other: Self) -> Self::Output {
        Ast::Add(Arc::new(self), Arc::new(other))
    }
}

impl<'a, E: Clone, F: Field, B: Basis> Add<&'a Ast<E, F, B>> for &'a Ast<E, F, B> {
    type Output = Ast<E, F, B>;

    fn add(self, other: &'a Ast<E, F, B>) -> Self::Output {
        self.clone() + other.clone()
    }
}

impl<E, F: Field, B: Basis> Add<AstLeaf<E, B>> for Ast<E, F, B> {
    type Output = Ast<E, F, B>;

    fn add(self, other: AstLeaf<E, B>) -> Self::Output {
        Ast::Add(Arc::new(self), Arc::new(other.into()))
    }
}

impl<E, F: Field, B: Basis> Sub for Ast<E, F, B> {
    type Output = Ast<E, F, B>;

    fn sub(self, other: Self) -> Self::Output {
        self + (-other)
    }
}

impl<'a, E: Clone, F: Field, B: Basis> Sub<&'a Ast<E, F, B>> for &'a Ast<E, F, B> {
    type Output = Ast<E, F, B>;

    fn sub(self, other: &'a Ast<E, F, B>) -> Self::Output {
        self + &(-other)
    }
}

impl<E, F: Field, B: Basis> Sub<AstLeaf<E, B>> for Ast<E, F, B> {
    type Output = Ast<E, F, B>;

    fn sub(self, other: AstLeaf<E, B>) -> Self::Output {
        self + (-Ast::from(other))
    }
}

impl<E, F: Field> Mul for Ast<E, F, LagrangeCoeff> {
    type Output = Ast<E, F, LagrangeCoeff>;

    fn mul(self, other: Self) -> Self::Output {
        Ast::Mul(AstMul(Arc::new(self), Arc::new(other)))
    }
}

impl<'a, E: Clone, F: Field> Mul<&'a Ast<E, F, LagrangeCoeff>> for &'a Ast<E, F, LagrangeCoeff> {
    type Output = Ast<E, F, LagrangeCoeff>;

    fn mul(self, other: &'a Ast<E, F, LagrangeCoeff>) -> Self::Output {
        self.clone() * other.clone()
    }
}

impl<E, F: Field> Mul<AstLeaf<E, LagrangeCoeff>> for Ast<E, F, LagrangeCoeff> {
    type Output = Ast<E, F, LagrangeCoeff>;

    fn mul(self, other: AstLeaf<E, LagrangeCoeff>) -> Self::Output {
        Ast::Mul(AstMul(Arc::new(self), Arc::new(other.into())))
    }
}

impl<E, F: Field> Mul for Ast<E, F, ExtendedLagrangeCoeff> {
    type Output = Ast<E, F, ExtendedLagrangeCoeff>;

    fn mul(self, other: Self) -> Self::Output {
        Ast::Mul(AstMul(Arc::new(self), Arc::new(other)))
    }
}

impl<'a, E: Clone, F: Field> Mul<&'a Ast<E, F, ExtendedLagrangeCoeff>>
    for &'a Ast<E, F, ExtendedLagrangeCoeff>
{
    type Output = Ast<E, F, ExtendedLagrangeCoeff>;

    fn mul(self, other: &'a Ast<E, F, ExtendedLagrangeCoeff>) -> Self::Output {
        self.clone() * other.clone()
    }
}

impl<E, F: Field> Mul<AstLeaf<E, ExtendedLagrangeCoeff>> for Ast<E, F, ExtendedLagrangeCoeff> {
    type Output = Ast<E, F, ExtendedLagrangeCoeff>;

    fn mul(self, other: AstLeaf<E, ExtendedLagrangeCoeff>) -> Self::Output {
        Ast::Mul(AstMul(Arc::new(self), Arc::new(other.into())))
    }
}

impl<E, F: Field, B: Basis> Mul<F> for Ast<E, F, B> {
    type Output = Ast<E, F, B>;

    fn mul(self, other: F) -> Self::Output {
        Ast::Scale(Arc::new(self), other)
    }
}

impl<E: Clone, F: Field, B: Basis> Mul<F> for &Ast<E, F, B> {
    type Output = Ast<E, F, B>;

    fn mul(self, other: F) -> Self::Output {
        Ast::Scale(Arc::new(self.clone()), other)
    }
}

impl<E: Clone, F: Field> MulAssign for Ast<E, F, ExtendedLagrangeCoeff> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone().mul(rhs)
    }
}

/// Operations which can be performed over a given basis.
pub trait BasisOps: Basis {
    fn empty_poly<F: FieldExt>(domain: &EvaluationDomain<F>) -> Polynomial<F, Self>;
    fn constant_term<F: FieldExt>(
        poly_len: usize,
        chunk_size: usize,
        chunk_index: usize,
        scalar: F,
    ) -> Vec<F>;
    fn linear_term<F: FieldExt>(
        domain: &EvaluationDomain<F>,
        poly_len: usize,
        chunk_size: usize,
        chunk_index: usize,
        scalar: F,
    ) -> Vec<F>;
    fn get_chunk_of_rotated<F: FieldExt>(
        domain: &EvaluationDomain<F>,
        chunk_size: usize,
        chunk_index: usize,
        poly: &Polynomial<F, Self>,
        rotation: Rotation,
    ) -> Vec<F>;
}

impl BasisOps for Coeff {
    fn empty_poly<F: FieldExt>(domain: &EvaluationDomain<F>) -> Polynomial<F, Self> {
        domain.empty_coeff()
    }

    fn constant_term<F: FieldExt>(
        poly_len: usize,
        chunk_size: usize,
        chunk_index: usize,
        scalar: F,
    ) -> Vec<F> {
        let mut chunk = vec![F::zero(); cmp::min(chunk_size, poly_len - chunk_size * chunk_index)];
        if chunk_index == 0 {
            chunk[0] = scalar;
        }
        chunk
    }

    fn linear_term<F: FieldExt>(
        _: &EvaluationDomain<F>,
        poly_len: usize,
        chunk_size: usize,
        chunk_index: usize,
        scalar: F,
    ) -> Vec<F> {
        let mut chunk = vec![F::zero(); cmp::min(chunk_size, poly_len - chunk_size * chunk_index)];
        // If the chunk size is 1 (e.g. if we have a small k and many threads), then the
        // linear coefficient is the second chunk. Otherwise, the chunk size is greater
        // than one, and the linear coefficient is the second element of the first chunk.
        // Note that we check against the original chunk size, not the potentially-short
        // actual size of the current chunk, because we want to know whether the size of
        // the previous chunk was 1.
        if chunk_size == 1 && chunk_index == 1 {
            chunk[0] = scalar;
        } else if chunk_index == 0 {
            chunk[1] = scalar;
        }
        chunk
    }

    fn get_chunk_of_rotated<F: FieldExt>(
        _: &EvaluationDomain<F>,
        _: usize,
        _: usize,
        _: &Polynomial<F, Self>,
        _: Rotation,
    ) -> Vec<F> {
        panic!("Can't rotate polynomials in the standard basis")
    }
}

impl BasisOps for LagrangeCoeff {
    fn empty_poly<F: FieldExt>(domain: &EvaluationDomain<F>) -> Polynomial<F, Self> {
        domain.empty_lagrange()
    }

    fn constant_term<F: FieldExt>(
        poly_len: usize,
        chunk_size: usize,
        chunk_index: usize,
        scalar: F,
    ) -> Vec<F> {
        vec![scalar; cmp::min(chunk_size, poly_len - chunk_size * chunk_index)]
    }

    fn linear_term<F: FieldExt>(
        domain: &EvaluationDomain<F>,
        poly_len: usize,
        chunk_size: usize,
        chunk_index: usize,
        scalar: F,
    ) -> Vec<F> {
        // Take every power of omega within the chunk, and multiply by scalar.
        let omega = domain.get_omega();
        let start = chunk_size * chunk_index;
        (0..cmp::min(chunk_size, poly_len - start))
            .scan(omega.pow_vartime(&[start as u64]) * scalar, |acc, _| {
                let ret = *acc;
                *acc *= omega;
                Some(ret)
            })
            .collect()
    }

    fn get_chunk_of_rotated<F: FieldExt>(
        _: &EvaluationDomain<F>,
        chunk_size: usize,
        chunk_index: usize,
        poly: &Polynomial<F, Self>,
        rotation: Rotation,
    ) -> Vec<F> {
        poly.get_chunk_of_rotated(rotation, chunk_size, chunk_index)
    }
}

impl BasisOps for ExtendedLagrangeCoeff {
    fn empty_poly<F: FieldExt>(domain: &EvaluationDomain<F>) -> Polynomial<F, Self> {
        domain.empty_extended()
    }

    fn constant_term<F: FieldExt>(
        poly_len: usize,
        chunk_size: usize,
        chunk_index: usize,
        scalar: F,
    ) -> Vec<F> {
        vec![scalar; cmp::min(chunk_size, poly_len - chunk_size * chunk_index)]
    }

    fn linear_term<F: FieldExt>(
        domain: &EvaluationDomain<F>,
        poly_len: usize,
        chunk_size: usize,
        chunk_index: usize,
        scalar: F,
    ) -> Vec<F> {
        // Take every power of the extended omega within the chunk, and multiply by scalar.
        let omega = domain.get_extended_omega();
        let start = chunk_size * chunk_index;
        (0..cmp::min(chunk_size, poly_len - start))
            .scan(
                omega.pow_vartime(&[start as u64]) * F::ZETA * scalar,
                |acc, _| {
                    let ret = *acc;
                    *acc *= omega;
                    Some(ret)
                },
            )
            .collect()
    }

    fn get_chunk_of_rotated<F: FieldExt>(
        domain: &EvaluationDomain<F>,
        chunk_size: usize,
        chunk_index: usize,
        poly: &Polynomial<F, Self>,
        rotation: Rotation,
    ) -> Vec<F> {
        domain.get_chunk_of_rotated_extended(poly, rotation, chunk_size, chunk_index)
    }
}

#[cfg(test)]
mod tests {
    use pasta_curves::pallas;

    use super::{get_chunk_params, new_evaluator, Ast, BasisOps, Evaluator};
    use crate::poly::{Coeff, EvaluationDomain, ExtendedLagrangeCoeff, LagrangeCoeff};

    #[test]
    fn short_chunk_regression_test() {
        // Pick the smallest polynomial length that is guaranteed to produce a short chunk
        // on this machine.
        let k = match (1..16)
            .map(|k| (k, get_chunk_params(1 << k)))
            .find(|(k, (chunk_size, num_chunks))| (1 << k) < chunk_size * num_chunks)
            .map(|(k, _)| k)
        {
            Some(k) => k,
            None => {
                // We are on a machine with a power-of-two number of threads, and cannot
                // trigger the bug.
                eprintln!(
                    "can't find a polynomial length for short_chunk_regression_test; skipping"
                );
                return;
            }
        };
        eprintln!("Testing short-chunk regression with k = {}", k);

        fn test_case<E: Copy + Send + Sync, B: BasisOps>(
            k: u32,
            mut evaluator: Evaluator<E, pallas::Base, B>,
        ) {
            // Instantiate the evaluator with a trivial polynomial.
            let domain = EvaluationDomain::new(1, k);
            evaluator.register_poly(B::empty_poly(&domain));

            // With the bug present, these will panic.
            let _ = evaluator.evaluate(&Ast::ConstantTerm(pallas::Base::zero()), &domain);
            let _ = evaluator.evaluate(&Ast::LinearTerm(pallas::Base::zero()), &domain);
        }

        test_case(k, new_evaluator::<_, _, Coeff>(|| {}));
        test_case(k, new_evaluator::<_, _, LagrangeCoeff>(|| {}));
        test_case(k, new_evaluator::<_, _, ExtendedLagrangeCoeff>(|| {}));
    }
}
