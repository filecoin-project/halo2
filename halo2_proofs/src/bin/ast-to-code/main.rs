use std::{
    cmp, env,
    fs::{self, File},
    io::{Read, Write},
    mem,
    path::Path,
    process, slice,
};

use halo2_proofs::pasta::group::ff::PrimeField;
use halo2_proofs::{
    arithmetic::FieldExt,
    pasta::Fp,
    poly::{
        Ast, AstMul, Basis, BasisOps, EvaluationDomain, Evaluator, ExtendedLagrangeCoeff,
        Polynomial,
    },
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::Serialize;
use sha2::{Digest, Sha256};

mod evaluate;

// Based on poly/evaluator.rs.
struct AstContext<'a, F: FieldExt, B: Basis> {
    domain: &'a EvaluationDomain<F>,
    poly_len: usize,
    polys: &'a [Polynomial<F, B>],
    /// The `counter` is increased every time this function is called. It is used to create unique
    /// identifier.
    counter: usize,
}

impl<'a, F: FieldExt, B: Basis> AstContext<'a, F, B> {
    /// Increases the counter and returns a new context.
    fn increase_counter(&self) -> Self {
        Self {
            domain: &self.domain,
            poly_len: self.poly_len,
            polys: &self.polys,
            counter: self.counter + 1,
        }
    }

    /// Sets the counter to a specific value and returns a new context.
    fn set_counter(&self, counter: usize) -> Self {
        Self {
            domain: &self.domain,
            poly_len: self.poly_len,
            polys: &self.polys,
            counter,
        }
    }
}

struct AstReturn {
    // Always increasing counter.
    counter: usize,
    // The counter for the current result.
    result_counter: usize,
    source: Vec<String>,
}

//fn get_of_rotated_pos(
//    pos: usize,
//    rotation_is_negative: bool,
//    rotation_abs: usize,
//    poly_len: usize,
//) -> usize {
//    let (mid, k) = if rotation_is_negative {
//        (poly_len - rotation_abs, rotation_abs)
//    } else {
//        (rotation_abs, poly_len - rotation_abs)
//    };
//
//    if pos < k {
//        mid + pos
//    } else {
//        pos - k
//    }
//}

//fn vmx_get_of_rotated_pos(
//    pos: i32,
//    rotation_is_negative: bool,
//    rotation_abs: usize,
//    //rotation: i32,
//    poly_len: usize,
//) -> usize {
//    //if rotation_is_negative {
//    //    let (mid, k) =  (poly_len - rotation_abs, rotation_abs);
//    //    if pos < k {
//    //        mid + pos
//    //    } else {
//    //        pos - k
//    //    }
//    //} else {
//    //    let (mid, k) = (rotation_abs, poly_len - rotation_abs);
//    //    if pos < k {
//    //        mid + pos
//    //    } else {
//    //        pos - k
//    //    }
//    //}
//    //if rotation_is_negative {
//    //    if pos < rotation_abs {
//    //        pos + (poly_len - rotation_abs)
//    //    } else {
//    //        pos - rotation_abs
//    //    }
//    //} else {
//    //    if pos < (poly_len - rotation_abs) {
//    //        pos + rotation_abs
//    //    } else {
//    //        pos - (poly_len - rotation_abs)
//    //    }
//    //}
//
//    //// This one is not complete, it would need more work.
//    //let new_pos = pos + rotation;
//    //if new_pos > 0 {
//    //    if new_pos < poly_len {
//    //        new_pos as usize
//    //    } else {
//    //        (new_pos - poly_len) as usize
//    //    }
//    //} else {
//    //    (new_pos + poly_len) as usize
//    //}

//fn vmx_get_of_rotated_pos(
//   pos: i32,
//   rotation: i32,
//   poly_len: i32,
//) -> usize {
//   // The position is at the beginning, the rotation is negative and so large, that it would
//   // lead to an out of bounds error.
//   if pos + rotation < 0 {
//       // Hence wrap around and use a position at the end of the polynomial.
//       debug_assert!(rotation < 0);
//       (pos + poly_len + rotation) as usize
//   }
//   // The position is at the end, the rotation is positive and so large, that it would lead to an
//   // out of bounds error.
//   else if pos + rotation > poly_len {
//       // Hence wrap around and use a position at the beginning of the polynomial.
//       debug_assert!(rotation > 0);
//       (pos - poly_len + rotation) as usize
//   }
//   // It is outside those range, hence the rotation (being positive or negative) won't lead to an
//   // out of bounds position.
//   else {
//       (pos + rotation) as usize
//   }
//}

//fn vmx_get_of_rotated_pos(
//    pos: i32,
//    rotation: i32,
//    poly_len: i32,
//) -> usize {
//    let new_pos = pos + rotation;
//    // The position is at the beginning, the rotation is negative and so large, that it would
//    // lead to an out of bounds error.
//    if new_pos < 0 {
//        // Hence wrap around and use a position at the end of the polynomial.
//        (poly_len + new_pos) as usize
//    }
//    // The position is at the end, the rotation is positive and so large, that it would lead to an
//    // out of bounds error.
//    else if new_pos > poly_len {
//        // Hence wrap around and use a position at the beginning of the polynomial.
//        (new_pos - poly_len) as usize
//    }
//    // It is outside those range, hence the rotation (being positive or negative) won't lead to an
//    // out of bounds position.
//    else {
//        new_pos as usize
//    }
//}

fn get_of_rotated_pos(pos: usize, rotation: i32, poly_len: usize) -> usize {
    // Making sure that casting from `i32` to `usize` is OK.
    debug_assert!(usize::BITS >= 32, "Platform must be >= 32-bit");

    let new_pos =
        i32::try_from(pos).expect("Polynomial cannot have more then 2^31 coefficients") + rotation;
    // The position is at the beginning, the rotation is negative and so large, that it would
    // lead to an out of bounds error.
    if new_pos < 0 {
        // Hence wrap around and use a position at the end of the polynomial.
        poly_len - new_pos.abs() as usize
    }
    // The position is at the end, the rotation is positive and so large, that it would lead to an
    // out of bounds error.
    else if new_pos as usize > poly_len {
        // Hence wrap around and use a position at the beginning of the polynomial.
        new_pos as usize - poly_len
    }
    // It is outside those range, hence the rotation (being positive or negative) won't lead to an
    // out of bounds position.
    else {
        new_pos as usize
    }
}

fn to_fp_from_raw<F: PrimeField>(elem: &F) -> String {
    let repr = elem.to_repr();
    let mut result = "Fp::from_raw([".to_string();
    result.push_str(&format!(
        "0x{:016x},",
        u64::from_le_bytes(repr.as_ref()[0..8].try_into().unwrap())
    ));
    result.push_str(&format!(
        "0x{:016x},",
        u64::from_le_bytes(repr.as_ref()[8..16].try_into().unwrap())
    ));
    result.push_str(&format!(
        "0x{:016x},",
        u64::from_le_bytes(repr.as_ref()[16..24].try_into().unwrap())
    ));
    result.push_str(&format!(
        "0x{:016x}",
        u64::from_le_bytes(repr.as_ref()[24..32].try_into().unwrap())
    ));
    result.push_str("])");
    result
}

fn to_fp_to_cuda<F: PrimeField>(elem: &F) -> String {
    let bytes = unsafe {
        std::slice::from_raw_parts(elem as *const F as *const u8, mem::size_of_val(elem))
    };
    let mut result = "{ { ".to_string();
    result.push_str(&format!(
        "0x{:08x},",
        u32::from_le_bytes(bytes[0..4].try_into().unwrap())
    ));
    result.push_str(&format!(
        "0x{:08x},",
        u32::from_le_bytes(bytes[4..8].try_into().unwrap())
    ));
    result.push_str(&format!(
        "0x{:08x},",
        u32::from_le_bytes(bytes[8..12].try_into().unwrap())
    ));
    result.push_str(&format!(
        "0x{:08x},",
        u32::from_le_bytes(bytes[12..16].try_into().unwrap())
    ));
    result.push_str(&format!(
        "0x{:08x},",
        u32::from_le_bytes(bytes[16..20].try_into().unwrap())
    ));
    result.push_str(&format!(
        "0x{:08x},",
        u32::from_le_bytes(bytes[20..24].try_into().unwrap())
    ));
    result.push_str(&format!(
        "0x{:08x},",
        u32::from_le_bytes(bytes[24..28].try_into().unwrap())
    ));
    result.push_str(&format!(
        "0x{:08x}",
        u32::from_le_bytes(bytes[28..32].try_into().unwrap())
    ));
    result.push_str(" } }");
    result
}

fn to_fp_to_opencl<F: PrimeField>(elem: &F) -> String {
    let bytes = unsafe {
        std::slice::from_raw_parts(elem as *const F as *const u8, mem::size_of_val(elem))
    };
    let mut result = "{ { ".to_string();
    result.push_str(&format!(
        "0x{:016x},",
        u64::from_le_bytes(bytes[0..8].try_into().unwrap())
    ));
    result.push_str(&format!(
        "0x{:016x},",
        u64::from_le_bytes(bytes[8..16].try_into().unwrap())
    ));
    result.push_str(&format!(
        "0x{:016x},",
        u64::from_le_bytes(bytes[16..24].try_into().unwrap())
    ));
    result.push_str(&format!(
        "0x{:016x}",
        u64::from_le_bytes(bytes[24..32].try_into().unwrap())
    ));
    result.push_str(" } }");
    result
}

/// Converts a list of polynomials (with the same size each) into a linear byte buffer).
fn polys_to_bytes(polys: &[Polynomial<Fp, ExtendedLagrangeCoeff>]) -> Vec<u8> {
    let mut bytes = Vec::new();

    println!(
        "vmx: polys_to_bytes: num polys, poly_len: {} {}",
        polys.len(),
        polys.first().unwrap().len()
    );
    let mut tmp_poly = Vec::new();
    for poly in polys {
        poly.write(&mut tmp_poly).unwrap();
        // `write` prepends the length of the polynomial as 32-bit prefix. Strip that off.
        bytes.extend_from_slice(&tmp_poly[4..]);
        tmp_poly.clear();
    }

    bytes
}

/// Converts a linear buffer of polynomials into a vector of `Polynomial`.
fn bytes_to_polys(
    bytes: &[u8],
    num_polys: usize,
    poly_len: usize,
) -> Vec<Polynomial<Fp, ExtendedLagrangeCoeff>> {
    (0..num_polys)
        .map(|offset| {
            let start = offset * poly_len * mem::size_of::<Fp>();
            let end = (offset + 1) * poly_len * mem::size_of::<Fp>();
            let buffer = bytes[start..end].to_vec();
            Polynomial::<Fp, ExtendedLagrangeCoeff>::from_bytes(buffer)
        })
        .collect()
}

// Based on poly/evaluator.rs.
/// Traverse the AST and generate source code.
///
/// Returns the counter that was used.
fn recurse<E, F: FieldExt + Serialize, B: BasisOps + Serialize>(
    ast: &Ast<E, F, B>,
    ctx: &AstContext<'_, F, B>,
) -> AstReturn {
    let mut source = Vec::new();
    let (result_counter, counter) = match ast {
        Ast::Poly(leaf) => {
            //B::get_chunk_of_rotated(
            //ctx.domain,
            //ctx.chunk_size,
            //ctx.chunk_index,
            //&ctx.polys[leaf.index],
            //leaf.rotation,
            let rotation_abs =
                ((1 << (ctx.domain.extended_k - ctx.domain.k)) * leaf.rotation.0.abs()) as usize;
            //let pos = get_of_rotated_pos(1, leaf.rotation.0 < 0, rotation_abs, ctx.polys[leaf.index].len());
            //let pos = get_of_rotated_pos(65511, leaf.rotation.0 < 0, rotation_abs, ctx.poly_len);

            //source.push(format!("const rotations_abs{}: usize = (1 << {}) * {};", ctx.counter, (ctx.domain.extended_k - ctx.domain.k), leaf.rotation.0.abs()));
            source.push(format!(
                "let poly_pos{}: usize = get_of_rotated_pos(pos, {}, {}, {});",
                ctx.counter,
                leaf.rotation.0 < 0,
                rotation_abs,
                ctx.poly_len
            ));

            source.push(format!(
                "let mut elem{} = polys[{}][poly_pos{}];",
                ctx.counter, leaf.index, ctx.counter
            ));
            (ctx.counter, ctx.counter)
        }
        Ast::Add(a, b) => {
            let lhs = recurse(a, &ctx.increase_counter());
            let rhs = recurse(b, &ctx.set_counter(lhs.counter + 1));
            //for (lhs, rhs) in lhs.iter_mut().zip(rhs.iter()) {
            //    *lhs += *rhs;
            //}
            //lhs
            source.extend_from_slice(&lhs.source);
            source.extend_from_slice(&rhs.source);
            source.push(format!(
                "elem{} += elem{};",
                lhs.result_counter, rhs.result_counter
            ));
            (lhs.result_counter, rhs.counter)
        }
        Ast::Mul(AstMul(a, b)) => {
            let lhs = recurse(a, &ctx.increase_counter());
            let rhs = recurse(b, &ctx.set_counter(lhs.counter + 1));
            //for (lhs, rhs) in lhs.iter_mut().zip(rhs.iter()) {
            //    *lhs *= *rhs;
            //}
            //lhs
            source.extend_from_slice(&lhs.source);
            source.extend_from_slice(&rhs.source);
            source.push(format!(
                "elem{} *= elem{};",
                lhs.result_counter, rhs.result_counter
            ));
            (lhs.result_counter, rhs.counter)
        }
        Ast::Scale(a, scalar) => {
            let lhs = recurse(a, &ctx.increase_counter());
            //for lhs in lhs.iter_mut() {
            //    *lhs *= scalar;
            //}
            //lhs
            source.extend_from_slice(&lhs.source);
            source.push(format!(
                "elem{} *= {};",
                lhs.result_counter,
                to_fp_from_raw(scalar)
            ));
            (lhs.result_counter, lhs.counter)
        }
        Ast::DistributePowers(terms, base) => {
            //terms.iter().fold(
            //    B::constant_term(ctx.poly_len, ctx.chunk_size, ctx.chunk_index, F::zero()),
            //    |mut acc, term| {
            //        let term = recurse(term, ctx);
            //        for (acc, term) in acc.iter_mut().zip(term) {
            //            *acc *= base;
            //            *acc += term;
            //        }
            //    acc
            //    },
            //)
            source.push(format!("let mut result = {};", to_fp_from_raw(&F::zero())));
            // A context for the loop, which is incrases the counter for every iteration correctly.
            let mut ctx_loop = ctx.increase_counter();
            for term in terms.iter() {
                let term_result = recurse(term, &ctx_loop.increase_counter());
                source.extend_from_slice(&term_result.source);
                source.push(format!("result *= {};", to_fp_from_raw(base)));
                source.push(format!("result += elem{};", term_result.result_counter));
                ctx_loop = ctx_loop.set_counter(term_result.counter);
            }
            source.push(format!("result"));
            (ctx.counter, ctx.counter)
        }
        Ast::LinearTerm(scalar) => {
            //    B::linear_term(
            //    ctx.domain,
            //    ctx.poly_len,
            //    ctx.chunk_size,
            //    ctx.chunk_index,
            //    *scalar,
            //)
            // NOTE vmx 2022-10-10: This is specific to ExtendedLagrangeCoeff, others work
            // differently.
            let omega = ctx.domain.get_extended_omega();
            let zeta_scalar = F::ZETA * scalar;
            source.push(format!(
                "let omega{} = {};",
                ctx.counter,
                to_fp_from_raw(&omega)
            ));
            source.push(format!(
                "let mut elem{} = omega{}.pow_vartime(&[pos as u64]) * {};",
                ctx.counter,
                ctx.counter,
                to_fp_from_raw(&zeta_scalar)
            ));
            (ctx.counter, ctx.counter)
        }
        Ast::ConstantTerm(scalar) => {
            //B::constant_term(ctx.poly_len, ctx.chunk_size, ctx.chunk_index, *scalar)
            source.push(format!(
                "let mut elem{} = {};",
                ctx.counter,
                to_fp_from_raw(scalar)
            ));
            (ctx.counter, ctx.counter)
        }
    };
    AstReturn {
        counter,
        result_counter,
        source,
    }
}

/// Traverse the AST and generate source code.
///
/// Returns the counter that was used.
fn recurse_gpu<E, F: FieldExt + Serialize, B: BasisOps + Serialize>(
    ast: &Ast<E, F, B>,
    ctx: &AstContext<'_, F, B>,
) -> AstReturn {
    let mut source = Vec::new();
    let (result_counter, counter) = match ast {
        Ast::Poly(leaf) => {
            let rotation_abs =
                ((1 << (ctx.domain.extended_k - ctx.domain.k)) * leaf.rotation.0.abs()) as usize;
            source.push(format!(
                "const uint poly_pos{} = get_of_rotated_pos(pos, {}, {}, {});",
                ctx.counter,
                leaf.rotation.0 < 0,
                rotation_abs,
                ctx.poly_len
            ));

            source.push(format!(
                "FIELD elem{} = polys[{}][poly_pos{}];",
                ctx.counter, leaf.index, ctx.counter
            ));
            (ctx.counter, ctx.counter)
        }
        Ast::Add(a, b) => {
            let lhs = recurse_gpu(a, &ctx.increase_counter());
            let rhs = recurse_gpu(b, &ctx.set_counter(lhs.counter + 1));
            source.extend_from_slice(&lhs.source);
            source.extend_from_slice(&rhs.source);
            source.push(format!(
                "elem{} = FIELD_add(elem{}, elem{});",
                lhs.result_counter, lhs.result_counter, rhs.result_counter
            ));
            (lhs.result_counter, rhs.counter)
        }
        Ast::Mul(AstMul(a, b)) => {
            let lhs = recurse_gpu(a, &ctx.increase_counter());
            let rhs = recurse_gpu(b, &ctx.set_counter(lhs.counter + 1));
            source.extend_from_slice(&lhs.source);
            source.extend_from_slice(&rhs.source);
            source.push(format!(
                "elem{} = FIELD_mul(elem{}, elem{});",
                lhs.result_counter, lhs.result_counter, rhs.result_counter
            ));
            (lhs.result_counter, rhs.counter)
        }
        Ast::Scale(a, scalar) => {
            let lhs = recurse_gpu(a, &ctx.increase_counter());
            source.extend_from_slice(&lhs.source);
            source.push(format!(
                "elem{} = FIELD_mul(elem{}, {});",
                lhs.result_counter,
                lhs.result_counter,
                to_fp_to_cuda(scalar)
            ));
            (lhs.result_counter, lhs.counter)
        }
        Ast::DistributePowers(terms, base) => {
            // A context for the loop, which is incrases the counter for every iteration correctly.
            let mut ctx_loop = ctx.increase_counter();
            for term in terms.iter() {
                let term_result = recurse_gpu(term, &ctx_loop.increase_counter());
                source.extend_from_slice(&term_result.source);
                source.push(format!(
                    "*result = FIELD_mul(*result, {});",
                    to_fp_to_cuda(base)
                ));
                source.push(format!(
                    "*result = FIELD_add(*result, elem{});",
                    term_result.result_counter
                ));
                ctx_loop = ctx_loop.set_counter(term_result.counter);
            }
            (ctx.counter, ctx.counter)
        }
        Ast::LinearTerm(scalar) => {
            // NOTE vmx 2022-10-10: This is specific to ExtendedLagrangeCoeff, others work
            // differently.
            let omega = ctx.domain.get_extended_omega();
            let zeta_scalar = F::ZETA * scalar;
            source.push(format!(
                "FIELD omega{} = {};",
                ctx.counter,
                to_fp_to_cuda(&omega)
            ));
            source.push(format!(
                "FIELD elem{} = FIELD_mul(FIELD_pow(omega{}, pos), {});",
                ctx.counter,
                ctx.counter,
                to_fp_to_cuda(&zeta_scalar)
            ));
            (ctx.counter, ctx.counter)
        }
        Ast::ConstantTerm(scalar) => {
            source.push(format!(
                "FIELD elem{} = {};",
                ctx.counter,
                to_fp_to_cuda(scalar)
            ));
            (ctx.counter, ctx.counter)
        }
    };
    AstReturn {
        counter,
        result_counter,
        source,
    }
}

/// Traverse the AST and generate a stack machine.
fn recurse_stack_machine<E, F: FieldExt + Serialize, B: BasisOps + Serialize>(
    ast: &Ast<E, F, B>,
    domain: &EvaluationDomain<F>,
) -> Vec<String> {
    let mut instructions = Vec::new();
    // The stack starts with a zero field element, hence it's `1`.
    let mut stack_size = 1;
    let mut max_stack_size = 0;
    match ast {
        Ast::Poly(leaf) => {
            let rotation_abs =
                ((1 << (domain.extended_k - domain.k)) * leaf.rotation.0.abs()) as usize;
            // Pushes the field element at `[poly_index][result-of-the-call]`;
            // TODO vmx 2022-12-01: `pos` and `poly_len` are input parameters of the stack machine.
            instructions.push(format!(
                "get_of_rotated_pos: poly_index={} rotation_is_negative={} rotation={}",
                leaf.index,
                leaf.rotation.0 < 0,
                rotation_abs
            ));
            stack_size += 1;
            max_stack_size = cmp::max(max_stack_size, stack_size);
        }
        Ast::Add(a, b) => {
            let lhs = recurse_stack_machine(a, domain);
            let rhs = recurse_stack_machine(b, domain);
            instructions.extend_from_slice(&lhs);
            instructions.extend_from_slice(&rhs);
            // Pops two elements, adds them and pushes the result.
            instructions.push("add".to_string());
            stack_size -= 1;
        }
        Ast::Mul(AstMul(a, b)) => {
            let lhs = recurse_stack_machine(a, domain);
            let rhs = recurse_stack_machine(b, domain);
            instructions.extend_from_slice(&lhs);
            instructions.extend_from_slice(&rhs);
            // Pops two elements, multiplies them and pushes the result.
            instructions.push("mul".to_string());
            stack_size -= 1;
        }
        Ast::Scale(a, scalar) => {
            let lhs = recurse_stack_machine(a, domain);
            instructions.extend_from_slice(&lhs);
            // Pops one elements, scales it and pushes the result.
            instructions.push(format!("scale: scalar={}", to_fp_to_cuda(scalar)));
        }
        // This is the entry point of the AST.
        Ast::DistributePowers(terms, base) => {
            for term in terms.iter() {
                let term = recurse_stack_machine(term, domain);
                instructions.extend_from_slice(&term);
                // Pushes one element, pops two elements, multiplies them and pushes the result.
                instructions.push(format!("push: value={}", to_fp_to_cuda(base)));
                instructions.push("mul".to_string());
                // Pops two elements, adds then and pushes the result.
                instructions.push("add".to_string());
                stack_size -= 1;
            }
        }
        Ast::LinearTerm(scalar) => {
            // NOTE vmx 2022-10-10: This is specific to ExtendedLagrangeCoeff, others work
            // differently.
            let omega = domain.get_extended_omega();
            let zeta_scalar = F::ZETA * scalar;
            //// Pushes two elements
            //instructions.push(format!("push: value={}", to_fp_to_cuda(&omega)));
            //instructions.push(format!("push: value={}", to_fp_to_cuda(&zeta_scalar)));
            // Does some calculations and pushes the result.
            // The pos is given as a global variable.
            instructions.push(format!(
                "linearterm: omega={} zeta={}",
                to_fp_to_cuda(&omega),
                to_fp_to_cuda(&zeta_scalar)
            ));
            stack_size += 1;
            max_stack_size = cmp::max(max_stack_size, stack_size);
        }
        Ast::ConstantTerm(scalar) => {
            // Pushes one element.
            instructions.push(format!("push: value={}", to_fp_to_cuda(scalar)));
            stack_size += 1;
            max_stack_size = cmp::max(max_stack_size, stack_size);
        }
    };
    instructions
}

/// Traverse the AST and generate a stack machine in Rust that can be executed.
fn recurse_stack_rust<E, F: FieldExt + Serialize, B: BasisOps + Serialize>(
    ast: &Ast<E, F, B>,
    domain: &EvaluationDomain<F>,
) -> Vec<String> {
    let mut instructions = Vec::new();
    match ast {
        Ast::Poly(leaf) => {
            let rotation_abs =
                ((1 << (domain.extended_k - domain.k)) * leaf.rotation.0.abs()) as usize;
            // Pushes the field element at `[poly_index][result-of-the-call]`;
            // TODO vmx 2022-12-01: `pos` and `poly_len` are input parameters of the stack machine.
            instructions.push(format!(
                "stack.push(polys[{}][get_of_rotated_pos(pos, {}, {}, POLY_LEN)]);",
                leaf.index,
                leaf.rotation.0 < 0,
                rotation_abs,
            ));
        }
        Ast::Add(a, b) => {
            let lhs = recurse_stack_rust(a, domain);
            let rhs = recurse_stack_rust(b, domain);
            instructions.extend_from_slice(&lhs);
            instructions.extend_from_slice(&rhs);
            // Pops two elements, adds them and pushes the result.
            instructions
                .push("stack.push(stack.pop().unwrap() + stack.pop().unwrap());".to_string());
        }
        Ast::Mul(AstMul(a, b)) => {
            let lhs = recurse_stack_rust(a, domain);
            let rhs = recurse_stack_rust(b, domain);
            instructions.extend_from_slice(&lhs);
            instructions.extend_from_slice(&rhs);
            // Pops two elements, multiplies them and pushes the result.
            instructions
                .push("stack.push(stack.pop().unwrap() * stack.pop().unwrap());".to_string());
        }
        Ast::Scale(a, scalar) => {
            let lhs = recurse_stack_rust(a, domain);
            instructions.extend_from_slice(&lhs);
            // Pops one elements, scales it and pushes the result.
            instructions.push(format!(
                "stack.push(stack.pop().unwrap() * {});",
                to_fp_from_raw(scalar)
            ));
        }
        // This is the entry point of the AST.
        Ast::DistributePowers(terms, base) => {
            instructions.push(format!(
                "let mut stack = vec![{}];",
                to_fp_from_raw(&F::zero())
            ));
            for term in terms.iter() {
                let term = recurse_stack_rust(term, domain);
                instructions.extend_from_slice(&term);
                // Pushes one element, pops two elements, multiplies them and pushes the result.
                instructions.push(format!("stack.push({});", to_fp_from_raw(base)));
                instructions
                    .push("stack.push(stack.pop().unwrap() * stack.pop().unwrap());".to_string());
                // Pops two elements, adds then and pushes the result.
                instructions
                    .push("stack.push(stack.pop().unwrap() + stack.pop().unwrap());".to_string());
            }
            instructions.push("stack.pop().unwrap()".to_string());
        }
        Ast::LinearTerm(scalar) => {
            // NOTE vmx 2022-10-10: This is specific to ExtendedLagrangeCoeff, others work
            // differently.
            let omega = domain.get_extended_omega();
            let zeta_scalar = F::ZETA * scalar;
            //// Pushes two elements
            //instructions.push(format!("push: value={}", to_fp_to_cuda(&omega)));
            //instructions.push(format!("push: value={}", to_fp_to_cuda(&zeta_scalar)));
            // Does some calculations and pushes the result.
            // The pos is given as a global variable.
            // TODO vmx 2022-12-07: Omega is static per domain. so it can be defined once globally.
            instructions.push(format!(
                "stack.push({}.pow_vartime(&[pos as u64]) * {});",
                to_fp_from_raw(&omega),
                to_fp_from_raw(&zeta_scalar)
            ));
        }
        Ast::ConstantTerm(scalar) => {
            // Pushes one element.
            instructions.push(format!("stack.push({});", to_fp_from_raw(scalar)));
        }
    };
    instructions
}

#[derive(Debug, Clone)]
enum Instruction<F: FieldExt> {
    /// Pops two elements, adds them and pushes the result.
    Add,
    /// Pops two elements, multiplies them and pushes the result.
    Mul,
    /// Pops one element, scales it and pushes the result.
    Scale { scalar: F },
    /// Pushes one element.
    Push { element: F },
    /// Does some calculations and pushes the result. The position and omega is passed into the
    /// stack machine.
    LinearTerm { zeta_scalar: F },
    /// Pushes the field element at `[poly_index][result-of-the-call]`;
    Rotated { index: u32, rotation: i32 },
}

fn fieldext_to_bytes<F: FieldExt>(element: &F) -> &[u8] {
    debug_assert_eq!(mem::size_of::<F>(), 32);
    unsafe { slice::from_raw_parts(element as *const F as *const u8, mem::size_of::<F>()) }
}

impl<F: FieldExt> Instruction<F> {
    /// Converts the instruction into bytes that can be interpreted as a tagged union of structs in
    /// C.
    pub fn to_bytes(&self) -> [u8; 33] {
        let mut bytes = [0; 33];
        match self {
            Self::Add => {
                bytes[0] = 1;
            }
            Self::Mul => {
                bytes[0] = 2;
            }
            Self::Scale { scalar } => {
                bytes[0] = 3;
                bytes[1..33].copy_from_slice(fieldext_to_bytes(scalar));
            }
            Self::Push { element } => {
                bytes[0] = 4;
                bytes[1..33].copy_from_slice(fieldext_to_bytes(element));
            }
            Self::LinearTerm { zeta_scalar } => {
                bytes[0] = 5;
                bytes[1..33].copy_from_slice(fieldext_to_bytes(zeta_scalar));
            }
            Self::Rotated { index, rotation } => {
                bytes[0] = 6;
                bytes[1..5].copy_from_slice(&index.to_le_bytes());
                bytes[5..9].copy_from_slice(&rotation.to_le_bytes());
            }
        }
        bytes
    }
}

/// Contains the instructions and the maximum size of the stack.
#[derive(Debug, Default)]
struct StackContext<F: FieldExt> {
    instructions: Vec<Instruction<F>>,
    stack_size: usize,
    max_stack_size: usize,
}

/// Traverse the AST and generate a stack machine in Rust that can be executed.
fn ast_to_stack_machine_rust<E, F: FieldExt + Serialize, B: BasisOps + Serialize>(
    ast: &Ast<E, F, B>,
    domain: &EvaluationDomain<F>,
    ctx: &StackContext<F>,
) -> StackContext<F> {
    let mut instructions = Vec::new();
    match ast {
        Ast::Poly(leaf) => {
            // Pushes the field element at `[poly_index][result-of-the-call]`;
            let rotation = i32::try_from((1 << (domain.extended_k - domain.k)) * leaf.rotation.0)
                .expect("Polynomial cannot have more then 2^31 coefficients");
            instructions.push(Instruction::Rotated {
                index: u32::try_from(leaf.index)
                    .expect("Polynomial cannot have more then 2^32 coefficients"),
                rotation,
            });
            let stack_size = ctx.stack_size + 1;
            StackContext {
                instructions,
                stack_size,
                max_stack_size: cmp::max(stack_size, ctx.max_stack_size),
            }
        }
        Ast::Add(a, b) => {
            let lhs = ast_to_stack_machine_rust(a, domain, ctx);
            let rhs = ast_to_stack_machine_rust(b, domain, &lhs);
            instructions.extend_from_slice(&lhs.instructions);
            instructions.extend_from_slice(&rhs.instructions);
            // Pops two elements, adds them and pushes the result.
            instructions.push(Instruction::Add);
            //let max_stack_size = cmp::max(lhs.max_stack_size, rhs.max_stack_size);
            StackContext {
                instructions,
                stack_size: rhs.stack_size - 1,
                max_stack_size: cmp::max(rhs.max_stack_size, ctx.max_stack_size),
            }
        }
        Ast::Mul(AstMul(a, b)) => {
            let lhs = ast_to_stack_machine_rust(a, domain, ctx);
            let rhs = ast_to_stack_machine_rust(b, domain, &lhs);
            instructions.extend_from_slice(&lhs.instructions);
            instructions.extend_from_slice(&rhs.instructions);
            // Pops two elements, multiplies them and pushes the result.
            instructions.push(Instruction::Mul);
            //let max_stack_size = cmp::max(lhs.max_stack_size, rhs.max_stack_size);
            StackContext {
                instructions,
                stack_size: rhs.stack_size - 1,
                max_stack_size: cmp::max(rhs.max_stack_size, ctx.max_stack_size),
            }
        }
        Ast::Scale(a, scalar) => {
            let lhs = ast_to_stack_machine_rust(a, domain, ctx);
            instructions.extend_from_slice(&lhs.instructions);
            // Pops one element, scales it and pushes the result.
            instructions.push(Instruction::Scale { scalar: *scalar });
            StackContext {
                instructions,
                stack_size: lhs.stack_size,
                max_stack_size: cmp::max(lhs.max_stack_size, ctx.max_stack_size),
            }
        }
        // TODO vmx 2022-12-08: Think about moving this outside this function, as it really is the
        // entry point and we won't match on it again. This might simplify the code a bit.
        // This is the entry point of the AST.
        Ast::DistributePowers(terms, base) => {
            let mut max_stack_size = 0;
            instructions.push(Instruction::Push { element: F::zero() });
            for term in terms.iter() {
                instructions.push(Instruction::Push { element: *base });
                instructions.push(Instruction::Mul);
                let term = ast_to_stack_machine_rust(
                    term,
                    domain,
                    &StackContext {
                        instructions: Vec::new(),
                        stack_size: 1,
                        max_stack_size: 1,
                    },
                );
                instructions.extend_from_slice(&term.instructions);
                // Pushes one element, pops two elements, multiplies them and pushes the result.
                // Pops two elements, adds then and pushes the result.
                instructions.push(Instruction::Add);

                // The stack contains a single element (the result) after processing a single term.
                // Hence we can take whatever the maximum of all the runs was.
                max_stack_size = cmp::max(max_stack_size, term.max_stack_size);
            }
            StackContext {
                instructions,
                stack_size: 1,
                max_stack_size, //: cmp::max(max_stack_size, ctx.max_stack_size),
            }
        }
        Ast::LinearTerm(scalar) => {
            // NOTE vmx 2022-10-10: This is specific to ExtendedLagrangeCoeff, others work
            // differently.
            let zeta_scalar = F::ZETA * scalar;
            // Does some calculations and pushes the result.
            instructions.push(Instruction::LinearTerm { zeta_scalar });
            let stack_size = ctx.stack_size + 1;
            StackContext {
                instructions,
                stack_size,
                max_stack_size: cmp::max(stack_size, ctx.max_stack_size),
            }
        }
        Ast::ConstantTerm(scalar) => {
            // Pushes one element.
            instructions.push(Instruction::Push { element: *scalar });
            let stack_size = ctx.stack_size + 1;
            StackContext {
                instructions,
                stack_size,
                max_stack_size: cmp::max(stack_size, ctx.max_stack_size),
            }
        }
    }
}

/// Run the stack machine that the given position.
fn run_stack_machine<F: FieldExt>(
    instructions: &[Instruction<F>],
    omega: &F,
    polys: &[Polynomial<F, ExtendedLagrangeCoeff>],
    poly_len: usize,
    pos: usize,
) -> F {
    let mut stack = Vec::new();
    for instruction in instructions {
        //println!("vmx: stack: {:?}", stack);
        //println!("vmx: stack: len: {}", stack.len());
        match instruction {
            &Instruction::Add => {
                let lhs = stack.pop().unwrap();
                let rhs = stack.pop().unwrap();
                stack.push(lhs + rhs);
            }
            &Instruction::Mul => {
                let lhs = stack.pop().unwrap();
                let rhs = stack.pop().unwrap();
                stack.push(lhs * rhs);
            }
            &Instruction::Scale { scalar } => {
                let lhs = stack.pop().unwrap();
                stack.push(lhs * scalar);
            }
            &Instruction::Push { element } => {
                stack.push(element);
            }
            &Instruction::LinearTerm { zeta_scalar } => {
                stack.push(omega.pow_vartime(&[pos as u64]) * zeta_scalar);
            }
            &Instruction::Rotated { index, rotation } => {
                let rotated_pos = get_of_rotated_pos(pos, rotation, poly_len);
                let index_usize = usize::try_from(index).expect("Platform must be >= 32-bit");
                stack.push(polys[index_usize][rotated_pos]);
            }
        }
    }
    stack.pop().unwrap()
}
//// From https://stackoverflow.com/questions/28127165/how-to-convert-struct-to-u8/42186553#42186553
//unsafe fn to_bytes<T: Sized>(p: &T) -> &[u8] {
//    slice::from_raw_parts((p as *const T) as *const u8, mem::size_of::<T>())
//}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() != 4 {
        println!(
            "Usage: {} <ast-file> <polys-file> <eval|gen|cuda|stacksrc|stackrust|stackmachinerust>",
            args[0]
        );
        process::exit(1);
    }

    let ast_path = &args[1];
    let polys_path = &args[2];
    let mode = &args[3];

    let parameters: Vec<&str> = Path::new(ast_path)
        .file_name()
        .unwrap()
        .to_str()
        .unwrap()
        .split("-")
        .collect();
    let j = parameters[1].parse::<u32>().unwrap();
    let k = parameters[2].parse::<u32>().unwrap();
    println!("parameters: j={}, k={}", j, k);

    println!("deserializing AST");
    let ast_bytes = fs::read(ast_path).unwrap();
    let ast: Ast<u8, Fp, ExtendedLagrangeCoeff> =
        bincode::deserialize(&ast_bytes).expect("AST cannot be serialized");
    //println!("vmx: ast: {:?}", ast);
    //let ast_json = serde_json::to_string(&ast).unwrap();
    //println!("vmx: ast:\n{}", ast_json);

    println!("deserializing polynomials");
    let mut polys_file = File::open(polys_path).unwrap();
    let mut buffer = [0u8; 4];
    polys_file.read_exact(&mut buffer[..]).unwrap();
    let num_polys = usize::try_from(u32::from_le_bytes(buffer)).unwrap();
    println!("num polys: {}", num_polys);
    polys_file.read_exact(&mut buffer[..]).unwrap();
    let poly_len = usize::try_from(u32::from_le_bytes(buffer)).unwrap();
    println!("poly len: {}", poly_len);
    println!(
        "polys byte size: {}",
        num_polys * poly_len * mem::size_of::<Fp>()
    );
    // Linear memory of all the polynomials.
    let mut polys_bytes = vec![0; num_polys * poly_len * mem::size_of::<Fp>()];
    polys_file.read_exact(&mut polys_bytes).unwrap();

    println!("evaluating");
    let domain = EvaluationDomain::<Fp>::new(j, k);

    //// AST with the first element only
    ////let ast_subset = match ast {
    //let ast = match ast {
    // Ast::DistributePowers(terms, base) => {
    //     println!("vmx: num terms: {}", terms.len());
    //     //let subset = terms[38..39].to_vec();
    //     let subset = terms[140..141].to_vec();
    //     Ast::DistributePowers(std::sync::Arc::new(subset), base)
    // },
    // _ => panic!("not supported")
    //};

    ////let ast_json = serde_json::to_string(&ast_subset).unwrap();
    //let ast_json = serde_json::to_string(&ast).unwrap();
    //println!("vmx: ast:\n{}", ast_json);

    match &mode[..] {
        "eval" => {
            let polys = bytes_to_polys(&polys_bytes, num_polys, poly_len);

            let evaluator = Evaluator { polys, _context: 0 };
            let result = evaluator.evaluate(&ast, &domain);
            //let result = evaluator.evaluate(&ast_subset, &domain);
            println!("result full: {:?}", result[0]);
        }
        "gen" => {
            let polys = bytes_to_polys(&polys_bytes, num_polys, poly_len);

            let ctx = AstContext {
                domain: &domain,
                poly_len,
                polys: &polys,
                counter: 0,
            };

            let result = recurse(&ast, &ctx);
            // NOTE vmx 2022-10-20: Only do a subset for now
            //let result = recurse(&ast_subset, &ctx);

            let mut source = Vec::new();
            source.push("#![allow(unused_mut)]".to_string());
            source.push("use halo2_proofs::{arithmetic::Field, pasta::Fp, poly::{ExtendedLagrangeCoeff, Polynomial}};".to_string());
            source.push(r#"
const fn get_of_rotated_pos(pos: usize, rotation_is_negative: bool, rotation_abs: usize, poly_len: usize) -> usize {
    let (mid, k) = if rotation_is_negative {
        (poly_len - rotation_abs, rotation_abs)
    } else {
         (rotation_abs, poly_len - rotation_abs)
    };

    if pos < k {
        mid + pos
    } else {
        pos - k
    }
}"#.to_string());
            source.push("pub fn evaluate(pos: usize, polys: &[Polynomial<Fp, ExtendedLagrangeCoeff>]) -> Fp {".to_string());
            source.extend_from_slice(&result.source);
            source.push("}".to_string());
            //println!("source: {}", source.join("\n"));
            let mut outfile = File::create("/tmp/evaluate.rs").unwrap();
            outfile.write_all(source.join("\n").as_bytes()).unwrap();

            let result = evaluate::evaluate(0, &polys);
            println!("result first (manual): {:?}", result);

            //let result = (0..poly_len)
            //    .into_par_iter()
            //    .map(|pos| evaluate::evaluate(pos, &polys))
            //    .collect::<Vec<_>>();
            //println!("result (gen) full: {:?}", result[0]);
        }
        "cuda" => {
            let polys = bytes_to_polys(&polys_bytes, num_polys, poly_len);

            let ctx = AstContext {
                domain: &domain,
                poly_len,
                polys: &polys,
                counter: 0,
            };

            let result = recurse_gpu(&ast, &ctx);
            // NOTE vmx 2022-10-20: Only do a subset for now
            //let result = recurse_gpu(&ast_subset, &ctx);

            let mut source = Vec::new();
            //source.push("#![allow(unused_mut)]".to_string());
            //source.push("use halo2_proofs::{arithmetic::Field, pasta::Fp, poly::{ExtendedLagrangeCoeff, Polynomial}};".to_string());
            // TODO vmx 2022-11-11: Find a better way than using a define here
            source.push(format!("#define POLY_LEN {}", poly_len));
            source.push(r#"
DEVICE uint get_of_rotated_pos(uint pos, bool rotation_is_negative, uint rotation_abs, uint poly_len) {
    uint mid;
    uint k;
    if (rotation_is_negative) {
        mid = poly_len - rotation_abs;
        k = rotation_abs;
    } else {
        mid = rotation_abs;
        k = poly_len - rotation_abs;
    }

    if (pos < k) {
        return mid + pos;
    } else {
        return pos - k;
    }
}"#.to_string());

            source.push("DEVICE void evaluate_at_pos(GLOBAL FIELD polys[][POLY_LEN], GLOBAL FIELD* result, uint pos) {".to_string());
            source.extend_from_slice(&result.source);
            source.push("}".to_string());

            source.push(
                r#"
// `poly_len` is the lengths of a single polynomial (all have the same length).
KERNEL void evaluate(GLOBAL FIELD polys[][POLY_LEN], GLOBAL FIELD* result, uint poly_len) {
    const uint index = GET_GLOBAL_ID();
    // TODO vmx 2022-10-22: Add the stride to common.cl in ec-gpu-gen and add an OpenCL version.
    const uint stride = blockDim.x * gridDim.x;

    for (uint i = index; i < poly_len; i += stride) {
        // TODO vmx 2022-11-11: check if this if statement is really needed.
        if (i <= poly_len) {
            evaluate_at_pos(polys, &result[i], i);
        }
    }

}"#
                .to_string(),
            );

            // Only write a new file if the `cuda` feature is *not* given. This way the kernel
            // isn't rebuilt on every run.
            #[cfg(not(feature = "cuda"))]
            {
                //println!("source: {}", source.join("\n"));
                let mut outfile = File::create("/tmp/evaluate.cu").unwrap();
                outfile.write_all(source.join("\n").as_bytes()).unwrap();
            }

            #[cfg(feature = "cuda")]
            {
                use ec_gpu_gen::rust_gpu_tools::{program_closures, Device, Program};
                use ec_gpu_gen::EcResult;

                // NOTE vmx 2022-10-23: This value is arbitrarily choosen.
                const LOCAL_WORK_SIZE: usize = 128;

                let devices = Device::all();
                let device = devices.first().unwrap();
                let program = ec_gpu_gen::program!(device).unwrap();

                let closures = program_closures!(|program, _arg| -> EcResult<Vec<Fp>> {
                    // All polynomials have the same length.
                    //let poly_len = polys.first().unwrap().len();

                    //let polys_bytes = polys_to_bytes(&polys);
                    println!("vmx: polys bytes len: {:?}", polys_bytes.len());
                    let polys_buffer = program.create_buffer_from_slice(&polys_bytes)?;
                    //let mut polys_buffer = unsafe {
                    //    program.create_buffer::<Fp>(&poly_len * polys.len())?
                    //};
                    //program.write_from_buffer(&mut polys_buffer, &polys);

                    //// It is safe as the GPU will initialize that buffer
                    //let result_buffer = unsafe { program.create_buffer::<Fp>(poly_len)? };
                    let result_buffer =
                        program.create_buffer_from_slice(&vec![Fp::zero(); poly_len])?;

                    // The global work size follows CUDA's definition and is the number of
                    // `LOCAL_WORK_SIZE` sized thread groups.
                    // NOTE vmx 2022-10-23: This value is arbitrarily choosen.
                    let global_work_size = 1024;

                    let kernel =
                        program.create_kernel("evaluate", global_work_size, LOCAL_WORK_SIZE)?;

                    kernel
                        .arg(&polys_buffer)
                        .arg(&result_buffer)
                        .arg(&(poly_len as u32))
                        .run()?;

                    let mut results = vec![Fp::zero(); poly_len];
                    program.read_into_buffer(&result_buffer, &mut results)?;

                    Ok(results)
                });

                let results = program.run(closures, ()).unwrap();
                println!("result (cuda) full: {:?}", results[0]);
            }

            ////let result = evaluate::evaluate(0, &polys);
            ////println!("result first (manual): {:?}", result);
            //
            //let result = (0..poly_len)
            //    .into_par_iter()
            //    .map(|pos| evaluate::evaluate(pos, &polys))
            //    .collect::<Vec<_>>();
            //println!("result (cuda) full: {:?}", result[0]);
        }
        "stacksrc" => {
            let source = recurse_stack_machine(&ast, &domain);
            //println!("vmx: stack:\n{:?}", result);
            //print!("stack machine source:");
            //for line in result {
            //    println!("{}", line);
            //}
            let mut outfile = File::create("/tmp/stackmaschine.txt").unwrap();
            outfile.write_all(source.join("\n").as_bytes()).unwrap();
        }
        "stackrust" => {
            let stack_machine_source = recurse_stack_rust(&ast, &domain);
            //println!("vmx: stack:\n{:?}", result);
            //print!("stack machine source:");
            //for line in result {
            //    println!("{}", line);
            //}
            let mut source = Vec::new();
            source.push("#![allow(unused_mut)]".to_string());
            source.push("use halo2_proofs::{arithmetic::Field, pasta::Fp, poly::{ExtendedLagrangeCoeff, Polynomial}};".to_string());
            source.push(format!("const POLY_LEN: usize = {};", poly_len));
            source.push(r#"
const fn get_of_rotated_pos(pos: usize, rotation_is_negative: bool, rotation_abs: usize, poly_len: usize) -> usize {
    let (mid, k) = if rotation_is_negative {
        (poly_len - rotation_abs, rotation_abs)
    } else {
         (rotation_abs, poly_len - rotation_abs)
    };

    if pos < k {
        mid + pos
    } else {
        pos - k
    }
}"#.to_string());
            source.push("pub fn evaluate(pos: usize, polys: &[Polynomial<Fp, ExtendedLagrangeCoeff>]) -> Fp {".to_string());
            source.extend_from_slice(&stack_machine_source);
            source.push("}".to_string());
            let mut outfile = File::create("/tmp/stackrust.rs").unwrap();
            outfile.write_all(source.join("\n").as_bytes()).unwrap();
        }
        "stackmachinerust" => {
            let stack_machine = ast_to_stack_machine_rust(&ast, &domain, &StackContext::default());
            println!(
                "vmx: stackmachine: max stack size: {:?}",
                stack_machine.max_stack_size
            );
            //println!("vmx: stackmachine:\n{:?}", stack_machine);
            println!("vmx: stackmachine:");
            //for (ii, instruction) in stack_machine.instructions.iter().enumerate() {
            //    println!("{:?}", instruction);
            //    let mut outfile = File::create(format!("/tmp/instructions/{:0>5}.rs", ii)).unwrap();
            //    let bytes = instruction.to_bytes();
            //    outfile.write_all(&bytes).unwrap();
            //}
            let mut outfile = File::create("/tmp/instructions.dat").unwrap();
            for instruction in &stack_machine.instructions {
               let bytes = instruction.to_bytes();
               outfile.write_all(&bytes).unwrap();
            }
            let polys = bytes_to_polys(&polys_bytes, num_polys, poly_len);
            let omega = domain.get_extended_omega();
            let result =
                run_stack_machine(&stack_machine.instructions, &omega, &polys, poly_len, 0);
            println!("vmx: stackmachine: {:?}", result);
        }
        _ => {
            panic!(
                "Unknown mode `{}`, use `eval`, `gen`, `cuda`, `stacksrc` `stackrust` or `stackmachinerust`",
                mode
            )
        }
    }
}
