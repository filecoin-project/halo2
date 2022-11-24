use std::{
    env,
    fs::{self, File},
    io::{Read, Write},
    mem,
    path::Path,
    process,
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

fn get_of_rotated_pos(
    pos: usize,
    rotation_is_negative: bool,
    rotation_abs: usize,
    poly_len: usize,
) -> usize {
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
    let repr = elem.to_repr();
    let mut result = "{ { ".to_string();
    result.push_str(&format!(
        "0x{:08x},",
        u32::from_le_bytes(repr.as_ref()[0..4].try_into().unwrap())
    ));
    result.push_str(&format!(
        "0x{:08x},",
        u32::from_le_bytes(repr.as_ref()[4..8].try_into().unwrap())
    ));
    result.push_str(&format!(
        "0x{:08x},",
        u32::from_le_bytes(repr.as_ref()[8..12].try_into().unwrap())
    ));
    result.push_str(&format!(
        "0x{:08x},",
        u32::from_le_bytes(repr.as_ref()[12..16].try_into().unwrap())
    ));
    result.push_str(&format!(
        "0x{:08x},",
        u32::from_le_bytes(repr.as_ref()[16..20].try_into().unwrap())
    ));
    result.push_str(&format!(
        "0x{:08x},",
        u32::from_le_bytes(repr.as_ref()[20..24].try_into().unwrap())
    ));
    result.push_str(&format!(
        "0x{:08x},",
        u32::from_le_bytes(repr.as_ref()[24..28].try_into().unwrap())
    ));
    result.push_str(&format!(
        "0x{:08x}",
        u32::from_le_bytes(repr.as_ref()[28..32].try_into().unwrap())
    ));
    result.push_str(" } }");
    result
}

fn to_fp_to_opencl<F: PrimeField>(elem: &F) -> String {
    let repr = elem.to_repr();
    let mut result = "{ { ".to_string();
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

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() != 4 {
        println!("Usage: {} <ast-file> <polys-file> <eval|gen>", args[0]);
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
    println!("polys byte size: {}", num_polys * poly_len * mem::size_of::<Fp>());
    // Linear memory of all the polynomials.
    let mut polys_bytes = vec![0; num_polys * poly_len * mem::size_of::<Fp>()];
    polys_file.read_exact(&mut polys_bytes).unwrap();
    let polys: Vec<_> = (0..num_polys)
        .map(|offset| {
            let start = offset * poly_len * mem::size_of::<Fp>();
            let end = (offset + 1) * poly_len * mem::size_of::<Fp>();
            let buffer = polys_bytes[start..end].to_vec();
            Polynomial::<Fp, ExtendedLagrangeCoeff>::from_bytes(buffer)
        })
        .collect();

    println!("evaluating");
    let domain = EvaluationDomain::<Fp>::new(j, k);

    //// AST with the first element only
    //let ast_subset = match ast {
    //   Ast::DistributePowers(terms, base) => {
    //       println!("vmx: num terms: {}", terms.len());
    //       let subset = terms[38..39].to_vec();
    //       Ast::DistributePowers(std::sync::Arc::new(subset), base)
    //   },
    //   _ => panic!("not supported")
    //};
    //
    //let ast_json = serde_json::to_string(&ast_subset).unwrap();
    //println!("vmx: ast:\n{}", ast_json);

    match &mode[..] {
        "eval" => {
            let evaluator = Evaluator { polys, _context: 0 };
            let result = evaluator.evaluate(&ast, &domain);
            //let result = evaluator.evaluate(&ast_subset, &domain);
            println!("result full: {:?}", result[0]);
        }
        "gen" => {
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

            //let result = evaluate::evaluate(0, &polys);
            //println!("result first (manual): {:?}", result);

            let result = (0..poly_len)
                .into_par_iter()
                .map(|pos| evaluate::evaluate(pos, &polys))
                .collect::<Vec<_>>();
            println!("result (gen) full: {:?}", result[0]);
        }
        "cuda" => {
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
        _ => {
            panic!("Unknown mode `{}`, use `eval`, `gen` or `cuda`", mode)
        }
    }
}
