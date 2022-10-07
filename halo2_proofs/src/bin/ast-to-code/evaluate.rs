#![allow(unused_mut)]
use halo2_proofs::{
    arithmetic::Field,
    pasta::Fp,
    poly::{ExtendedLagrangeCoeff, Polynomial},
};

const fn get_of_rotated_pos(
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
pub fn evaluate(pos: usize, polys: &[Polynomial<Fp, ExtendedLagrangeCoeff>]) -> Fp {
    let mut result = Fp::from_raw([
        0x0000000000000000,
        0x0000000000000000,
        0x0000000000000000,
        0x0000000000000000,
    ]);
    result
}
