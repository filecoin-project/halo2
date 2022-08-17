use pasta_curves::{EpAffine, EqAffine, Fp, Fq};
use ec_gpu_gen::SourceBuilder;

/// The build script is used to generate the CUDA kernel and OpenCL source at compile-time, if the
/// `bls12-381` feature is enabled.
//#[cfg(all(feature = "bls12-381", not(feature = "cargo-clippy")))]
fn main() {
    let source_builder = SourceBuilder::new()
        .add_fft::<Fp>();
        //.add_fft::<Fq>();
        //.add_multiexp::<EpAffine, Fp>()
        //.add_multiexp::<EqAffine, Fq>();

    ec_gpu_gen::generate(&source_builder);
}
