#[cfg(not(any(feature = "opencl", feature = "cuda")))]
fn main() {}

#[cfg(any(feature = "opencl", feature = "cuda"))]
fn main() {
    use pasta_curves::{EpAffine, EqAffine, Fp, Fq};
    use ec_gpu_gen::SourceBuilder;

    let source_builder = SourceBuilder::new()
        .add_fft::<Fp>();
        //.add_fft::<Fq>();
        //.add_multiexp::<EpAffine, Fp>()
        //.add_multiexp::<EqAffine, Fq>();

    ec_gpu_gen::generate(&source_builder);
}
