#[cfg(not(any(feature = "opencl", feature = "cuda")))]
fn main() {}

#[cfg(any(feature = "opencl", feature = "cuda"))]
fn main() {
    use ec_gpu::GpuName;
    use ec_gpu_gen::SourceBuilder;
    use pasta_curves::{EpAffine, EqAffine, Fp, Fq};

    let halosource = include_str!("/tmp/evaluate.cu")
        .to_string()
        .replace("FIELD", &Fp::name());

    //use std::io::Write;
    //let mut outfile = std::fs::File::create("/tmp/halosource.cu").unwrap();
    //outfile.write_all(halosource.as_bytes()).unwrap();

    let source_builder = SourceBuilder::new()
        .add_fft::<Fp>()
        .add_fft::<Fq>()
        .add_multiexp::<EpAffine, Fp>()
        .add_multiexp::<EqAffine, Fq>()
        .append_source(halosource);

    ec_gpu_gen::generate(&source_builder);
}
