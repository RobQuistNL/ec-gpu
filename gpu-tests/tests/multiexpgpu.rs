#![cfg(any(feature = "cuda", feature = "opencl"))]

use std::sync::Arc;
use std::time::Instant;

use blstrs::Bls12;
use ec_gpu::GpuName;
use ec_gpu_gen::multiexp_cpu::{FullDensity, QueryDensity, SourceBuilder};
use ec_gpu_gen::{
    multiexp::MultiexpKernel, program, rust_gpu_tools::Device, threadpool::Worker, EcError,
};
use ff::{Field, PrimeField};
use group::Curve;
use group::{prime::PrimeCurveAffine, Group};
use pairing::Engine;

fn multiexp_gpu<Q, D, G, S>(
    pool: &Worker,
    bases: S,
    density_map: D,
    exponents: Arc<Vec<<G::Scalar as PrimeField>::Repr>>,
    kern: &mut MultiexpKernel<G>,
) -> Result<G::Curve, EcError>
where
    for<'a> &'a Q: QueryDensity,
    D: Send + Sync + 'static + Clone + AsRef<Q>,
    G: PrimeCurveAffine + GpuName,
    S: SourceBuilder<G>,
{
    let exps = density_map.as_ref().generate_exps::<G::Scalar>(exponents);
    let (bss, skip) = bases.get();
    kern.multiexp(pool, bss, exps, skip).map_err(Into::into)
}

#[test]
fn gpu_multiexp_consistency() {
    fil_logger::maybe_init();
    const MAX_LOG_D: usize = 28;
    const START_LOG_D: usize = 26;
    const REPEAT: i32 = 100;

    let devices = Device::all();
    let programs = devices
        .iter()
        .map(|device| crate::program!(device))
        .collect::<Result<_, _>>()
        .expect("Cannot create programs!");
    let mut kern = MultiexpKernel::<<Bls12 as Engine>::G1Affine>::create(programs, &devices)
        .expect("Cannot initialize kernel!");
    let pool = Worker::new();

    let mut rng = rand::thread_rng();

    println!("Builing bases...");
    let mut bases = (0..(1 << START_LOG_D-12))
            .map(|_| <Bls12 as Engine>::G1::random(&mut rng).to_affine())
            .collect::<Vec<_>>();

    println!("Expanding bases {} times...", 12);
    for _copy_base in 0..12 {
        bases = [bases.clone(), bases.clone()].concat();
    }

    for repetition in 0..=REPEAT {

        println!("Repetition {}", repetition);
        for log_d in START_LOG_D..=MAX_LOG_D {
            let g = Arc::new(bases.clone());

            let samples = 1 << log_d;
            println!("Building {} elements for Arc samples...", samples);

            let v = Arc::new(
                (0..samples)
                    .map(|_| <Bls12 as Engine>::Fr::random(&mut rng).to_repr())
                    .collect::<Vec<_>>(),
            );

            println!("Running Multiexp on GPU...");
            for internal_rep in 0..=REPEAT {
                let now = Instant::now();
                let _gpu = multiexp_gpu(&pool, (g.clone(), 0), FullDensity, v.clone(), &mut kern).unwrap();
                let gpu_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
                println!("GPU repetition {} took {}ms.", internal_rep, gpu_dur);
            }
            println!("============================");
            if repetition == 0 {
                bases = [bases.clone(), bases.clone()].concat();
            }
        }
    }
}
