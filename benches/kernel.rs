//! Times every kernel this CPU supports on the non-ASCII inputs; ASCII input
//! returns from the ASCII scan before any kernel runs. Kept in its own binary
//! so these cases don't move the code of the other benchmarks.

mod inputs;

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use simd_utf16_len::__kernels;

fn bench_kernels(c: &mut Criterion) {
    // The original set plus the boundary, medium, large, and distribution
    // groups, so the kernels are timed across the whole size range.
    let mut list = inputs::all();
    list.extend(inputs::boundaries());
    list.extend(inputs::medium());
    list.extend(inputs::large());
    list.extend(inputs::distributions());
    let inputs: Vec<_> = list
        .into_iter()
        .filter(|(_, input)| !input.is_ascii())
        .collect();
    let mut group = c.benchmark_group("kernel");
    // As in benches/utf16_len.rs: many benches, so the default 8 s each is
    // too slow locally, and CodSpeed's simulation mode ignores the timing.
    group
        .warm_up_time(Duration::from_millis(200))
        .measurement_time(Duration::from_millis(800))
        .sample_size(50);
    for kernel in __kernels::available() {
        for (name, input) in &inputs {
            group.bench_function(BenchmarkId::new(*name, kernel.name), |b| {
                b.iter(|| (kernel.utf16_len)(black_box(input)));
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_kernels);
criterion_main!(benches);
