mod inputs;

use std::time::Duration;

use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use simd_utf16_len::utf16_len;

const ASCII: &str = "The quick brown fox jumps over the lazy dog. This is a longer sentence to provide more data for benchmarking purposes, with various words and punctuation marks included.";

const CJK: &str = "这是一段中文测试文本，用于测试UTF-8编码中多字节字符的处理性能。日本語のテキストも含まれています。한국어 텍스트도 포함되어 있습니다。";

const EMOJI: &str = "Hello 🌍🌎🌏! Flags: 🇺🇸🇬🇧🇯🇵🇨🇳 Family: 👨\u{200d}👩\u{200d}👧\u{200d}👦 Skin: 👋🏻👋🏼👋🏽👋🏾👋🏿 Fun: 🎉🎊🎈🎁🎄🎃";

const MIXED: &str = "Hello, 世界! 🌍 Привет мир! こんにちは世界！Héllo wörld! 你好世界！안녕하세요 세계! مرحبا بالعالم";

#[inline]
fn ascii_guard_len(s: &str) -> usize {
    if s.is_ascii() {
        s.len()
    } else {
        s.encode_utf16().count()
    }
}

/// The README's standard-library baseline, which skips counting for ASCII.
#[inline]
fn std_len(s: &str) -> usize {
    if s.is_ascii() {
        s.len()
    } else {
        s.encode_utf16().count()
    }
}

/// The new groups hold hundreds of benches, which can't keep criterion's
/// default 3 s warmup + 5 s measurement each: these keep a full local run
/// in minutes. CodSpeed's simulation mode overrides the timing anyway.
fn quick(group: &mut BenchmarkGroup<WallTime>) {
    group
        .warm_up_time(Duration::from_millis(200))
        .measurement_time(Duration::from_millis(800))
        .sample_size(50);
}

// The long ASCII fixture has its own benchmark identity: the historical
// bench_inputs::ascii used 169 bytes, while this fixture uses 10,816 bytes.
fn bench_ascii(c: &mut Criterion) {
    let ascii = ASCII.repeat(64);
    let input = ascii.as_str();
    let mut group = c.benchmark_group("utf16_len");
    group.bench_function(BenchmarkId::new("ascii", "simd"), |b| {
        b.iter(|| utf16_len(black_box(input)));
    });
    group.bench_function(BenchmarkId::new("ascii", "encode_utf16"), |b| {
        b.iter(|| black_box(input).encode_utf16().count());
    });
    group.bench_function(BenchmarkId::new("ascii", "is_ascii"), |b| {
        b.iter(|| ascii_guard_len(black_box(input)));
    });
    group.finish();
}

fn bench_inputs(c: &mut Criterion) {
    let inputs: &[(&str, &str)] = &[("cjk", CJK), ("emoji", EMOJI), ("mixed", MIXED)];

    let mut group = c.benchmark_group("utf16_len");
    for &(name, input) in inputs {
        group.bench_function(BenchmarkId::new(name, "simd"), |b| {
            b.iter(|| utf16_len(black_box(input)));
        });
        group.bench_function(BenchmarkId::new(name, "encode_utf16"), |b| {
            b.iter(|| black_box(input).encode_utf16().count());
        });
    }
    group.finish();
}

/// Every length 0..=64, ASCII and mixed: dispatch and short-path cost at
/// single-byte resolution, in ns/call.
fn bench_tiny(c: &mut Criterion) {
    let mut group = c.benchmark_group("tiny");
    quick(&mut group);
    for (name, input) in inputs::tiny() {
        group.bench_function(BenchmarkId::new(name, "simd"), |b| {
            b.iter(|| utf16_len(black_box(input)));
        });
    }
    group.finish();
}

/// Block, tail, and NEON batch boundary lengths, ASCII and mixed.
fn bench_boundaries(c: &mut Criterion) {
    let mut group = c.benchmark_group("boundary");
    quick(&mut group);
    for (name, input) in inputs::boundaries() {
        group.bench_function(BenchmarkId::new(name, "simd"), |b| {
            b.iter(|| utf16_len(black_box(input)));
        });
    }
    group.finish();
}

/// One 4 KiB ASCII window at every start offset 0..=63.
fn bench_alignment(c: &mut Criterion) {
    let mut group = c.benchmark_group("alignment");
    quick(&mut group);
    for (name, input) in inputs::alignment() {
        group.bench_function(BenchmarkId::new(name, "simd"), |b| {
            b.iter(|| utf16_len(black_box(input)));
        });
    }
    group.finish();
}

/// 256 B through 1 MiB against the standard-library baseline.
fn bench_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("size");
    quick(&mut group);
    let mut inputs = inputs::medium();
    inputs.extend(inputs::large());
    for (name, input) in inputs {
        group.bench_function(BenchmarkId::new(name, "simd"), |b| {
            b.iter(|| utf16_len(black_box(input)));
        });
        group.bench_function(BenchmarkId::new(name, "std"), |b| {
            b.iter(|| std_len(black_box(input)));
        });
    }
    group.finish();
}

/// One 4 KiB input per distribution, plus the shuffled realistic corpus.
fn bench_distributions(c: &mut Criterion) {
    let mut group = c.benchmark_group("distribution");
    quick(&mut group);
    let mut inputs = inputs::distributions();
    inputs.extend(inputs::realistic());
    for (name, input) in inputs {
        group.bench_function(BenchmarkId::new(name, "simd"), |b| {
            b.iter(|| utf16_len(black_box(input)));
        });
        group.bench_function(BenchmarkId::new(name, "std"), |b| {
            b.iter(|| std_len(black_box(input)));
        });
    }
    group.finish();
}

/// The optimized reference: simdutf's dedicated count routine (Section
/// 11.3). Development dependency only, never linked into the crate.
fn bench_simdutf(c: &mut Criterion) {
    let mut group = c.benchmark_group("simdutf");
    quick(&mut group);
    let mut inputs = inputs::medium();
    inputs.extend(inputs::large());
    inputs.extend(inputs::distributions());
    for (name, input) in inputs {
        group.bench_function(BenchmarkId::new(name, "simd_utf16_len"), |b| {
            b.iter(|| utf16_len(black_box(input)));
        });
        group.bench_function(BenchmarkId::new(name, "simdutf"), |b| {
            b.iter(|| simdutf::utf16_length_from_utf8(black_box(input.as_bytes())));
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_ascii,
    bench_inputs,
    bench_tiny,
    bench_boundaries,
    bench_alignment,
    bench_sizes,
    bench_distributions,
    bench_simdutf,
);
criterion_main!(benches);
