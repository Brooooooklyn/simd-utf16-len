//! Times every kernel this CPU supports, and the `utf16_len` dispatch, on
//! non-ASCII inputs of increasing size, to show where a wider kernel starts
//! to pay off. A streaming section adds the cases too large for the codspeed
//! criterion benches: 64 MiB buffers and a rotating working set.
//!
//! Run `cargo run --release --manifest-path perf/kernels/Cargo.toml`, with
//! `--features avx512` to include the AVX-512 kernel, or the Kernel sweep
//! workflow, which does both on every runner.

use std::fmt::Write as _;
use std::hint::black_box;
use std::process::Command;
use std::time::{Duration, Instant};

use simd_utf16_len::__kernels;

/// A kernel, or the dispatch, under its display name.
type Column = (&'static str, fn(&str) -> usize);

const SIZES: &[usize] = &[
    13, 16, 24, 32, 48, 64, 96, 128, 144, 160, 192, 208, 256, 320, 384, 512, 768, 1024, 2048,
    4096, 16384,
];
/// Target duration of one timed batch.
const BATCH: Duration = Duration::from_millis(1);
const ROUNDS: usize = 21;

/// CJK text, three bytes per character, padded with ASCII to exactly `size`
/// bytes, so every input starts with a non-ASCII byte.
fn input(size: usize) -> String {
    let mut s = "中".repeat(size / 3);
    while s.len() < size {
        s.push('a');
    }
    s
}

fn main() {
    let mut columns: Vec<Column> = __kernels::available()
        .into_iter()
        .map(|kernel| (kernel.name, kernel.utf16_len))
        .collect();
    columns.push(("utf16_len", simd_utf16_len::utf16_len));
    let inputs: Vec<String> = SIZES.iter().map(|&size| input(size)).collect();

    for input in &inputs {
        let expected = input.encode_utf16().count();
        for (name, f) in &columns {
            assert_eq!(f(input), expected, "{name} miscounts {} bytes", input.len());
        }
    }

    let mut out = String::new();
    writeln!(
        out,
        "### Kernel sweep on {} {}\n",
        std::env::consts::OS,
        std::env::consts::ARCH
    )
    .unwrap();
    writeln!(
        out,
        "{}, {}. Median ns per call over {ROUNDS} rounds; every input starts with a non-ASCII byte, and `utf16_len` is the dispatch.\n",
        cpu_model().unwrap_or_else(|| "unknown CPU".to_owned()),
        command_output("rustc", &["--version"]).unwrap_or_else(|| "unknown rustc".to_owned()),
    )
    .unwrap();
    write!(out, "| Bytes |").unwrap();
    for (name, _) in &columns {
        write!(out, " {name} |").unwrap();
    }
    out.push('\n');
    write!(out, "|------:|").unwrap();
    for _ in &columns {
        write!(out, "------:|").unwrap();
    }
    out.push('\n');

    for input in &inputs {
        let iters: Vec<u64> = columns.iter().map(|(_, f)| calibrate(f, input)).collect();
        let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(ROUNDS); columns.len()];
        for _ in 0..ROUNDS {
            // Interleave the columns, so drift within a round affects them equally.
            for (i, (_, f)) in columns.iter().enumerate() {
                let elapsed = time_batch(f, input, iters[i]);
                samples[i].push(elapsed.as_secs_f64() * 1e9 / iters[i] as f64);
            }
        }
        write!(out, "| {} |", input.len()).unwrap();
        for values in &mut samples {
            values.sort_by(f64::total_cmp);
            write!(out, " {:.1} |", values[values.len() / 2]).unwrap();
        }
        out.push('\n');
    }

    streaming(&columns, &mut out);

    print!("{out}");
    if let Ok(path) = std::env::var("GITHUB_STEP_SUMMARY") {
        use std::io::Write as _;
        let written = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(&path)
            .and_then(|mut file| file.write_all(out.as_bytes()));
        if let Err(error) = written {
            eprintln!("failed to append to {path}: {error}");
        }
    }
}

/// The cases too large for the codspeed criterion benches: one 64 MiB
/// buffer per distribution, and a 16 x 4 MiB rotating working set so no
/// call reuses the previous call's buffer. These exceed any single core's
/// caches, so next to ns/call they report the effective rate, which is
/// memory behavior rather than the hot-loop table above.
fn streaming(columns: &[Column], out: &mut String) {
    const MIB: usize = 1 << 20;
    let fixed: [(&str, String); 2] = [
        ("ascii_64mib", "a".repeat(64 * MIB)),
        ("cjk_64mib", input(64 * MIB)),
    ];
    let rotating: Vec<String> = (0..16).map(|_| input(4 * MIB)).collect();

    for (case, input) in &fixed {
        let expected = input.encode_utf16().count();
        for (name, f) in columns {
            assert_eq!(f(input), expected, "{name} miscounts {case}");
        }
    }
    let expected = rotating[0].encode_utf16().count();
    for (name, f) in columns {
        for buffer in &rotating {
            assert_eq!(f(buffer), expected, "{name} miscounts a rotating buffer");
        }
    }

    writeln!(out, "\n### Streaming\n").unwrap();
    writeln!(
        out,
        "One 64 MiB buffer per distribution and a 16 x 4 MiB rotating working set (64 MiB); median ns per call over {ROUNDS} rounds, and the effective rate in GB/s (bytes per ns).\n",
    )
    .unwrap();
    write!(out, "| Case | Bytes |").unwrap();
    for (name, _) in columns {
        write!(out, " {name} ns/call | {name} GB/s |").unwrap();
    }
    out.push('\n');
    write!(out, "|:-----|------:|").unwrap();
    for _ in columns {
        write!(out, "------:|------:|").unwrap();
    }
    out.push('\n');

    let mut row = |out: &mut String, case: &str, bytes: usize, samples: &mut [Vec<f64>]| {
        write!(out, "| {case} | {bytes} |").unwrap();
        for values in samples {
            values.sort_by(f64::total_cmp);
            let median = values[values.len() / 2];
            write!(out, " {median:.1} | {:.2} |", bytes as f64 / median).unwrap();
        }
        out.push('\n');
    };

    for (case, input) in &fixed {
        let iters: Vec<u64> = columns.iter().map(|(_, f)| calibrate(f, input)).collect();
        let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(ROUNDS); columns.len()];
        for _ in 0..ROUNDS {
            for (i, (_, f)) in columns.iter().enumerate() {
                let elapsed = time_batch(f, input, iters[i]);
                samples[i].push(elapsed.as_secs_f64() * 1e9 / iters[i] as f64);
            }
        }
        row(out, case, input.len(), &mut samples);
    }

    let iters: Vec<u64> = columns
        .iter()
        .map(|(_, f)| calibrate_rotating(f, &rotating))
        .collect();
    let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(ROUNDS); columns.len()];
    for _ in 0..ROUNDS {
        for (i, (_, f)) in columns.iter().enumerate() {
            let elapsed = time_rotating(f, &rotating, iters[i]);
            samples[i].push(elapsed.as_secs_f64() * 1e9 / iters[i] as f64);
        }
    }
    row(out, "rotating_16x4mib", rotating[0].len(), &mut samples);
}

/// One batch cycling through `buffers`, so each call sees a different one.
#[inline(never)]
fn time_rotating(f: &fn(&str) -> usize, buffers: &[String], iters: u64) -> Duration {
    let start = Instant::now();
    let mut i = 0;
    for _ in 0..iters {
        black_box(f(black_box(&buffers[i])));
        i = (i + 1) % buffers.len();
    }
    start.elapsed()
}

/// The iteration count that makes one rotating batch take at least `BATCH`.
fn calibrate_rotating(f: &fn(&str) -> usize, buffers: &[String]) -> u64 {
    let mut iters = 1;
    while time_rotating(f, buffers, iters) < BATCH {
        iters *= 2;
    }
    iters
}

/// The iteration count that makes one batch take at least `BATCH`.
fn calibrate(f: &fn(&str) -> usize, input: &str) -> u64 {
    let mut iters = 1;
    while time_batch(f, input, iters) < BATCH {
        iters *= 2;
    }
    iters
}

#[inline(never)]
fn time_batch(f: &fn(&str) -> usize, input: &str, iters: u64) -> Duration {
    let start = Instant::now();
    for _ in 0..iters {
        black_box(f(black_box(input)));
    }
    start.elapsed()
}

fn cpu_model() -> Option<String> {
    #[cfg(target_os = "linux")]
    {
        let field = |text: String, key: &str| {
            text.lines().find_map(|line| {
                let (name, value) = line.split_once(':')?;
                (name.trim() == key).then(|| value.trim().to_owned())
            })
        };
        std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|info| field(info, "model name"))
            .or_else(|| command_output("lscpu", &[]).and_then(|info| field(info, "Model name")))
    }
    #[cfg(target_os = "macos")]
    {
        command_output("sysctl", &["-n", "machdep.cpu.brand_string"])
    }
    #[cfg(target_os = "windows")]
    {
        command_output(
            "powershell",
            &[
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor).Name",
            ],
        )
        .or_else(|| std::env::var("PROCESSOR_IDENTIFIER").ok())
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        None
    }
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    let text = String::from_utf8(output.stdout).ok()?;
    let text = text.trim();
    (output.status.success() && !text.is_empty()).then(|| text.to_owned())
}
