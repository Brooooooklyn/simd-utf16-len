//! Benchmark inputs shared by `benches/utf16_len.rs`, `benches/kernel.rs`,
//! `perf/ab`, and `perf/kernels`, grouped by the code path they exercise.
//!
//! Inputs are leaked `&'static str`: benches and harness children build them
//! once, and the alignment group hands out slices of one padded buffer, which
//! an owned `String` per input cannot express.

// Each includer uses a different subset.
#![allow(dead_code)]

pub const ASCII: &str = "The quick brown fox jumps over the lazy dog. This is a longer sentence to provide more data for benchmarking purposes, with various words and punctuation marks included.";

pub const CJK: &str = "这是一段中文测试文本，用于测试UTF-8编码中多字节字符的处理性能。日本語のテキストも含まれています。한국어 텍스트도 포함되어 있습니다。";

pub const EMOJI: &str = "Hello 🌍🌎🌏! Flags: 🇺🇸🇬🇧🇯🇵🇨🇳 Family: 👨\u{200d}👩\u{200d}👧\u{200d}👦 Skin: 👋🏻👋🏼👋🏽👋🏾👋🏿 Fun: 🎉🎊🎈🎁🎄🎃";

pub const MIXED: &str = "Hello, 世界! 🌍 Привет мир! こんにちは世界！Héllo wörld! 你好世界！안녕하세요 세계! مرحبا بالعالم";

/// Every input of the original repository's benchmarks, from 144 bytes to
/// about 11 KB: nothing shorter. CodSpeed and the README table keep to this
/// set, so its benchmark identities stay comparable across commits.
pub fn all() -> Vec<(&'static str, &'static str)> {
    let long_ascii = ASCII.repeat(64);
    vec![
        ("ascii", ASCII.to_owned()),
        ("cjk", CJK.to_owned()),
        ("emoji", EMOJI.to_owned()),
        ("mixed", MIXED.to_owned()),
        // Leave 3 and 15 bytes after the last full 16-byte vector.
        ("cjk_tail3", "中".repeat(65)),
        ("cjk_tail15", "中".repeat(69)),
        // Longer than one 1,920-byte batch of the counting loop.
        ("ascii_large", long_ascii.clone()),
        ("cjk_large", CJK.repeat(64)),
        ("emoji_large", EMOJI.repeat(64)),
        ("mixed_large", MIXED.repeat(64)),
        // Mostly ASCII, like source code with one non-ASCII character.
        ("early_non_ascii", format!("é{long_ascii}")),
        ("late_non_ascii", format!("{long_ascii}é")),
    ]
    .into_iter()
    .map(|(name, input)| (name, leak(input)))
    .collect()
}

/// Inputs shorter than a 64-byte block, the lengths of identifiers and short
/// literals. The Perf A/B harness measures them on top of `all`, as in the
/// original repository.
pub fn short() -> Vec<(&'static str, &'static str)> {
    vec![
        // Shorter than one 16-byte vector.
        ("ascii_tiny", "hello world".to_owned()),
        ("utf8_tiny", "héllo wörld".to_owned()),
        // Shorter than one 64-byte ASCII block.
        (
            "ascii_short",
            "The quick brown fox jumps over the lazy dog.".to_owned(),
        ),
        ("utf8_short", "Привет, мир! Как дела сегодня?".to_owned()),
    ]
    .into_iter()
    .map(|(name, input)| (name, leak(input)))
    .collect()
}

/// Every length from 0 through 64 bytes, ASCII and mixed multibyte: dispatch,
/// ASCII-prefix, and short-path costs at single-byte resolution. Lengths
/// under 2 bytes cannot hold a multibyte character, so the smallest "mixed"
/// inputs are ASCII.
pub fn tiny() -> Vec<(&'static str, &'static str)> {
    let mut inputs = Vec::new();
    for len in 0..=64 {
        inputs.push((name(format_args!("tiny_ascii_{len:02}")), leak("a".repeat(len))));
        inputs.push((name(format_args!("tiny_mixed_{len:02}")), leak(mixed_of_len(len))));
    }
    inputs
}

/// Lengths straddling the 8/16/32/64/128/256-byte block and tail steps plus
/// the NEON kernel's structural lengths: 1,920 bytes is one batch of
/// MAX_BATCH=30 chunks of four vectors between widening flushes of the byte
/// lanes, and 16,384 the 16 KiB medium ceiling. ASCII exercises loads and
/// tails; the mixed copies force the counting path.
pub fn boundaries() -> Vec<(&'static str, &'static str)> {
    const LENGTHS: &[usize] = &[
        7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1919, 1920,
        1921, 3839, 3840, 3841, 7679, 7680, 7681, 16383, 16384, 16385,
    ];
    let mut inputs = Vec::new();
    for &len in LENGTHS {
        inputs.push((name(format_args!("bnd_ascii_{len:05}")), leak("a".repeat(len))));
        inputs.push((name(format_args!("bnd_mixed_{len:05}")), leak(mixed_of_len(len))));
    }
    inputs
}

/// Common source chunks, 256 B through 16 KiB, where the ASCII prefix scan
/// and the counting kernel cross over.
pub fn medium() -> Vec<(&'static str, &'static str)> {
    let mut inputs = Vec::new();
    for &len in &[256, 1024, 4096, 16384] {
        inputs.push((name(format_args!("med_ascii_{len:05}")), leak("a".repeat(len))));
        inputs.push((name(format_args!("med_mixed_{len:05}")), leak(mixed_of_len(len))));
    }
    inputs
}

/// Long-loop behavior at 64 KiB, 256 KiB, and 1 MiB, ASCII and CJK. Nothing
/// larger: the streaming cases live in `perf/kernels`, which the codspeed
/// criterion benches never build.
pub fn large() -> Vec<(&'static str, &'static str)> {
    let mut inputs = Vec::new();
    for &(label, len) in &[("64k", 64 * 1024), ("256k", 256 * 1024), ("1m", 1024 * 1024)] {
        inputs.push((name(format_args!("lg_ascii_{label}")), leak("a".repeat(len))));
        inputs.push((
            name(format_args!("lg_cjk_{label}")),
            leak(pad_to_len("中".repeat(len / 3), len)),
        ));
    }
    inputs
}

/// One 4 KiB input per classifier distribution: pure ASCII, 2-byte Latin,
/// 3-byte CJK, 4-byte emoji, an even mix, and a single non-ASCII character
/// at the start, middle, or end of otherwise-ASCII text.
pub fn distributions() -> Vec<(&'static str, &'static str)> {
    const LEN: usize = 4096;
    let half = "a".repeat(LEN / 2 - 1);
    vec![
        ("dist_ascii", "a".repeat(LEN)),
        ("dist_latin2", "é".repeat(LEN / 2)),
        ("dist_cjk", pad_to_len("中".repeat(LEN / 3), LEN)),
        ("dist_emoji", "🙂".repeat(LEN / 4)),
        ("dist_mixed", mixed_of_len(LEN)),
        ("dist_sparse_start", format!("é{}", "a".repeat(LEN - 2))),
        ("dist_sparse_middle", format!("{half}中{}", "a".repeat(LEN - half.len() - 3))),
        ("dist_sparse_end", format!("{}é", "a".repeat(LEN - 2))),
    ]
    .into_iter()
    .map(|(name, input)| (name, leak(input)))
    .collect()
}

/// One 4 KiB ASCII window at every byte offset 0..=63 of a padded buffer, so
/// the measured pointer sweeps every start alignment. ASCII keeps every
/// offset on a char boundary; the padding leaves the content identical, so
/// only the start alignment and tail position change.
pub fn alignment() -> Vec<(&'static str, &'static str)> {
    const LEN: usize = 4096;
    let padded = leak(format!("{}{}", " ".repeat(64), ASCII.repeat(25)));
    (0..64)
        .map(|offset| {
            (
                name(format_args!("align_ascii_off{offset:02}")),
                &padded[offset..offset + LEN],
            )
        })
        .collect()
}

/// Deterministic shuffled identifier/keyword/punctuation fragments, like
/// source text, with a few non-ASCII identifiers and comments mixed in. A
/// fixed-seed xorshift makes the corpus identical on every run and machine;
/// nothing is fetched or read from disk.
pub fn realistic() -> Vec<(&'static str, &'static str)> {
    const FRAGMENTS: &[&str] = &[
        "fn ", "let ", "const ", "pub ", "impl ", "struct ", "return ", "if ", "else ", "for ",
        "while ", "use ", "mod ", "match ", "utf16_len", "ascii_prefix_len", "bytes", "start",
        "index", "count", "buffer", "value", "result", "chunk_size", "(", ")", "{", "}", ";\n",
        "::", " => ", ", ", " = ", ".", " ", "// 边界检查\n", "café", "naïve", "世界", "🙂",
        "\"hello world\"", "'a'", "0x1F600", "u8", "usize", "&str",
    ];
    let mut inputs = Vec::new();
    for &(label, len, seed) in &[
        ("256", 256, 0x9E3779B97F4A7C15u64),
        ("4k", 4096, 0xD1B54A32D192ED03),
        ("64k", 64 * 1024, 0x94D049BB133111EB),
    ] {
        let mut rng = Rng(seed);
        let mut corpus = String::with_capacity(len + 16);
        while corpus.len() < len {
            corpus.push_str(rng.pick(FRAGMENTS));
        }
        // The last fragment can overshoot; drop it at its char boundary
        // and re-pad to the exact length.
        let mut end = len;
        while !corpus.is_char_boundary(end) {
            end -= 1;
        }
        corpus.truncate(end);
        inputs.push((
            name(format_args!("realistic_{label}")),
            leak(pad_to_len(corpus, len)),
        ));
    }
    inputs
}

/// Exactly `len` bytes of valid mixed UTF-8, starting with a 2-byte
/// character where one fits: cycles 2-, 3-, and 4-byte characters with
/// ASCII and pads the remainder with 'a'.
fn mixed_of_len(len: usize) -> String {
    const FRAGMENTS: [&str; 4] = ["é", "中", "🙂", "a"];
    let mut s = String::with_capacity(len + 4);
    let mut i = 0;
    while s.len() + FRAGMENTS[i % 4].len() <= len {
        s.push_str(FRAGMENTS[i % 4]);
        i += 1;
    }
    pad_to_len(s, len)
}

/// Pads with 'a' up to `len` bytes; `s` must be at most `len` bytes long.
fn pad_to_len(mut s: String, len: usize) -> String {
    debug_assert!(s.len() <= len);
    while s.len() < len {
        s.push('a');
    }
    s
}

fn leak(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

fn name(args: std::fmt::Arguments) -> &'static str {
    leak(format!("{args}"))
}

/// Fixed-seed xorshift64*, so the realistic corpus never changes between
/// runs, machines, or harnesses.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn pick<'a>(&mut self, fragments: &[&'a str]) -> &'a str {
        fragments[(self.next() >> 32) as usize % fragments.len()]
    }
}
