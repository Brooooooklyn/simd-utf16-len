//! NEON-based UTF-16 length calculation (always available on aarch64).
//!
//! Shaped like napi-rs/json-escape-simd's kernels: a pointer cursor with a
//! remaining-byte count, four unrolled vectors per iteration counted into
//! four byte-lane accumulators, leftover vectors and an in-register tail
//! folded into the same accumulators, and one horizontal sum. Inputs of
//! fewer than four vectors skip the accumulators and count into one. Each
//! byte's units come from a `tbl` lookup of its high nibble, as in
//! json-escape-simd's nibble-table classifier; NEON shifts bytes directly,
//! so the index needs no masking.
//!
//! Two FEAT_DotProd variants follow the same shape with their own batch
//! bounds: `neon-udot` replaces the byte-lane adds with `udot` against ones
//! into four u32x4 accumulators, and `neon-cmp-sdot` is the compare-based
//! measurement control with two `sdot` chains. Both detect the feature
//! once, fall back to the baseline kernel without it, and are selected only
//! by this crate's tests and benches: `utf16_len` still runs the baseline.

use std::arch::aarch64::*;
use std::sync::atomic::{AtomicU8, Ordering};

/// Bytes per vector, and per unrolled iteration of four vectors.
const LANES: usize = 16;
const CHUNK: usize = LANES * 4;

/// The units of each byte of `$v`, by its high nibble through the `tbl`
/// table `$table` (UNITS_BY_HIGH_NIBBLE).
macro_rules! units {
    ($table:expr, $v:expr) => {
        vqtbl1q_u8($table, vshrq_n_u8::<4>($v))
    };
}

/// The units of the last `$nb` bytes of the input, from the vector that
/// ends at `$sptr + $nb`, which holds at least a vector: only its last
/// `$nb` lanes are uncounted, and byte-wise counting needs no UTF-8
/// boundary care.
macro_rules! tail_units {
    ($table:expr, $sptr:expr, $nb:expr) => {{
        let v = vld1q_u8(($sptr).add($nb).sub(LANES));
        vandq_u8(units!($table, v), vld1q_u8(crate::keep_last(LANES, $nb)))
    }};
}

/// Iterations of four vectors between two sums of the byte-lane
/// accumulators. A lane gains at most 2 per vector, so the merged
/// accumulators hold at most 4 * 2 * 30 = 240 after a batch, which leaves
/// room for 3 leftover vectors and the tail: 240 + 3 * 2 + 2 = 248 < 256.
const MAX_BATCH: usize = 30;

/// Compute the number of UTF-16 code units for UTF-8 string using NEON.
#[inline]
pub fn utf16_len(s: &str) -> usize {
    let bytes = s.as_bytes();
    let start = crate::ascii::ascii_prefix_len(bytes);
    if start == bytes.len() {
        start
    } else {
        // SAFETY: bytes comes from a valid str, and start is a verified ASCII
        // prefix shorter than it.
        unsafe { utf16_len_neon(bytes, start) }
    }
}

/// One batch's merged byte lanes, widened pairwise into the running 64-bit
/// total. Each u64 lane counts only its own eight byte lanes' units, so it
/// stays below the input's length and cannot wrap for any input a `&str`
/// can hold. Factored out so the regression tests drive the production
/// accumulation, not a copy of it.
///
/// # Safety
/// NEON is baseline on aarch64.
#[inline(always)]
unsafe fn widen_into_total(total: uint64x2_t, merged: uint8x16_t) -> uint64x2_t {
    unsafe { vpadalq_u32(total, vpaddlq_u16(vpaddlq_u8(merged))) }
}

/// The full count: the ASCII prefix, the residual merged byte lanes, and
/// the batch total, reduced in 64 bits. Factored out with `widen_into_total`
/// so the regression tests drive the production reduction, not a copy of it.
///
/// # Safety
/// NEON is baseline on aarch64.
#[inline(always)]
unsafe fn combine(start: usize, merged: uint8x16_t, total: uint64x2_t) -> usize {
    unsafe { start + vaddlvq_u8(merged) as usize + vaddvq_u64(total) as usize }
}

/// A function of its own, but always inlined: a caller that inlines
/// `utf16_len` gets the whole count without a call, and one the inliner
/// turns down calls `utf16_len` as before.
///
/// # Safety
/// `bytes` must be valid UTF-8 with an ASCII prefix of `start` bytes, and
/// `start < bytes.len()`.
#[inline(always)]
unsafe fn utf16_len_neon(bytes: &[u8], start: usize) -> usize {
    // SAFETY: NEON is baseline on aarch64. Each full-vector load stays within
    // the input, the mask table, or the short input's halves.
    unsafe {
        let len = bytes.len();
        let mut sptr = bytes.as_ptr().add(start);
        let mut nb = len - start;

        let zero = vdupq_n_u8(0);
        let table = vld1q_u8(crate::UNITS_BY_HIGH_NIBBLE.as_ptr());

        if len < LANES {
            // The whole input is shorter than a vector.
            let (v, keep) = short_vector!(sptr, nb, vld1q_u8);
            return start + vaddlvq_u8(vandq_u8(units!(table, v), keep)) as usize;
        }

        if nb < CHUNK {
            // Fewer than four vectors: one accumulator, one sum.
            let mut acc = zero;
            while nb >= LANES {
                acc = vaddq_u8(acc, units!(table, vld1q_u8(sptr)));
                sptr = sptr.add(LANES);
                nb -= LANES;
            }
            if nb > 0 {
                acc = vaddq_u8(acc, tail_units!(table, sptr, nb));
            }
            return start + vaddlvq_u8(acc) as usize;
        }

        macro_rules! merge {
            ($a0:expr, $a1:expr, $a2:expr, $a3:expr) => {
                vaddq_u8(vaddq_u8($a0, $a1), vaddq_u8($a2, $a3))
            };
        }

        // Sums of full batches, as two u64 lanes without a horizontal add:
        // u32 lanes and a u32 reduction returned the count modulo 2^32 for
        // inputs past 4,294,968,959 units.
        let mut total = vdupq_n_u64(0);
        let (mut a0, mut a1, mut a2, mut a3) = (zero, zero, zero, zero);
        macro_rules! chunk {
            () => {
                a0 = vaddq_u8(a0, units!(table, vld1q_u8(sptr)));
                a1 = vaddq_u8(a1, units!(table, vld1q_u8(sptr.add(LANES))));
                a2 = vaddq_u8(a2, units!(table, vld1q_u8(sptr.add(LANES * 2))));
                a3 = vaddq_u8(a3, units!(table, vld1q_u8(sptr.add(LANES * 3))));
                sptr = sptr.add(CHUNK);
            };
        }

        while nb >= CHUNK * MAX_BATCH {
            for _ in 0..MAX_BATCH {
                chunk!();
            }
            nb -= CHUNK * MAX_BATCH;
            // More follows: widen now so the byte lanes can't overflow.
            total = widen_into_total(total, merge!(a0, a1, a2, a3));
            (a0, a1, a2, a3) = (zero, zero, zero, zero);
        }
        // Fewer than MAX_BATCH iterations remain.
        while nb >= CHUNK {
            chunk!();
            nb -= CHUNK;
        }
        while nb >= LANES {
            a0 = vaddq_u8(a0, units!(table, vld1q_u8(sptr)));
            sptr = sptr.add(LANES);
            nb -= LANES;
        }
        if nb > 0 {
            a1 = vaddq_u8(a1, tail_units!(table, sptr, nb));
        }

        combine(start, merge!(a0, a1, a2, a3), total)
    }
}

/// FEAT_DotProd support, cached: 0 unknown, 1 present, 2 absent. The sysctl
/// behind the detection macro costs too much to run on every call. Never
/// detected under Miri: it cannot interpret the dot-product intrinsics
/// (llvm.aarch64.neon.udot/sdot are unimplemented), so the kernels behind
/// this check must stay unreachable there.
#[inline]
pub(crate) fn dotprod() -> bool {
    if cfg!(miri) {
        return false;
    }
    match HAS_DOTPROD.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            let present = std::arch::is_aarch64_feature_detected!("dotprod");
            HAS_DOTPROD.store(if present { 1 } else { 2 }, Ordering::Relaxed);
            present
        }
    }
}

static HAS_DOTPROD: AtomicU8 = AtomicU8::new(0);

/// Iterations of four vectors between two flushes of the udot
/// accumulators: 16,384 iterations x 64 bytes = 1 MiB per batch. Each UDOT
/// adds at most 4 x 2 = 8 to one u32 lane, so a lane of one accumulator
/// reaches at most 16,384 x 8 = 131,072 per batch, the four merged
/// accumulators 524,288 per lane, and their widening horizontal sum
/// 2,097,152 (the review's Section 8.2). The residual loops after the last
/// full batch add at most 16,383 chunk iterations, 3 leftover vectors, and
/// the tail to the same accumulators, 16,383 x 8 + 4 x 8 = 131,096 per
/// lane, far below u32::MAX. The u64 total counts at most 2 units per
/// input byte and cannot wrap for any input a `&str` can hold.
const MAX_UDOT_BATCH: usize = 16384;

/// The baseline's high-nibble `tbl` classifier, counted by `udot` against
/// ones into four u32x4 accumulators instead of byte-lane adds. Falls back
/// to the baseline kernel without FEAT_DotProd, so the function pointer
/// stays safe to call on any aarch64 CPU.
pub fn utf16_len_udot(s: &str) -> usize {
    let bytes = s.as_bytes();
    let start = crate::ascii::ascii_prefix_len(bytes);
    if start == bytes.len() {
        start
    } else if dotprod() {
        // SAFETY: bytes comes from a valid str, start is a verified ASCII
        // prefix shorter than it, and FEAT_DotProd was detected.
        unsafe { utf16_len_neon_udot(bytes, start) }
    } else {
        // SAFETY: as above; NEON is baseline on aarch64.
        unsafe { utf16_len_neon(bytes, start) }
    }
}

/// The full count, as `utf16_len_neon` but with UDOT accumulation: the
/// units vectors feed `vdotq_u32` against ones, so the keep-masked tail
/// and the zero-padded short vector contribute exactly their masked lanes.
///
/// # Safety
/// `bytes` must be valid UTF-8 with an ASCII prefix of `start` bytes, and
/// `start < bytes.len()`. Requires FEAT_DotProd.
#[target_feature(enable = "dotprod")]
unsafe fn utf16_len_neon_udot(bytes: &[u8], start: usize) -> usize {
    // SAFETY: FEAT_DotProd is the caller's contract. Each full-vector load
    // stays within the input, the mask table, or the short input's halves.
    unsafe {
        let len = bytes.len();
        let mut sptr = bytes.as_ptr().add(start);
        let mut nb = len - start;

        let table = vld1q_u8(crate::UNITS_BY_HIGH_NIBBLE.as_ptr());
        let ones = vdupq_n_u8(1);
        let zero = vdupq_n_u32(0);

        if len < LANES {
            // The whole input is shorter than a vector. The zero padding
            // classifies as 1 unit per lane, so the keep mask, not the
            // LUT, is what excludes it.
            let (v, keep) = short_vector!(sptr, nb, vld1q_u8);
            let counted = vdotq_u32(zero, vandq_u8(units!(table, v), keep), ones);
            return start + vaddlvq_u32(counted) as usize;
        }

        if nb < CHUNK {
            // Fewer than four vectors: one accumulator, one sum.
            let mut acc = zero;
            while nb >= LANES {
                acc = vdotq_u32(acc, units!(table, vld1q_u8(sptr)), ones);
                sptr = sptr.add(LANES);
                nb -= LANES;
            }
            if nb > 0 {
                acc = vdotq_u32(acc, tail_units!(table, sptr, nb), ones);
            }
            return start + vaddlvq_u32(acc) as usize;
        }

        macro_rules! merge {
            ($a0:expr, $a1:expr, $a2:expr, $a3:expr) => {
                vaddq_u32(vaddq_u32($a0, $a1), vaddq_u32($a2, $a3))
            };
        }

        let mut total = 0u64;
        let (mut a0, mut a1, mut a2, mut a3) = (zero, zero, zero, zero);
        macro_rules! chunk {
            () => {
                a0 = vdotq_u32(a0, units!(table, vld1q_u8(sptr)), ones);
                a1 = vdotq_u32(a1, units!(table, vld1q_u8(sptr.add(LANES))), ones);
                a2 = vdotq_u32(a2, units!(table, vld1q_u8(sptr.add(LANES * 2))), ones);
                a3 = vdotq_u32(a3, units!(table, vld1q_u8(sptr.add(LANES * 3))), ones);
                sptr = sptr.add(CHUNK);
            };
        }

        while nb >= CHUNK * MAX_UDOT_BATCH {
            for _ in 0..MAX_UDOT_BATCH {
                chunk!();
            }
            nb -= CHUNK * MAX_UDOT_BATCH;
            // More follows: flush now, while the merged lanes are bounded.
            total += vaddlvq_u32(merge!(a0, a1, a2, a3));
            (a0, a1, a2, a3) = (zero, zero, zero, zero);
        }
        // Fewer than MAX_UDOT_BATCH iterations remain: the same bound.
        while nb >= CHUNK {
            chunk!();
            nb -= CHUNK;
        }
        while nb >= LANES {
            a0 = vdotq_u32(a0, units!(table, vld1q_u8(sptr)), ones);
            sptr = sptr.add(LANES);
            nb -= LANES;
        }
        if nb > 0 {
            a1 = vdotq_u32(a1, tail_units!(table, sptr, nb), ones);
        }

        start + (total + vaddlvq_u32(merge!(a0, a1, a2, a3))) as usize
    }
}

/// Compare-based measurement control for `utf16_len_udot`: every byte that
/// is not a continuation (signed byte > -65) contributes one unit and
/// every four-byte leader (unsigned byte >= 0xF0) one more, counted by two
/// `sdot` chains against -1 (compare masks are -1 per lane). Four SIMD ops
/// per vector instead of the lookup's three. Same fallback as
/// `utf16_len_udot`.
pub fn utf16_len_cmp_sdot(s: &str) -> usize {
    let bytes = s.as_bytes();
    let start = crate::ascii::ascii_prefix_len(bytes);
    if start == bytes.len() {
        start
    } else if dotprod() {
        // SAFETY: bytes comes from a valid str, start is a verified ASCII
        // prefix shorter than it, and FEAT_DotProd was detected.
        unsafe { utf16_len_neon_cmp_sdot(bytes, start) }
    } else {
        // SAFETY: as above; NEON is baseline on aarch64.
        unsafe { utf16_len_neon(bytes, start) }
    }
}

/// The full count by compare masks and signed dot products. Its i32
/// counter lanes gain at most 4 per vector per counter, 16 per four-vector
/// iteration: at 16,384 iterations that is 262,144 per lane, leaders and
/// fours merged 524,288, and the residual loops add at most 16,383 x 16 +
/// 4 x 4 = 262,144 more, all far within i32 (the review's Section 9).
/// `vaddlvq_s32` widens the merged sum, at most 2,097,152 and never
/// negative, into the u64 total.
///
/// # Safety
/// `bytes` must be valid UTF-8 with an ASCII prefix of `start` bytes, and
/// `start < bytes.len()`. Requires FEAT_DotProd.
#[target_feature(enable = "dotprod")]
unsafe fn utf16_len_neon_cmp_sdot(bytes: &[u8], start: usize) -> usize {
    // SAFETY: FEAT_DotProd is the caller's contract. Each full-vector load
    // stays within the input, the mask table, or the short input's halves.
    unsafe {
        let len = bytes.len();
        let mut sptr = bytes.as_ptr().add(start);
        let mut nb = len - start;

        let neg_ones = vdupq_n_s8(-1);
        let cont_max = vdupq_n_s8(-65);
        let four_min = vdupq_n_u8(0xf0);

        // -1 in the lanes of `$v` that are not continuation bytes, and in
        // those that are four-byte leaders, respectively.
        macro_rules! leaders {
            ($v:expr) => {
                vreinterpretq_s8_u8(vcgtq_s8(vreinterpretq_s8_u8($v), cont_max))
            };
        }
        macro_rules! fours {
            ($v:expr) => {
                vreinterpretq_s8_u8(vcgeq_u8($v, four_min))
            };
        }
        // Both masks of the input's last `$nb` bytes, from the vector that
        // ends at `$sptr + $nb`, with the already-counted lanes masked off:
        // a masked lane contributes 0 to the dot products.
        macro_rules! tail_masks {
            ($sptr:expr, $nb:expr) => {{
                let v = vld1q_u8(($sptr).add($nb).sub(LANES));
                let keep = vld1q_u8(crate::keep_last(LANES, $nb));
                (
                    vreinterpretq_s8_u8(vandq_u8(vreinterpretq_u8_s8(leaders!(v)), keep)),
                    vreinterpretq_s8_u8(vandq_u8(vreinterpretq_u8_s8(fours!(v)), keep)),
                )
            }};
        }

        if len < LANES {
            // The whole input is shorter than a vector. A zero byte counts
            // as a non-continuation here, so the keep mask, not the
            // compares, excludes the zero padding.
            let (v, keep) = short_vector!(sptr, nb, vld1q_u8);
            let leader = vreinterpretq_s8_u8(vandq_u8(vreinterpretq_u8_s8(leaders!(v)), keep));
            let four = vreinterpretq_s8_u8(vandq_u8(vreinterpretq_u8_s8(fours!(v)), keep));
            let zero = vdupq_n_s32(0);
            let counts = vaddq_s32(
                vdotq_s32(zero, leader, neg_ones),
                vdotq_s32(zero, four, neg_ones),
            );
            return start + vaddlvq_s32(counts) as usize;
        }

        if nb < CHUNK {
            // Fewer than four vectors: one counter pair, one sum.
            let mut leaders_acc = vdupq_n_s32(0);
            let mut fours_acc = vdupq_n_s32(0);
            while nb >= LANES {
                let v = vld1q_u8(sptr);
                leaders_acc = vdotq_s32(leaders_acc, leaders!(v), neg_ones);
                fours_acc = vdotq_s32(fours_acc, fours!(v), neg_ones);
                sptr = sptr.add(LANES);
                nb -= LANES;
            }
            if nb > 0 {
                let (leader, four) = tail_masks!(sptr, nb);
                leaders_acc = vdotq_s32(leaders_acc, leader, neg_ones);
                fours_acc = vdotq_s32(fours_acc, four, neg_ones);
            }
            return start + vaddlvq_s32(vaddq_s32(leaders_acc, fours_acc)) as usize;
        }

        let mut total = 0u64;
        let mut leaders_acc = vdupq_n_s32(0);
        let mut fours_acc = vdupq_n_s32(0);
        macro_rules! count_vec {
            ($v:expr) => {{
                let v = $v;
                leaders_acc = vdotq_s32(leaders_acc, leaders!(v), neg_ones);
                fours_acc = vdotq_s32(fours_acc, fours!(v), neg_ones);
            }};
        }
        macro_rules! chunk {
            () => {
                count_vec!(vld1q_u8(sptr));
                count_vec!(vld1q_u8(sptr.add(LANES)));
                count_vec!(vld1q_u8(sptr.add(LANES * 2)));
                count_vec!(vld1q_u8(sptr.add(LANES * 3)));
                sptr = sptr.add(CHUNK);
            };
        }

        while nb >= CHUNK * MAX_UDOT_BATCH {
            for _ in 0..MAX_UDOT_BATCH {
                chunk!();
            }
            nb -= CHUNK * MAX_UDOT_BATCH;
            // More follows: flush now, while the counters are bounded.
            total += vaddlvq_s32(vaddq_s32(leaders_acc, fours_acc)) as u64;
            leaders_acc = vdupq_n_s32(0);
            fours_acc = vdupq_n_s32(0);
        }
        // Fewer than MAX_UDOT_BATCH iterations remain: the same bound.
        while nb >= CHUNK {
            chunk!();
            nb -= CHUNK;
        }
        while nb >= LANES {
            count_vec!(vld1q_u8(sptr));
            sptr = sptr.add(LANES);
            nb -= LANES;
        }
        if nb > 0 {
            let (leader, four) = tail_masks!(sptr, nb);
            leaders_acc = vdotq_s32(leaders_acc, leader, neg_ones);
            fours_acc = vdotq_s32(fours_acc, four, neg_ones);
        }

        start + (total + vaddlvq_s32(vaddq_s32(leaders_acc, fours_acc)) as u64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The merged byte lanes of one full batch of the review's Section-4.2
    /// counterexample (a leading U+00E9, then ASCII): every lane of a real
    /// ASCII batch holds 30 * 4 = 120 units, and the first batch's first
    /// lane loses one unit to the two-byte leader, so the pairwise u32 sums
    /// are [479, 480, 480, 480] for the first batch and [480, 480, 480, 480]
    /// for every later one, as the exact arithmetic model computes.
    const FIRST_BATCH: [u8; 16] = {
        let mut lanes = [120; 16];
        lanes[0] = 119;
        lanes
    };
    const ASCII_BATCH: [u8; 16] = [120; 16];

    fn merged(lanes: &[u8; 16]) -> uint8x16_t {
        // SAFETY: the array holds 16 readable bytes, and NEON is baseline.
        unsafe { vld1q_u8(lanes.as_ptr()) }
    }

    fn widen(total: uint64x2_t, lanes: &[u8; 16]) -> uint64x2_t {
        // SAFETY: NEON is baseline on aarch64.
        unsafe { widen_into_total(total, merged(lanes)) }
    }

    /// `FIRST_BATCH` once and `ASCII_BATCH` `batches - 1` times, counted
    /// through the production widening fold and final combine.
    fn counted(batches: usize) -> usize {
        // SAFETY: NEON is baseline on aarch64.
        let mut total = widen(unsafe { vdupq_n_u64(0) }, &FIRST_BATCH);
        for _ in 1..batches {
            total = widen(total, &ASCII_BATCH);
        }
        // SAFETY: NEON is baseline on aarch64.
        unsafe { combine(0, merged(&[0; 16]), total) }
    }

    /// The Section-4.2 counterexample without its 4 GiB allocation: the
    /// exact total exceeds u32::MAX, so a 32-bit final reduction of these
    /// batches returned 1,663.
    #[test]
    fn batch_total_survives_a_wrapping_u32_sum() {
        assert_eq!(counted(2_236_963), 4_294_968_959);
    }

    /// The model's second case: 8,947,849 batches push every former u32
    /// lane itself over u32::MAX, so even widening only the final reduction
    /// of 32-bit lanes returned 895.
    #[test]
    fn batch_total_survives_wrapping_u32_lanes() {
        assert_eq!(counted(8_947_849), 17_179_870_079);
    }

    /// The final combine still adds the ASCII prefix and the residual
    /// merged byte lanes, which the huge totals above leave at zero.
    #[test]
    fn combine_adds_prefix_and_residual() {
        let mut residual = [0u8; 16];
        residual[0] = 7;
        residual[15] = 3;
        // SAFETY: NEON is baseline on aarch64.
        let total = widen(unsafe { vdupq_n_u64(0) }, &ASCII_BATCH);
        // SAFETY: NEON is baseline on aarch64.
        assert_eq!(
            unsafe { combine(5, merged(&residual), total) },
            5 + 10 + 1920
        );
    }

    /// The dot-product kernels' keep masks, not their classifiers, exclude
    /// lanes: a zero byte is one LUT unit and a compare "non-continuation",
    /// so the zero-padded lanes of the short vector and the already-counted
    /// lanes of the tail vector contribute only because the mask zeroed
    /// them first.
    #[test]
    fn masked_lanes_contribute_zero() {
        // SAFETY: NEON is baseline on aarch64, and the dot products run
        // only where FEAT_DotProd was detected.
        unsafe {
            let table = vld1q_u8(crate::UNITS_BY_HIGH_NIBBLE.as_ptr());
            let zero = vdupq_n_u8(0);
            // Sixteen zero bytes classify as one unit each.
            assert_eq!(vaddlvq_u8(units!(table, zero)) as u32, 16);
            for nb in 1..LANES {
                let first = vandq_u8(units!(table, zero), vld1q_u8(crate::keep_first(nb)));
                assert_eq!(vaddlvq_u8(first) as u32, nb as u32);
                let last = vandq_u8(units!(table, zero), vld1q_u8(crate::keep_last(LANES, nb)));
                assert_eq!(vaddlvq_u8(last) as u32, nb as u32);
            }
            if !dotprod() {
                return;
            }
            let ones = vdupq_n_u8(1);
            let neg_ones = vdupq_n_s8(-1);
            let cont_max = vdupq_n_s8(-65);
            for nb in 1..LANES {
                // The udot kernel counts the masked units exactly.
                let keep = vld1q_u8(crate::keep_first(nb));
                let counted = vdotq_u32(vdupq_n_u32(0), vandq_u8(units!(table, zero), keep), ones);
                assert_eq!(vaddlvq_u32(counted), nb as u64);
                let keep = vld1q_u8(crate::keep_last(LANES, nb));
                let counted = vdotq_u32(vdupq_n_u32(0), vandq_u8(units!(table, zero), keep), ones);
                assert_eq!(vaddlvq_u32(counted), nb as u64);
                // The cmp kernel's leader mask is all-ones on zero bytes
                // unmasked, and exactly the kept lanes once masked.
                let leader = vcgtq_s8(vreinterpretq_s8_u8(zero), cont_max);
                let unmasked = vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(leader), neg_ones);
                assert_eq!(vaddlvq_s32(unmasked), 16);
                let keep = vld1q_u8(crate::keep_last(LANES, nb));
                let masked = vreinterpretq_s8_u8(vandq_u8(leader, keep));
                assert_eq!(
                    vaddlvq_s32(vdotq_s32(vdupq_n_s32(0), masked, neg_ones)),
                    nb as i64
                );
            }
        }
    }

    /// Section 8.2's bound against the shipped batch constant: per batch a
    /// u32 lane of one udot accumulator holds at most 8 x MAX_UDOT_BATCH,
    /// and the residual loops add 3 leftover vectors and a tail; the four
    /// merged accumulators must stay below u32::MAX.
    #[test]
    fn udot_batch_bound_holds() {
        let per_lane = 8 * MAX_UDOT_BATCH + 8 * 4;
        assert!(4 * per_lane < u32::MAX as usize);
        // Section 9's signed bound, same batch: 16 per iteration per
        // counter, plus 4 vectors of residual; leaders + fours merged must
        // stay below i32::MAX.
        let per_counter = 16 * MAX_UDOT_BATCH + 4 * 4;
        assert!(2 * per_counter < i32::MAX as usize);
    }
}
