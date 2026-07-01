//! Macros that generate fast byte-search functions from compile-time patterns.
//!
//! [`swar_search!`] scans a whole machine word (`usize`) at a time with SWAR
//! (SIMD-within-a-register) bit tricks; the word width follows the platform
//! (8 bytes on 64-bit, 4 on 32-bit). [`simd_search!`] does the same with explicit
//! SIMD vectors via the [`wide`] crate. Both accept identical syntax and return
//! identical results; pick SWAR for zero dependencies or SIMD for vector units.
//!
//! [`wide`]: https://docs.rs/wide
//!
//! # Example
//!
//! ```
//! use byte_search::swar_search;
//!
//! swar_search!(find_newlines, b'\n');   // single byte
//! swar_search!(find_sync, 0xAA, 0x55);  // two bytes
//!
//! assert_eq!(find_newlines(b"a\nbc\n"), vec![1, 4]);
//! assert_eq!(find_sync(&[0x00, 0xAA, 0x55, 0x01, 0xAA, 0x55]), vec![1, 4]);
//! ```

/// Byte lanes in a machine word (`usize`): 8 on 64-bit, 4 on 32-bit.
pub const WORD_BYTES: usize = ::core::mem::size_of::<usize>();

/// `0x01` in every byte lane of a `usize`.
pub const LANE_ONES: usize = usize::MAX / 0xFF;
/// The low seven bits of every byte lane set.
pub const LANE_LOW_7_BITS: usize = LANE_ONES * 0x7F;
/// The high bit of every byte lane set.
pub const LANE_HIGH_BITS: usize = LANE_ONES * 0x80;

/// Flag zero bytes in a `usize`: returns a mask with bit 7 set in each byte
/// lane that is zero (exact per lane, no cross-lane borrows).
///
/// XORing a word against a broadcast pattern zeroes the matching lanes, so this
/// is the core of the SWAR search. Lane count follows the pointer width.
#[inline]
pub const fn zero_byte_mask(value: usize) -> usize {
    let tmp = ((value & LANE_LOW_7_BITS).wrapping_add(LANE_LOW_7_BITS)) | value | LANE_LOW_7_BITS;
    !tmp & LANE_HIGH_BITS
}

/// Broadcast a byte into every lane of a `usize`.
#[inline]
pub const fn broadcast(byte: u8) -> usize {
    (byte as usize) * LANE_ONES
}

/// Bytes processed per SIMD iteration: one 128-bit vector.
pub const SIMD_LANES: usize = 16;

/// SIMD counterpart to [`zero_byte_mask`]: compares a 16-byte block against a
/// broadcast `pattern` and returns a bitmask with bit `i` set per matching lane.
///
/// `wide` lowers this to hardware SIMD (SSE2/AVX/NEON) where available and a
/// scalar fallback otherwise; it is kept internal so callers need not depend on
/// it.
#[inline]
pub fn simd_match_mask(block: [u8; SIMD_LANES], pattern: u8) -> u32 {
    ::wide::u8x16::new(block)
        .simd_eq(::wide::u8x16::splat(pattern))
        .to_bitmask()
}

/// Generate a `fn(&[u8]) -> Vec<usize>` that returns every index where the
/// pattern begins, using a SWAR (SIMD-within-a-register) scan.
///
/// Pattern bytes are evaluated in a `const` context, so they must be
/// compile-time constants. An optional visibility qualifier is accepted.
///
/// ```text
/// swar_search!(name, first);                 // single-byte pattern
/// swar_search!(name, first, second);         // two-byte pattern
/// swar_search!(pub(crate) name, first);      // with visibility
/// ```
///
/// See [`simd_search!`] for an equivalent backed by explicit SIMD.
///
/// # Examples
///
/// ```
/// use byte_search::swar_search;
///
/// swar_search!(pub find_zeros, 0x00);
/// assert_eq!(find_zeros(&[1, 0, 2, 0, 0]), vec![1, 3, 4]);
/// ```
#[macro_export]
macro_rules! swar_search {
    // Single-byte pattern.
    ($vis:vis $name:ident, $first:expr $(,)?) => {
        $vis fn $name(data: &[u8]) -> ::std::vec::Vec<usize> {
            const PATTERN: usize = $crate::broadcast($first);
            // 8 bytes on 64-bit, 4 on 32-bit.
            const WORD: usize = $crate::WORD_BYTES;

            let mut positions: ::std::vec::Vec<usize> = ::std::vec::Vec::new();
            let mut chunks = data.chunks_exact(WORD);

            for (chunk_index, chunk) in chunks.by_ref().enumerate() {
                let word = usize::from_le_bytes(
                    chunk
                        .try_into()
                        .expect("chunks_exact(WORD) always yields WORD-byte chunks"),
                );

                // Flag lanes equal to the pattern.
                let mut mask = $crate::zero_byte_mask(word ^ PATTERN);
                let base = chunk_index * WORD;

                while mask != 0 {
                    let byte_offset = (mask.trailing_zeros() as usize) / 8;
                    positions.push(base + byte_offset);
                    mask &= mask - 1; // clear lowest set bit
                }
            }

            // Sub-word tail.
            let remainder = chunks.remainder();
            let remainder_base = data.len() - remainder.len();
            const FIRST: u8 = $first;
            for (offset, &byte) in remainder.iter().enumerate() {
                if byte == FIRST {
                    positions.push(remainder_base + offset);
                }
            }

            positions
        }
    };

    // Two-byte pattern.
    ($vis:vis $name:ident, $first:expr, $second:expr $(,)?) => {
        $vis fn $name(data: &[u8]) -> ::std::vec::Vec<usize> {
            const PATTERN: usize = $crate::broadcast($first);
            const SECOND: u8 = $second;
            // 8 bytes on 64-bit, 4 on 32-bit.
            const WORD: usize = $crate::WORD_BYTES;

            let mut positions: ::std::vec::Vec<usize> = ::std::vec::Vec::new();
            let mut chunks = data.chunks_exact(WORD);

            for (chunk_index, chunk) in chunks.by_ref().enumerate() {
                let word = usize::from_le_bytes(
                    chunk
                        .try_into()
                        .expect("chunks_exact(WORD) always yields WORD-byte chunks"),
                );

                // Flag lanes equal to the first byte.
                let mut mask = $crate::zero_byte_mask(word ^ PATTERN);
                let base = chunk_index * WORD;

                while mask != 0 {
                    let byte_offset = (mask.trailing_zeros() as usize) / 8;
                    let index = base + byte_offset;
                    // Confirm the following byte (bounds-checked).
                    if data.get(index + 1) == Some(&SECOND) {
                        positions.push(index);
                    }
                    mask &= mask - 1; // clear lowest set bit
                }
            }

            // Sub-word tail.
            let remainder = chunks.remainder();
            let remainder_base = data.len() - remainder.len();
            const FIRST: u8 = $first;
            for (offset, &byte) in remainder.iter().enumerate() {
                let index = remainder_base + offset;
                if byte == FIRST && data.get(index + 1) == Some(&SECOND) {
                    positions.push(index);
                }
            }

            positions
        }
    };
}

/// Like [`swar_search!`], but backed by explicit SIMD via the
/// [`wide`](https://docs.rs/wide) crate: it loads [`SIMD_LANES`](crate::SIMD_LANES)
/// bytes into a 128-bit vector and compares every lane in one instruction.
///
/// Same syntax, forms, and results as [`swar_search!`]; pattern bytes must be
/// compile-time constants.
///
/// ```text
/// simd_search!(name, first);            // single-byte pattern
/// simd_search!(name, first, second);    // two-byte pattern
/// simd_search!(pub name, first);        // with visibility
/// ```
///
/// # Examples
///
/// ```
/// use byte_search::simd_search;
///
/// simd_search!(pub find_zeros, 0x00);
/// assert_eq!(find_zeros(&[1, 0, 2, 0, 0]), vec![1, 3, 4]);
/// ```
#[macro_export]
macro_rules! simd_search {
    // Single-byte pattern.
    ($vis:vis $name:ident, $first:expr $(,)?) => {
        $vis fn $name(data: &[u8]) -> ::std::vec::Vec<usize> {
            const FIRST: u8 = $first;
            const LANES: usize = $crate::SIMD_LANES;

            let mut positions: ::std::vec::Vec<usize> = ::std::vec::Vec::new();
            let mut chunks = data.chunks_exact(LANES);

            for (chunk_index, chunk) in chunks.by_ref().enumerate() {
                let block: [u8; LANES] = chunk
                    .try_into()
                    .expect("chunks_exact(LANES) always yields LANES-byte chunks");

                // Flag lanes equal to the pattern (one SIMD compare).
                let mut mask = $crate::simd_match_mask(block, FIRST);
                let base = chunk_index * LANES;

                while mask != 0 {
                    let lane = mask.trailing_zeros() as usize;
                    positions.push(base + lane);
                    mask &= mask - 1; // clear lowest set bit
                }
            }

            // Sub-vector tail.
            let remainder = chunks.remainder();
            let remainder_base = data.len() - remainder.len();
            for (offset, &byte) in remainder.iter().enumerate() {
                if byte == FIRST {
                    positions.push(remainder_base + offset);
                }
            }

            positions
        }
    };

    // Two-byte pattern.
    ($vis:vis $name:ident, $first:expr, $second:expr $(,)?) => {
        $vis fn $name(data: &[u8]) -> ::std::vec::Vec<usize> {
            const FIRST: u8 = $first;
            const SECOND: u8 = $second;
            const LANES: usize = $crate::SIMD_LANES;

            let mut positions: ::std::vec::Vec<usize> = ::std::vec::Vec::new();
            let mut chunks = data.chunks_exact(LANES);

            for (chunk_index, chunk) in chunks.by_ref().enumerate() {
                let block: [u8; LANES] = chunk
                    .try_into()
                    .expect("chunks_exact(LANES) always yields LANES-byte chunks");

                // Flag lanes equal to the first byte (one SIMD compare).
                let mut mask = $crate::simd_match_mask(block, FIRST);
                let base = chunk_index * LANES;

                while mask != 0 {
                    let lane = mask.trailing_zeros() as usize;
                    let index = base + lane;
                    // Confirm the following byte (bounds-checked).
                    if data.get(index + 1) == Some(&SECOND) {
                        positions.push(index);
                    }
                    mask &= mask - 1; // clear lowest set bit
                }
            }

            // Sub-vector tail.
            let remainder = chunks.remainder();
            let remainder_base = data.len() - remainder.len();
            for (offset, &byte) in remainder.iter().enumerate() {
                let index = remainder_base + offset;
                if byte == FIRST && data.get(index + 1) == Some(&SECOND) {
                    positions.push(index);
                }
            }

            positions
        }
    };
}

#[cfg(test)]
mod tests {
    use super::{LANE_HIGH_BITS, zero_byte_mask};

    const PATTERN_A: u8 = 0xFF;
    const PATTERN_B: u8 = 0xFE;

    // Functions generated by both macros under test.
    swar_search!(find_byte, PATTERN_A);
    swar_search!(find_pair, PATTERN_A, PATTERN_B);
    swar_search!(pub(crate) find_zeros, 0x00);

    simd_search!(simd_find_byte, PATTERN_A);
    simd_search!(simd_find_pair, PATTERN_A, PATTERN_B);
    simd_search!(pub(crate) simd_find_zeros, 0x00);

    // High bit of the highest byte lane in a `usize`.
    const HIGHEST_LANE_BIT: usize = 1 << (usize::BITS - 1);

    #[test]
    fn detects_zero_byte_in_lowest_position() {
        // Only the lowest lane is zero.
        assert_eq!(zero_byte_mask(usize::MAX & !0xFF), 0x80);
    }

    #[test]
    fn detects_zero_byte_in_highest_position() {
        // Only the highest lane is zero.
        assert_eq!(zero_byte_mask(usize::MAX >> 8), HIGHEST_LANE_BIT);
    }

    #[test]
    fn detects_multiple_zero_bytes() {
        // Lowest and highest lanes zero.
        let value = (usize::MAX >> 8) & !0xFF;
        assert_eq!(zero_byte_mask(value), HIGHEST_LANE_BIT | 0x80);
    }

    #[test]
    fn returns_zero_when_no_zero_bytes_present() {
        assert_eq!(zero_byte_mask(usize::MAX), 0);
    }

    #[test]
    fn flags_all_lanes_for_all_zeroes() {
        assert_eq!(zero_byte_mask(0), LANE_HIGH_BITS);
    }

    #[test]
    fn does_not_false_positive_adjacent_to_zero_byte() {
        // Lane 0 = 0x00, lane 1 = 0x01, rest 0xFF: only lane 0 flagged.
        let value = (usize::MAX & !0xFFFF) | 0x0100;
        assert_eq!(zero_byte_mask(value), 0x80);
    }

    #[test]
    fn single_byte_search_within_one_chunk() {
        // Fits in a single word.
        let data = [
            PATTERN_A, 0x00, PATTERN_A, 0x01, 0x02, PATTERN_A, 0x03, 0x04,
        ];
        assert_eq!(find_byte(&data), vec![0, 2, 5]);
    }

    #[test]
    fn single_byte_search_scans_remainder() {
        // Full word + a shorter tail holding matches.
        let mut data = vec![0u8; 8];
        data.extend_from_slice(&[PATTERN_A, 0x00, PATTERN_A]);
        assert_eq!(find_byte(&data), vec![8, 10]);
    }

    #[test]
    fn single_byte_search_empty_input() {
        assert!(find_byte(&[]).is_empty());
    }

    #[test]
    fn two_byte_search_basic() {
        let sequence = [
            PATTERN_A, PATTERN_B, 0x00, PATTERN_A, 0x00, PATTERN_A, PATTERN_B, PATTERN_A,
            PATTERN_B, PATTERN_A,
        ];
        assert_eq!(find_pair(&sequence), vec![0, 5, 7]);
    }

    #[test]
    fn two_byte_search_across_chunk_boundary() {
        // First byte ends one word; second byte starts the tail.
        let sequence = [
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, PATTERN_A, PATTERN_B, 0xAA,
        ];
        assert_eq!(find_pair(&sequence), vec![7]);
    }

    #[test]
    fn two_byte_search_first_byte_without_second() {
        // A trailing first byte with nothing after it must not match.
        let sequence = [PATTERN_A, 0x00, PATTERN_A];
        assert!(find_pair(&sequence).is_empty());
    }

    #[test]
    fn two_byte_search_in_remainder() {
        let mut data = vec![0u8; 8];
        data.extend_from_slice(&[PATTERN_A, PATTERN_B, 0x00]);
        assert_eq!(find_pair(&data), vec![8]);
    }

    #[test]
    fn generated_with_visibility_qualifier() {
        assert_eq!(find_zeros(&[1, 0, 2, 0, 0]), vec![1, 3, 4]);
    }

    #[test]
    fn simd_single_byte_search_within_one_vector() {
        let data = [
            PATTERN_A, 0x00, PATTERN_A, 0x01, 0x02, PATTERN_A, 0x03, 0x04,
        ];
        assert_eq!(simd_find_byte(&data), vec![0, 2, 5]);
    }

    #[test]
    fn simd_single_byte_search_scans_remainder() {
        // Full vector + a shorter tail holding matches.
        let mut data = vec![0u8; 16];
        data.extend_from_slice(&[PATTERN_A, 0x00, PATTERN_A]);
        assert_eq!(simd_find_byte(&data), vec![16, 18]);
    }

    #[test]
    fn simd_single_byte_search_empty_input() {
        assert!(simd_find_byte(&[]).is_empty());
    }

    #[test]
    fn simd_two_byte_search_basic() {
        let sequence = [
            PATTERN_A, PATTERN_B, 0x00, PATTERN_A, 0x00, PATTERN_A, PATTERN_B, PATTERN_A,
            PATTERN_B, PATTERN_A,
        ];
        assert_eq!(simd_find_pair(&sequence), vec![0, 5, 7]);
    }

    #[test]
    fn simd_two_byte_search_across_vector_boundary() {
        // First byte ends one vector; second byte starts the tail.
        let mut data = vec![0u8; 15];
        data.push(PATTERN_A);
        data.push(PATTERN_B);
        data.push(0xAA);
        assert_eq!(simd_find_pair(&data), vec![15]);
    }

    #[test]
    fn simd_two_byte_search_first_byte_without_second() {
        let sequence = [PATTERN_A, 0x00, PATTERN_A];
        assert!(simd_find_pair(&sequence).is_empty());
    }

    #[test]
    fn simd_generated_with_visibility_qualifier() {
        assert_eq!(simd_find_zeros(&[1, 0, 2, 0, 0]), vec![1, 3, 4]);
    }

    #[test]
    fn simd_and_swar_agree_over_many_vectors() {
        // Both macros must agree over many words/vectors with scattered markers.
        let mut data = vec![0u8; 500];
        let mut x: u32 = 0x9E37_79B9;
        for byte in &mut data {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            *byte = (x >> 16) as u8;
        }
        for i in (0..data.len().saturating_sub(1)).step_by(37) {
            data[i] = PATTERN_A;
            data[i + 1] = PATTERN_B;
        }

        assert_eq!(simd_find_byte(&data), find_byte(&data));
        assert_eq!(simd_find_pair(&data), find_pair(&data));
    }
}
