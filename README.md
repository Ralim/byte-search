# Byte Search

This crate provides fast byte search utilities using SIMD and SWAR techniques.
Its designed for searching for a one or two byte sync sequence in binary data.
It only operates on in-memory `&[u8]` slices.

Two macros generate the search functions from a compile-time pattern:

- [`swar_search!`] — SWAR (SIMD-within-a-register). Scans a whole machine word
  (`usize`) at a time using bit tricks; no dependencies, works everywhere.
- [`simd_search!`] — explicit SIMD (via the [`wide`](https://docs.rs/wide)
  crate). Compares a 128-bit vector of bytes per instruction where hardware
  support exists, falling back to scalar otherwise.

Both accept identical syntax and return identical results: a `Vec<usize>` of
every index where the pattern begins.

## Usage

Add the crate to your `Cargo.toml`:

```toml
[dependencies]
byte-search = "0.1"
```

### Single-byte search

```rust
use byte_search::swar_search;

// Generates `fn find_newlines(&[u8]) -> Vec<usize>`.
swar_search!(find_newlines, b'\n');

let positions = find_newlines(b"a\nbc\n");
assert_eq!(positions, vec![1, 4]);
```

### Two-byte (sync sequence) search

```rust
use byte_search::swar_search;

// Match a two-byte sync header, e.g. `0xAA 0x55`.
swar_search!(find_sync, 0xAA, 0x55);

let frame = [0x00, 0xAA, 0x55, 0x01, 0xAA, 0x55];
assert_eq!(find_sync(&frame), vec![1, 4]);
```

### Explicit SIMD

`simd_search!` is a drop-in replacement that uses vector instructions:

```rust
use byte_search::simd_search;

simd_search!(find_sync, 0xAA, 0x55);

let frame = [0x00, 0xAA, 0x55, 0x01, 0xAA, 0x55];
assert_eq!(find_sync(&frame), vec![1, 4]);
```

### Visibility

An optional visibility qualifier controls the generated function's visibility:

```rust
use byte_search::swar_search;

swar_search!(pub find_zeros, 0x00);
swar_search!(pub(crate) find_ones, 0x01);
```

## Benchmarks

Searching for a two-byte marker in a 512 KiB buffer (one marker every 97 bytes),
compared against a naive `windows(2).filter(..)` iterator. Numbers are median
times from the `byte_search` Criterion benchmark and will vary by CPU.

| Implementation | Time     | Throughput  | Speedup vs naive |
| -------------- | -------- | ----------- | ---------------- |
| naive          | 196.9 µs | 2.48 GiB/s  | 1.0×             |
| `swar_search!` | 62.9 µs  | 7.76 GiB/s  | 3.1×             |
| `simd_search!` | 30.7 µs  | 15.90 GiB/s | 6.4×             |

Reproduce with:

```sh
cargo bench --bench byte_search
```

[`swar_search!`]: https://docs.rs/byte-search/latest/byte_search/macro.swar_search.html
[`simd_search!`]: https://docs.rs/byte-search/latest/byte_search/macro.simd_search.html


## Licence

All Code is licenced under the MIT Licence
