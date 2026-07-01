//! Compares the generated `swar_search!` and `simd_search!` functions against a
//! naive iterator/filter baseline over overlapping byte windows.

use std::hint::black_box;

use byte_search::{simd_search, swar_search};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

// Arbitrary two-byte marker.
const MARKER1: u8 = 0xAB;
const MARKER2: u8 = 0xCD;

swar_search!(find_markers_swar, MARKER1, MARKER2);
simd_search!(find_markers_simd, MARKER1, MARKER2);

/// Naive baseline: an iterator/filter combo over 2-byte windows.
fn find_markers_naive(data: &[u8]) -> Vec<usize> {
    data.windows(2)
        .enumerate()
        .filter_map(|(i, window)| (window[0] == MARKER1 && window[1] == MARKER2).then_some(i))
        .collect()
}

/// Build a deterministic dataset with markers injected at regular intervals.
fn build_dataset(size: usize) -> Vec<u8> {
    let mut data = vec![0u8; size];

    // Deterministic filler, no extra dependencies.
    let mut x: u32 = 0x1234_5678;
    for byte in &mut data {
        x = x.wrapping_mul(1664525).wrapping_add(1013904223);
        *byte = (x >> 16) as u8;
    }

    // Inject the marker at regular intervals.
    let mut i = 0usize;
    while i + 1 < size {
        data[i] = MARKER1;
        data[i + 1] = MARKER2;
        i += 97;
    }

    // Clear the tail so no stray marker lands at the buffer edge.
    for byte in data.iter_mut().rev().take(2) {
        *byte = 0x00;
    }

    data
}

fn bench_byte_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("byte_search");

    // 512 KiB is the headline size; the small size is a baseline.
    for &size in &[4 * 1024, 512 * 1024] {
        let data = build_dataset(size);

        // All implementations must agree before timing.
        let naive = find_markers_naive(&data);
        let swar = find_markers_swar(&data);
        let simd = find_markers_simd(&data);
        assert_eq!(naive, swar, "naive and SWAR implementations diverged");
        assert_eq!(naive, simd, "naive and SIMD implementations diverged");

        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::new("naive", size), &data, |b, d| {
            b.iter(|| find_markers_naive(black_box(d)));
        });

        group.bench_with_input(BenchmarkId::new("swar", size), &data, |b, d| {
            b.iter(|| find_markers_swar(black_box(d)));
        });

        group.bench_with_input(BenchmarkId::new("simd", size), &data, |b, d| {
            b.iter(|| find_markers_simd(black_box(d)));
        });
    }

    group.finish();
}

criterion_group!(benches, bench_byte_search);
criterion_main!(benches);
