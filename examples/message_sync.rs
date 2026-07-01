//! Demonstrates `swar_search!` by locating message sync headers in a buffer.
//!
//! Run with:
//!
//! ```text
//! cargo run --example message_sync
//! ```

use byte_search::swar_search;

// A protocol that frames messages with the two-byte sync sequence `0xAA 0x55`.
const SYNC1: u8 = 0xAA;
const SYNC2: u8 = 0x55;

// Start index of every two-byte sync header.
swar_search!(find_message_starts, SYNC1, SYNC2);

// Every index of the single sync byte.
swar_search!(find_sync1, SYNC1);

fn main() {
    // Noisy buffer with three sync headers (indices 2, 9, 16).
    let stream: &[u8] = &[
        0x00, 0xFF, // noise
        SYNC1, SYNC2, 0x01, 0x02, 0x03, 0x04, 0x05, // message 1
        SYNC1, SYNC2, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, // message 2
        SYNC1, SYNC2, 0x10, // message 3 (header straddles a chunk)
    ];

    let starts = find_message_starts(stream);
    println!("Message starts at:   {starts:?}");

    let sync1_positions = find_sync1(stream);
    println!("0xAA bytes found at: {sync1_positions:?}");

    for (n, start) in starts.iter().enumerate() {
        println!("  message {} begins at byte {}", n + 1, start);
    }
}
