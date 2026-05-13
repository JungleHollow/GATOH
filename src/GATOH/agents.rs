use std::io;
use std::any;
use std::fmt;
use std::fs;
use std::option;
use std::slice;
use std::str;
use rand::Rng;
use rand::seq::IndexedRandom;

// Definition of all valid, existing Agent personality types
let PERSONALITIES: [str; 5] = ["neutral", "rational", "erratic", "impulsive", "social"];

fn _draw_personality() -> str {
    /// An Agent utility function that randomly draws a valid Agent personality type.
    let mut rng = rand::rng();
    let drawn_personality: str = PERSONALITIES.choose(&mut rng);
    return drawn_personality;
}
