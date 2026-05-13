use rand::seq::IndexedRandom;
use rand::Rng;
use std::any;
use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::io;
use std::option;
use std::slice;
use std::str;

// Definition of all valid, existing Agent personality types
const PERSONALITIES: [str; 5] = ["neutral", "rational", "erratic", "impulsive", "social"];

fn _draw_personality() -> str {
    /// An Agent utility function that randomly draws a valid Agent personality type.
    let mut rng = rand::rng();
    let drawn_personality: str = PERSONALITIES.choose(&mut rng);
    return drawn_personality;
}

struct Agent {
    /// A struct to define the Agent objects that will interact with each other in an agent-based model.
    id: str,
    index: u32,
    social_weightings: HashMap,
    is_silenced: bool,
    opinion: f32,
    previous_opinion: f32,
    personal_benefit: bool,
    social_susceptibility: f32,
    personality: str,
    radicalised: bool,
    rw_distributions: HashMap,
    opinion_rw: tuple,
}

impl Agent {
    fn new(
        id: &str,
        social_weightings: HashMap,
        opinion: f32,
        personal_benefit: bool,
        personality: &str,
        susceptibility: f32,
    ) -> Agent {
        /// A constructor for the Agent struct.
        Agent {
            id: String::from(id),
            // index
            social_weightings,
            // is_silenced
            opinion,
            // previous_opinion
            personal_benefit,
            susceptibility,
            personality: String::from(personality),
            // radicalised
            // rw_distributions
            // opinion_rw
        }
    }
}
