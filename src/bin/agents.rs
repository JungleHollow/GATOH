use rand::seq::IndexedRandom;
use rand::Rng;
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::io;
use std::option;
use std::slice;
use std::str;

// Definition of all valid, existing Agent personality types
const PERSONALITIES: [&str; 5] = ["neutral", "rational", "erratic", "impulsive", "social"];

fn _draw_personality() -> &str {
    /// An Agent utility function that randomly draws a valid Agent personality type.
    let mut rng = rand::rng();
    let drawn_personality: str = PERSONALITIES.choose(&mut rng);
    return &drawn_personality;
}

struct Agent {
    /// A struct to define the Agent objects that will interact with each other in an agent-based model.
    id: String,
    index: u32,
    social_weightings: HashMap,
    is_silenced: bool,
    opinion: f32,
    previous_opinion: f32,
    personal_benefit: bool,
    social_susceptibility: f32,
    personality: String,
    radicalised: bool,
    rw_distributions: HashMap,
    opinion_rw: tuple<f32, f32>,
}

impl Agent {
    // A constructor to create a new Agent.
    fn new(
        id: &str,
        index: Option<u32>,
        social_weightings: HashMap,
        is_silenced: Option<bool>,
        opinion: f32,
        previous_opinion: Option<f32>,
        personal_benefit: bool,
        personality: &str,
        susceptibility: f32,
        radicalised: Option<bool>,
        rw_distributions: Option<HashMap>,
        opinion_rw: Option<tuple<f32, f32>>
    ) -> Agent {
        Agent {
            id: String::from(id),
            index: index.unwrap_or(0), // An index of 0 is not valid, treated as a null value
            social_weightings: social_weightings,
            is_silenced: is_silenced.unwrap_or(false),
            opinion: opinion,
            previous_opinion: previous_opinion.unwrap_or(0.0),
            personal_benefit: personal_benefit,
            social_susceptibility: susceptibility,
            personality: String::from(personality),
            radicalised: radicalised.unwrap_or(false),
            rw_distributions: rw_distributions.unwrap_or(HashMap::new()),
            opinion_rw: opinion_rw.unwrap_or((0.0, 0.1))
        }
    }

    // Randomly generate the Agent attributes based on the input parameters.
    fn generate_agent(
        &self,
        id: &str,
        index: u32,
        hierarchies: Vec<&str>,
        distribution: Option<&str>,
        explicit_rw: Option<bool>,
        explicit_opinion_rw: Option<bool>,
        personality: Option<&str>,
        parameters: Option<HashMap>,
        personal_benefit: Option<bool>,
    ) -> Self {
        // Set the crucial information first
        &self.id = id.to_string();
        &self.index = index;

        &self.personality = personality.unwrap_or("neutral").to_string();
        &self.personal_benefit = personal_benefit.unwrap_or(false);

        // Generate a weighting for each hierarchy; initialise the is_silenced flag for that hierarchy
        for hierarchy in hierarchies {
            &self.social_weightings.insert(
                hierarchy.to_string(),
                draw_random_value(&distribution, parameters)
            );
            &self.is_silenced.insert(hierarchy.to_string(), false);
        }

        // Handle explicit hierarchy weighting random walk parameter generation
        if explicit_rw.is_some() {
            let mut rw_params = HashMap::new();
            for hierarchy in hierarchies {
                let rw_mean: f32 = draw_random_value("gaussian", parameters);
                let rw_variance: f32 = draw_random_value("gaussian", parameters);

                rw_params.insert(hierarchy.to_string(), (rw_mean, rw_variance));
            }
            &self.rw_distributions = rw_params;
        }

        // Handle the explicit opinion random walk parameter generation
        if explicit_opinion_rw.is_some() {
            let rw_mean: f32 = draw_random_value("gaussian", parameters);
            let rw_variance: f32 = draw_random_value("gaussian", parameters);
            &self.opinion_rw = (rw_mean, rw_variance);
        }

        // Generate the Agent's initial opinion
        &self.opinion = draw_random_value(&distribution, parameters);

        // If the initial opinion is very strong, the Agent is initialised as radicalised
        if &self.opinion <= -0.9 || &self.opinion >= 0.9 {
            &self.radicalised = true;
        }

        // Generate the Agent's susceptibility to social contagion
        &self.social_susceptibility = draw_random_value(&distribution, parameters);
    }

    // A setter method that stores the Agent's current opinion into the previous opinion.
    fn store_previous_opinion(&self) {
        &self.previous_opinion = &self.opinion;
    }

    // A setter method that changes the Agent's current opinion by a given delta value.
    fn change_opinion(&self, opinion_delta: f32) {
        &self.opinion += opinion_delta;

        if &self.opinion < -1.0 {
            &self.opinion = -1.0;
        } else if &self.opinion > 1.0 {
            &self.opinion = 1.0;
        }
    }
}
