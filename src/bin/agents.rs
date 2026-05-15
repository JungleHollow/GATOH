use rand::prelude::*;
use std::collections::HashMap;
use std::str;

/// Definition of all valid, existing Agent personality types
const PERSONALITIES: [&str; 5] = ["neutral", "rational", "erratic", "impulsive", "social"];

/// An Agent utility function that randomly draws a valid Agent personality type.
fn _draw_personality() -> String {
    let mut rng = rand::rng();
    return PERSONALITIES.choose(&mut rng).unwrap().to_string();
}

/// An Enum that simplifies the data typing for ``parameters'' HashMaps across functions
enum ParameterTypes {
    StringParam(String),
    IntParam(i64),  // i32 instead of u32 to allow for negatives
    FloatParam(f64),
    BoolParam(bool)
}

struct Agent {
    /// A struct to define the Agent objects that will interact with each other in an agent-based model.
    id: String,
    index: u64,
    social_weightings: HashMap<String, f64>,
    is_silenced: HashMap<String, bool>,
    opinion: f64,
    previous_opinion: f64,
    personal_benefit: bool,
    social_susceptibility: f64,
    personality: String,
    radicalised: bool,
    rw_distributions: HashMap<String, (f64, f64)>,
    opinion_rw: (f64, f64),
}

impl Agent {
    /// A constructor to create a new Agent.
    fn new(
        id: &str,
        index: Option<u64>,
        social_weightings: HashMap<String, f64>,
        is_silenced: Option<HashMap<String, bool>>,
        opinion: f64,
        previous_opinion: Option<f64>,
        personal_benefit: bool,
        personality: &str,
        susceptibility: f64,
        radicalised: Option<bool>,
        rw_distributions: Option<HashMap<String, (f64, f64)>>,
        opinion_rw: Option<(f64, f64)>
    ) -> Self {
        Agent {
            id: String::from(id),
            index: index.unwrap_or(0), // An index of 0 is not valid, treated as a null value
            social_weightings: social_weightings,
            is_silenced: is_silenced.unwrap_or(HashMap::new()),
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

    /// Randomly generate the Agent attributes based on the input parameters.
    fn generate_agent(
        &mut self,
        id: &str,
        index: u64,
        hierarchies: Vec<&str>,
        distribution: Option<&str>,
        explicit_rw: Option<bool>,
        explicit_opinion_rw: Option<bool>,
        personality: Option<&str>,
        parameters: Option<HashMap<String, ParameterTypes>>,
        personal_benefit: Option<bool>,
    ) {
        // Set the crucial information first
        self.id = id.to_string();
        self.index = index;

        self.personality = personality.unwrap_or("neutral").to_string();
        self.personal_benefit = personal_benefit.unwrap_or(false);

        // Generate a weighting for each hierarchy; initialise the is_silenced flag for that hierarchy
        for hierarchy in hierarchies {
            self.social_weightings.insert(
                hierarchy.to_string(),
                draw_random_value(&distribution, parameters)
            );
            self.is_silenced.insert(hierarchy.to_string(), false);
        }

        // Handle explicit hierarchy weighting random walk parameter generation
        if explicit_rw.is_some() {
            let mut rw_params = HashMap::new();
            for hierarchy in hierarchies {
                let rw_mean: f64 = draw_random_value("gaussian", parameters);
                let rw_variance: f64 = draw_random_value("gaussian", parameters);

                rw_params.insert(hierarchy.to_string(), (rw_mean, rw_variance));
            }
            self.rw_distributions = rw_params;
        }

        // Handle the explicit opinion random walk parameter generation
        if explicit_opinion_rw.is_some() {
            let rw_mean: f64 = draw_random_value("gaussian", parameters);
            let rw_variance: f64 = draw_random_value("gaussian", parameters);
            self.opinion_rw = (rw_mean, rw_variance);
        }

        // Generate the Agent's initial opinion
        self.opinion = draw_random_value(&distribution, parameters);

        // If the initial opinion is very strong, the Agent is initialised as radicalised
        if self.opinion <= -0.9 || self.opinion >= 0.9 {
            self.radicalised = true;
        }

        // Generate the Agent's susceptibility to social contagion
        self.social_susceptibility = draw_random_value(&distribution, parameters);
    }

    /// A setter method that stores the Agent's current opinion into the previous opinion.
    fn store_previous_opinion(&mut self) {
        let opinion: f64 = self.opinion;
        self.previous_opinion = opinion;
    }

    /// A setter method that changes the Agent's current opinion by a given delta value.
    fn change_opinion(&mut self, opinion_delta: f64) {
        self.opinion += opinion_delta;

        if self.opinion < -1.0 {
            self.opinion = -1.0;
        } else if self.opinion > 1.0 {
            self.opinion = 1.0;
        }
    }

    /// A setter method that changes the Agent's radicalisation value.
    fn change_radicalisation(&mut self, radicalisation: bool) {
        self.radicalised = radicalisation;
    }

    /// A setter method that changes the Agent's explicit random walk parameters for a specific social hierarchy.
    fn change_rw_distribution(&mut self, hierarchy: &str, parameters: (f64, f64)) {
        self.rw_distributions.insert(hierarchy.to_string(), parameters);
    }

    /// A setter method that changes the Agent's explicit opinion random walk parameters.
    fn change_opinion_rw(&mut self, rw_params: (f64, f64)) {
        self.opinion_rw = rw_params;
    }

    /// Step the individual Agent object:
    ///     1. Handle the dynamic social hierarchy weightings
    ///     2. Handle the stochastic opinion changes experienced by the Agent
    fn step(&mut self, rw_distributions: HashMap<String, (f64, f64)>, opinion_rw: (f64, f64)) {
        Self::evolve_hierarchies(rw_distributions);
        Self::stochastic_opinion(opinion_rw);
    }

    /// Updates the internal state of the Agent after the model has stepped:
    ///     1. Updates what social hierarchies the Agent's opinion is currently silenced in
    ///     2. Inverts the Agent's current opinion if opinion negation ocurred
    fn update(&mut self, opinion_silenced: HashMap<String, bool>, negation_ocurred: bool) {
        self.is_silenced = opinion_silenced;
        if negation_ocurred {
            self.opinion *= -1.0;
        }
    }

    /// Determines if an Agent will become silenced in a given social hierarchy based on their attributes.
    ///
    /// If no silencing threshold has been passed, each Agent's own social susceptibility is used as the threshold instead.
    fn opinion_silencing(&mut self, hierarchy: &str, estimated_opinion_climate: f64, silencing_threshold: Option<f64>) -> (bool, f64) {
        if self.radicalised {
            return (false, 0.0);
        } else {
            let threshold: f64 = silencing_threshold.unwrap_or(self.social_susceptibility);

            let mut absolute_difference: f64 = 0.0;
            let personality_str: &str = self.personality.as_str();
            let difference: f64 = estimated_opinion_climate - self.opinion;

            if ["neutral", "rational", "erratic"].contains(&personality_str) {
                // Cases where opinion silencing will be less influenced by the surrounding opinion climate.
                absolute_difference = difference.abs() * 0.8;
            } else if ["impulsive", "social"].contains(&personality_str) {
                // Cases where opinion silencing will be much more influenced by the surrounding opinion climate.
                absolute_difference = difference.abs();
            }

            return (absolute_difference > threshold, absolute_difference);
        }
    }

    /// Checks if the Agent has experienced sufficiently `overwhelming' social pressure in a hierarchy leading to a complete
    /// reversal of their opinion.
    fn opinion_negation(&mut self, hierarchy: &str, absolute_difference: f64, threshold: f64) -> bool {
        if self.radicalised {
            return false;
        } else {
            let mut negation_strength: f64 = absolute_difference;
            let personality_str: &str = self.personality.as_str();

            // Multiplication by (susceptibility * hierarchy weighting) will always decrease negation strength, whilst division will always increase it
            if ["neutral", "rational"].contains(&personality_str) {
                // Cases where opinion negation is less likely to occur
                negation_strength *= self.social_susceptibility * self.social_weightings[hierarchy];
            } else if ["erratic", "impulsive", "social"].contains(&personality_str) {
                negation_strength /= self.social_susceptibility * self.social_weightings[hierarchy];
            }
            return negation_strength > threshold;
        }
    }
}
