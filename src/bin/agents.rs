use conv::*;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use serde_pickle;
use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;
use std::str;
use zip::write::FileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

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
    IntParam(i64), // i32 instead of u32 to allow for negatives
    FloatParam(f64),
    BoolParam(bool),
}

/// A struct to define the Agent objects that will interact with each other in an agent-based model.
#[derive(Debug, Serialize, Deserialize)]
struct Agent {
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
        opinion_rw: Option<(f64, f64)>,
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
            opinion_rw: opinion_rw.unwrap_or((-99.9, 0.1)),
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
                draw_random_value(&distribution, parameters),
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
        self.rw_distributions
            .insert(hierarchy.to_string(), parameters);
    }

    /// A setter method that changes the Agent's explicit opinion random walk parameters.
    fn change_opinion_rw(&mut self, rw_params: (f64, f64)) {
        self.opinion_rw = rw_params;
    }

    /// Step the individual Agent object:
    ///     1. Handle the dynamic social hierarchy weightings
    ///     2. Handle the stochastic opinion changes experienced by the Agent
    fn step(&mut self, rw_distributions: HashMap<String, (f64, f64)>, opinion_rw: (f64, f64)) {
        Self::evolve_hierarchies(self, rw_distributions);
        Self::stochastic_opinion(self, opinion_rw);
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
    fn opinion_silencing(
        &mut self,
        hierarchy: &str,
        estimated_opinion_climate: f64,
        silencing_threshold: Option<f64>,
    ) -> (bool, f64) {
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
    fn opinion_negation(
        &mut self,
        hierarchy: &str,
        absolute_difference: f64,
        threshold: f64,
    ) -> bool {
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

    /// Uses the Agent's own opinion as well as the neighbours' opinions to determine if the Agent has become
    /// radicalised in their actions.
    fn radicalisation(
        &mut self,
        hierarchy_changes: &[f64],
        neighbour_benefits: &[bool],
        hierarchies: &[&str],
        threshold: f64,
    ) -> bool {
        if self.radicalised {
            return false;
        } else {
            let absolute_opinion: f64 = self.opinion.abs();

            if self.personality == "neutral" {
                // This will mean that radicalisation is exclusively determined by the strength of the Agent's opinion
                self.radicalised = true;
                return self.radicalised;
            } else if self.personality == "rational" {
                // This will likely mean that the agent is more disposed towards considering tangible benefits and their own
                // opinions when determining radicalisation, rather than external social influences
                let collective_benefit: f64 =
                    f64::value_from(neighbour_benefits.into_iter().filter(|b| **b).count())
                        .unwrap()
                        / f64::value_from(neighbour_benefits.len()).unwrap();

                if absolute_opinion >= threshold && collective_benefit >= 0.5 {
                    self.radicalised = true;
                    return self.radicalised;
                } else if absolute_opinion >= threshold && collective_benefit < 0.5 {
                    // In the case where the threshold is met but there is no explicit collective benefit, radicalisation is treated as a coinflip
                    self.radicalised = random_coinflip("bool");
                    return self.radicalised;
                } else {
                    return false;
                }
            } else if self.personality == "erratic" {
                // Radicalisation is influenced by personal opinion to some extent, but is largely stochastically determined
                if absolute_opinion * 1.25 >= threshold {
                    self.radicalised = random_coinflip("bool");
                    return self.radicalised;
                } else {
                    return false;
                }
            } else if self.personality == "impulsive" {
                // The Agent places very strong consideration on tangible benefits over anything else
                if absolute_opinion >= threshold / 2.0 {
                    self.radicalised = self.personal_benefit;
                    return self.radicalised;
                } else {
                    return false;
                }
            } else if self.personality == "social" {
                // Radicalisation is strongly determined by the opinion climate and neighbour opinions rather than internal factors
                let mut absolute_changes: f64 = 0.0;

                for change in hierarchy_changes {
                    let absolute_change: f64 = change.abs();
                    if absolute_change >= self.social_susceptibility {
                        // A strong opinion change was caused by some hierarchy
                        self.radicalised = true;
                        return self.radicalised;
                    } else {
                        absolute_changes += absolute_change;
                    }
                }
                // If no changes were strong enough individually, check for the aggregate (with a relatively lower threshold)
                if absolute_changes
                    >= self.social_susceptibility
                        * f64::value_from(hierarchy_changes.len() / 2).unwrap()
                {
                    self.radicalised = true;
                    return self.radicalised;
                } else {
                    return false;
                }
            } else {
                return false;
            }
        }
    }

    /// Experimental function that aims to model the constantly evolving `intrinsic value' that Agents place on the social hierarchies
    /// that they belong in over time.
    fn evolve_hierarchies(&mut self, rw_distributions: HashMap<String, (f64, f64)>) {
        for (key, value) in &rw_distributions {
            let rw_result: Option<f64> = None;

            if self.rw_distributions.len() > 0 {
                if self.rw_distributions.contains_key(key) {
                    rw_result = value_rw_delta(
                        self.social_weightings[key],
                        self.rw_distributions[key].0,
                        self.rw_distributions[key].1,
                    );
                }
            }

            if rw_result.is_none() {
                // No explicit rw distribution was found; use the input ones instead
                rw_result = value_rw_delta(self.social_weightings[key], value.0, value.1);
            }

            // Constrain the result back to [-1, 1] if necessary
            if rw_result.unwrap() < -1.0 {
                self.social_weightings[key] = -1.0;
            } else if rw_result.unwrap() > 1.0 {
                self.social_weightings[key] = 1.0;
            } else {
                self.social_weightings[key] = rw_result.unwrap();
            }
        }
    }

    /// Determine the stochastic direction and magnitude of a shift in the Agent's opinion and apply it.
    fn stochastic_opinion(&mut self, opinion_rw: (f64, f64)) {
        let rw_result: Option<f64> = None;

        if self.opinion_rw.0 != -99.9 {
            // -99.9 for mean is used to indicate a null value
            rw_result = value_rw_delta(self.opinion, self.opinion_rw.0, self.opinion_rw.1);
        }

        if rw_result.is_none() {
            rw_result = value_rw_delta(self.opinion, opinion_rw.0, opinion_rw.1);
        }

        if rw_result.unwrap() < -1.0 {
            self.opinion = -1.0;
        } else if rw_result.unwrap() > 1.0 {
            self.opinion = 1.0;
        } else {
            self.opinion = rw_result.unwrap();
        }
    }

    /// Experimental function that aims to model the ways in which Agent behaviours change according to major random life events over time.
    fn life_events(&mut self) {
        // TODO: Implement this function
        unimplemented!();
    }
}

impl PartialEq for Agent {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl fmt::Display for Agent {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.radicalised {
            write!(
                f,
                "Agent {} which is radicalised with an opinion value of {}",
                self.id, self.opinion
            )
        } else {
            write!(
                f,
                "Agent {} which is not radicalised with an opinion value of {}",
                self.id, self.opinion
            )
        }
    }
}

/// An ordered collection of Agent objects that maintains consistency for the Model.
struct AgentSet {
    parent_model: model::ABModel,
    agents: Vec<Agent>,
    random: rand::prelude::ThreadRng,
}

impl AgentSet {
    /// A constructor to create a new AgentSet.
    fn new(&parent_model: model::ABModel) -> AgentSet {
        AgentSet {
            parent_model: parent_model,
            agents: Vec::new(),
            random: rand::rng(),
        }
    }

    /// Save the Agent objects into a compressed subdirectory representing the saved AgentSet.
    fn save_agentset(&mut self, directory_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let mut subdirectory_path: String = directory_path.clone().to_string();
        subdirectory_path.push_str("/_agentset");

        let subdirectory_path_clone: String = subdirectory_path.clone();
        let subdir_path = Path::new(subdirectory_path_clone.as_str());

        // Removes the subdirectory if it already exists to allow for a new overwrite
        if subdir_path.exists() {
            fs::remove_dir_all(subdir_path);
        }

        // Create the _agentset subdirectory
        fs::create_dir(subdir_path);

        let zipfile_path: String = format!("{}.zip", subdir_path.to_str().unwrap());
        let zip_path = Path::new(zipfile_path.as_str());

        // Removes the zip file if it already exists to allow for a new overwrite
        if zip_path.exists() {
            fs::remove_dir_all(zip_path);
        }

        let zip_file_handle = File::create(&zip_path)?;
        let mut zip_writer = ZipWriter::new(zip_file_handle);
        let zip_options = FileOptions::default() // TODO: Check the typing of this variable...
            .compression_method(CompressionMethod::Deflated)
            .unix_permissions(0o644);

        for agent in self.agents.iter() {
            let agent_save_file: String = format!("_agent_{}.pkl", agent.id);

            let pickled_agent = serde_pickle::to_vec(&agent, Default::default()).unwrap();

            zip_writer.start_file(agent_save_file, zip_options)?;
            zip_writer.write_all(&pickled_agent)?;
        }

        zip_writer.finish()?;
        Ok(())
    }

    /// Loads an AgentSet that has been saved following the same process as in the save_agentset() function.
    fn load_agentset(&mut self, load_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let zip_load_path = Path::new(format!("{}/_agentset.zip", load_path).as_str());

        if !zip_load_path.exists() {
            panic!(format!(
                "No saved AgentSet was found at the path: {}",
                zip_load_path.to_str().unwrap()
            ))
        }

        // The path to the uncompressed agentset subdirectory
        let subdirectory_path = Path::new(format!("{}/_agentset", load_path).as_str());

        // Remove any existing subdirectory with the same name to replace it with the newly loaded one
        if subdirectory_path.is_dir() {
            fs::remove_dir_all(subdirectory_path);
        }

        // Create the uncompressed subdirectory
        fs::create_dir(subdirectory_path);

        let zip_file = File::open(zip_load_path)?;
        let mut zip_reader = ZipArchive::new(zip_file)?;

        // First extract all the Agent pickles to the uncompressed subdirectory
        for i in 0..zip_reader.len() {
            let mut file = zip_reader.by_index(i)?;
            let file_name = file.name().to_owned();

            // Path where the extracted file will be written to
            let file_path = Path::new(
                format!("{}/{}", subdirectory_path.to_str().unwrap(), file_name).as_str(),
            );

            let mut output_file = File::create(&file_path)?;
            io::copy(&mut file, &mut output_file);
        }

        // Next, read and deserialise the Agent pickles and add them to self.agents
        for dir_entry in subdirectory_path.read_dir().expect("read_dir call failed") {
            if let Ok(dir_entry) = dir_entry {
                let pickle_path = Path::new(&dir_entry.path());

                let pickle_string = fs::read_to_string(pickle_path)?;
                let pickle_bytes = pickle_string.as_bytes();

                let agent_object =
                    serde_pickle::from_slice::<Agent>(pickle_bytes, Default::default())?;
                self.agents.push(agent_object);
            }
        }

        Ok(())
    }

    /// Add an Agent to the Agentset.
    fn add(&mut self, agent: Agent) -> u64 {
        self.agents.push(agent);
        let agents_len = self.agents.len();
        self.agents[agents_len].index = u64::value_from(agents_len).unwrap();
        return self.agents[agents_len].index;
    }

    /// Iterate over the AgentSet and update the current Agent object index values.
    fn update_indices(&mut self) {
        for (idx, agent) in self.agents.iter_mut().enumerate() {
            agent.index = u64::value_from(idx).unwrap();
        }
    }

    /// Removes an Agent from the AgentSet which matches the input Agent; does not return an error if the Agent does not exist.
    fn discard(&mut self, agent: &Agent) -> bool {
        for (idx, other_agent) in self.agents.iter().enumerate() {
            if agent == other_agent {
                self.agents.remove(idx);

                self.update_indices();
                return true;
            }
        }
        return false;
    }

    /// Returns the Agent object at the given index in the AgentSet.
    fn agent_at_index(&mut self, index: usize) -> Option<&Agent> {
        let agent: Option<&Agent> = self.agents.get(index);
        return agent;
    }
}
