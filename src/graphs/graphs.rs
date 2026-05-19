use crate::agents::agents::Agent;
use conv::*;
use rand::prelude::*;
use rand::rand_core::utils::next_word_via_fill;
use rustworkx_core;
use rustworkx_core::petgraph::graph::DiGraph; // Explicit import to simplify calls
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
use graphrs::GraphSpecs;
use graphrs::readwrite::graphml::*; // To allow for reading and writing Graph objects to the GraphML format

/// A helper struct that allows rustworkx to more efficiently store information about Agents in the graph nodes.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GraphNode {
    index: u32,
    agent: Agent,
}

impl GraphNode {
    /// A function to create a new GraphNode object.
    fn new(agent: Agent, index: Option<u32>) -> Self {
        GraphNode {
            index: index.unwrap_or(0), // An index of 0 is not valid and is used as a null value
            agent: agent,
        }
    }

    /// A setter method to set the GraphNode's index value.
    fn set_index(&mut self, index: u32) {
        self.index = index;
    }
}

impl fmt::Display for GraphNode {
    /// An override of how a GraphNode object's representation will be printed out.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            format!("Agent ({}) at graph node ({})", self.agent.id, self.index)
        )
    }
}

/// A helper struct that allows rustworkx to more efficiently store information about Agent relationships in the graph edges.
/// As the social hierarchies are assumed to be DiGraphs, each GraphEdge is directional, and the social weighting that Agent
/// A places on Agent B will not necessarilly be equally reciprocated.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GraphEdge {
    index: u32,
    weighting: f32,
    from_node: u32,
    to_node: u32,
    hierarchy: String,
    rw_params: (f32, f32),
}

impl GraphEdge {
    /// A constructor function to create a new GraphEdge.
    fn new(
        hierarchy: &str,
        from_node: u32,
        to_node: u32,
        weighting: Option<f32>,
        rw_params: Option<(f32, f32)>,
    ) -> Self {
        GraphEdge {
            index: 0, // An index of 0 is not valid and is treated as a null value
            weighting: weighting.unwrap_or(0.0),
            from_node: from_node,
            to_node: to_node,
            hierarchy: String::from(hierarchy),
            rw_params: rw_params.unwrap_or((-99.9, 0.1)), // An rw mean of -99.9 is not valid and is treated as a null value
        }
    }

    /// A setter function that changes this GraphEdge's index value.
    fn set_index(&mut self, index: u32) {
        self.index = index;
    }

    /// A setter function that changes this GraphEdge's weighting value.
    fn set_weighting(&mut self, weighting: f32) {
        self.weighting = weighting;
    }

    /// A setter function that changes this GraphEdge's rw_params value.
    fn set_rw_params(&mut self, rw_params: (f32, f32)) {
        self.rw_params = rw_params;
    }

    /// A setter function that updates the from_node's index for this GraphEdge.
    fn update_from_node(&mut self, index: u32) {
        self.from_node = index;
    }

    /// A setter function that updates the to_node's index for this GraphEdge.
    fn update_to_node(&mut self, index: u32) {
        self.to_node = index;
    }

    /// A function that checks if this relationship has explicit random walk parameters
    fn has_rw_params(&mut self) -> bool {
        if self.rw_params.0 == -99.9 {
            // The null value assigned at initialisation
            return false;
        } else {
            return true;
        }
    }
}

impl fmt::Display for GraphEdge {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            format!(
                "GraphEdge of weight ({}) from node ({}) to node ({}) in the {} social layer",
                self.weighting, self.from_node, self.to_node, self.hierarchy
            )
        )
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
enum GraphGenerationParams {
    FloatParam(f32),
    IntParam(i32),
    StringParam(String),
    BoolParam(bool),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Graph {
    graph: DiGraph<GraphNode, GraphEdge>,
    node_count: u32,
    edge_count: u32,
    name: String,
    rw_params: (f32, f32),
    generation_params: HashMap<String, GraphGenerationParams>,
}

impl Graph {
    /// A constructor method that creates a new Graph object.
    fn new(name: &str, rw_params: (f32, f32)) -> Self {
        Graph {
            graph: DiGraph::new(),
            node_count: 0,
            edge_count: 0,
            name: String::from(name),
            rw_params: rw_params,
            generation_params: HashMap::from([
                (String::from("p"), GraphGenerationParams::FloatParam(0.4)),
                (String::from("m"), GraphGenerationParams::IntParam(3)),
                (
                    String::from("sbm_sizes"),
                    GraphGenerationParams::IntParam(10),
                ),
            ]),
        }
    }

    /// Setter function which outlines the existing generation parameters used in generate_graph()
    /// and allows the user to alter them.
    fn change_generation_params(&mut self, params: HashMap<String, GraphGenerationParams>) {
        for (key, value) in params.iter() {
            if !self.generation_params.contains_key(key) {
                println!(format!("WARNING: Invalid graph generation parameter ({}) specified when trying to modify parameter values.", key));
                continue;
            } else if !variant_eq(value, self.generation_params.get(key)) {
                println!(format!("WARNING: Invalid data type detected for the value when modifying parameter {}.", key));
                continue;
            } else {
                self.generation_params.insert(*key, *value);
            }
        }
    }

    /// Loads a Graph object stored in the GraphML format from the given path.
    /// The social hierarchy name must be explicitly passed with this call.
    fn load_graph(&mut self, path: &str, name: &str, rw_params: Option<(f32, f32)>) {
        let graph: DiGraph<GraphNode, GraphEdge> = ; //TODO: FINISH THIS FUNCTION...
    }
}
