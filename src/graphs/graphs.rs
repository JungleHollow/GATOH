use crate::agents::agents::Agent;
use conv::*;
use rand::prelude::*;
use rand::rand_core::utils::next_word_via_fill;
use rustworkx_core::petgraph::graph::DiGraph;
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

/// A helper struct that allows rustworkx to more efficiently store information about Agents in the graph nodes.
pub struct GraphNode<'a> {
    index: u32,
    agent: &'a Agent,
}

impl<'a> GraphNode<'a> {
    /// A function to create a new GraphNode object.
    fn new(agent: &'a Agent, index: Option<u32>) -> Self {
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

impl<'a> fmt::Display for GraphNode<'a> {
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
pub struct GraphEdge<'b> {
    index: u32,
    weighting: f32,
    from_node: u32,
    to_node: u32,
    hierarchy: &'b str,
    rw_params: (f32, f32),
}

impl<'b> GraphEdge<'b> {
    /// A constructor function to create a new GraphEdge.
    fn new(
        hierarchy: &'b str,
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
            hierarchy: hierarchy,
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

impl<'b> fmt::Display for GraphEdge<'b> {
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

pub struct Graph<'c> {
    // TODO: DEFINE THIS STRUCT...
}
