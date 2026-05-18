use crate::agents::agents::Agent;
use conv::*;
use rand::prelude::*;
use rand::rand_core::utils::next_word_via_fill;
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
    index: u64,
    agent: &'a Agent,
}

impl<'a> GraphNode<'a> {
    /// A function to create a new GraphNode object.
    fn new(agent: &'a Agent, index: Option<u64>) -> Self {
        GraphNode {
            index: index.unwrap_or(0), // An index of 0 is not valid and is used as a null value
            agent: agent,
        }
    }

    /// A setter method to set the GraphNode's index value.
    fn set_index(&mut self, index: u64) {
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
    index: u64,
    weighting: f64,
    from_node: u64,
    to_node: u64,
    hierarchy: &'b str,
    rw_params: (f64, f64),
}

impl<'b> GraphEdge<'b> {} // TODO: CONTINUE FROM HERE...
