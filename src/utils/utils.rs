use crate::{
    agents::agents,
    graphs::graphs::{GraphEdge, GraphNode},
};
use petgraph::graph::DiGraph;
use quick_xml::{
    Reader, Writer,
    events::{BytesEnd, BytesStart, BytesText, Event},
};
use rand::prelude;
use std::fs;
use std::io;
use std::sync::Arc;

/// This utils module should cover any miscellaneous functions that facilitate running the package across modules.

// ========== Graph utils ========== //

// ========== Math utils ========== //

// ========== Random generation utils ========== //

// ========== Random walk utils ========== //

// ========== Data persistence utils ========== //

pub fn read_graphml_file(path: &str) -> Result<DiGraph<GraphNode, GraphEdge>, io::Error> {
    let graph_string =
        fs::read_to_string(path).expect("Could not open the file at the specified path");
    read_graphml_string(&graph_string)
}

pub fn read_graphml_string(string: &str) -> Result<DiGraph<GraphNode, GraphEdge>, io::Error> {
    let mut reader = Reader::from_str(string);
    let mut buf = Vec::new();
    let mut directed: bool = true;
    let mut nodes: Vec<Arc<GraphNode>> = vec![];
    let mut edges: Vec<Arc<GraphEdge>> = vec![];
    let mut last_element_name: String = String::new();
    let mut edge_weight_attr_name = String::from("weight");
    // TODO: CONTINUE FROM HERE...
}

// ========== Visualisation and graphing utils ========== //
