use crate::agents::agents;
use crate::graphs::graphs::{GraphEdge, GraphNode};

use petgraph::graph::DiGraph;

use quick_xml::Error as XmlError;
use quick_xml::events::{BytesEnd, BytesStart, BytesText, Event};
use quick_xml::{Reader, Writer};

use rand::prelude;

use std::fs;
use std::io;
use std::num::{ParseFloatError, ParseIntError};
use std::str::ParseBoolError;
use std::sync::Arc;

/// This utils module should cover any miscellaneous functions that facilitate running the package across modules.

/// A utility enum to define various custom error types.
pub enum Error {
    Xml(String),
    ParseValue(String),
    NotFound(String),
    Unsupported(String),
    InvalidDoc(String),
    IO(String),
    ValueError(String),
}

impl From<XmlError> for Error {
    #[inline]
    fn from(e: XmlError) -> Error {
        Error::Xml(format!("Xml document not well-formed: {e}"))
    }
}

impl From<ParseBoolError> for Error {
    #[inline]
    fn from(e: ParseBoolError) -> Error {
        Error::ParseValue(format!("Failed conversion to 'bool': {e}"))
    }
}

impl From<ParseIntError> for Error {
    #[inline]
    fn from(e: ParseIntError) -> Error {
        Error::ParseValue(format!("Failed conversion to 'int' or 'long': {e}"))
    }
}

impl From<ParseFloatError> for Error {
    #[inline]
    fn from(e: ParseFloatError) -> Error {
        Error::ParseValue(format!("Failed conversion to 'float' or 'double': {e}"))
    }
}

impl From<std::io::Error> for Error {
    #[inline]
    fn from(e: std::io::Error) -> Error {
        Error::IO(format!("Input/output error: {e}"))
    }
}

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
