use indexmap::map::Entry;
use std::borrow::Cow;
use std::collections::HashSet;

use crate::utils::utils::Error;

use quick_xml::events::{BytesEnd, BytesStart, BytesText, Event};
use quick_xml::name::QName;
use quick_xml::{Reader, Writer};
use rustworkx_core::dictmap::{DictMap, InitWithHasher};

/// The code in this file is closely adapted from the `graphml.rs' file present in rustworkx-core, which is not normally
/// accessible through the library.

/// Extract an xml attribute from an input xml doc string.
fn xml_attribute<'a>(element: &'a BytesStart<'a>, key: &[u8]) -> Result<String, Error> {
    element
        .attributes()
        .find_map(|a| {
            if let Ok(a) = a {
                if a.key == QName(key) {
                    let decoded = a
                        .unescape_value()
                        .map_err(Error::from)
                        .map(|cow_str| cow_str.into_owned());
                    return Some(decoded);
                }
            }
            None
        })
        .unwrap_or_else(|| {
            Err(Error::NotFound(format!(
                "Attribute '{}' not found.",
                String::from_utf8_lossy(key)
            )))
        })
}

#[derive(Clone, Copy, PartialEq)]
pub enum Domain {
    Node,
    Edge,
    Graph,
    All,
}

impl TryFrom<&[u8]> for Domain {
    type Error = ();

    fn try_from(value: &[u8]) -> Result<Self, ()> {
        match value {
            b"node" => Ok(Domain::Node),
            b"edge" => Ok(Domain::Edge),
            b"graph" => Ok(Domain::Graph),
            b"all" => Ok(Domain::All),
            _ => Err(()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Type {
    Boolean,
    Int,
    Float,
    Double,
    String,
    Long,
}

impl From<Type> for &'static str {
    fn from(ty: Type) -> Self {
        match ty {
            Type::Boolean => "boolean",
            Type::Int => "int",
            Type::Float => "float",
            Type::Double => "double",
            Type::String => "string",
            Type::Long => "long",
        }
    }
}

#[derive(Clone, PartialEq)]
enum Value {
    Boolean(bool),
    Int(isize),
    Float(f32),
    Double(f64),
    String(String),
    Long(isize),
    UnDefined,
}

impl Value {
    fn serialize(&self) -> Option<Cow<'_, str>> {
        match self {
            Value::Boolean(val) => Some(Cow::from(val.to_string())),
            Value::Int(val) => Some(Cow::from(val.to_string())),
            Value::Float(val) => Some(Cow::from(val.to_string())),
            Value::Double(val) => Some(Cow::from(val.to_string())),
            Value::String(val) => Some(Cow::from(val.to_string())),
            Value::Long(val) => Some(Cow::from(val.to_string())),
            Value::UnDefined => None,
        }
    }

    fn ty(&self) -> Option<Type> {
        match self {
            Value::Boolean(_) => Some(Type::Boolean),
            Value::Int(_) => Some(Type::Int),
            Value::Float(_) => Some(Type::Float),
            Value::Double(_) => Some(Type::Double),
            Value::String(_) => Some(Type::String),
            Value::Long(_) => Some(Type::Long),
            Value::UnDefined => None,
        }
    }
}

struct Key {
    name: String,
    ty: Type,
    default: Value,
}

impl Key {
    fn parse(&self, val: String) -> Result<Value, Error> {
        Ok(match self.ty {
            Type::Boolean => Value::Boolean(val.parse()?),
            Type::Int => Value::Int(val.parse()?),
            Type::Float => Value::Float(val.parse()?),
            Type::Double => Value::Double(val.parse()?),
            Type::String => Value::String(val),
            Type::Long => Value::Long(val.parse()?),
        })
    }

    fn set_value(&mut self, val: String) -> Result<(), Error> {
        self.default = self.parse(val)?;
        Ok(())
    }
}

struct Node {
    id: String,
    data: DictMap<String, Value>,
}

struct Edge {
    id: Option<String>,
    source: String,
    target: String,
    data: DictMap<String, Value>,
}

enum Direction {
    Directed,
    Undirected,
}

struct Graph {
    id: Option<String>,
    dir: Direction,
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    attributes: DictMap<String, Value>,
}

impl Graph {
    fn new<'a, I>(id: Option<String>, dir: Direction, default_attrs: I) -> Self
    where
        I: Iterator<Item = &'a Key>,
    {
        Self {
            id,
            dir,
            nodes: Vec::new(),
            edges: Vec::new(),
            attributes: DictMap::from_iter(
                default_attrs.map(|key| (key.name.clone(), key.default.clone())),
            ),
        }
    }

    fn add_node<'a, I>(&mut self, element: &'a BytesStart<'a>, default_data: I) -> Result<(), Error>
    where
        I: Iterator<Item = &'a Key>,
    {
        self.nodes.push(Node {
            id: xml_attribute(element, b"id")?,
            data: DictMap::from_iter(
                default_data.map(|key| (key.name.clone(), key.default.clone())),
            ),
        });

        Ok(())
    }

    fn add_edge<'a, I>(&mut self, element: &'a BytesStart<'a>, default_data: I) -> Result<(), Error>
    where
        I: Iterator<Item = &'a Key>,
    {
        self.edges.push(Edge {
            id: xml_attribute(element, b"id").ok(),
            source: xml_attribute(element, b"source")?,
            target: xml_attribute(element, b"target")?,
            data: DictMap::from_iter(
                default_data.map(|key| (key.name.clone(), key.default.clone())),
            ),
        });

        Ok(())
    }

    fn last_node_set_data(&mut self, key: &Key, val: String) -> Result<(), Error> {
        if let Some(node) = self.nodes.last_mut() {
            node.data.insert(key.name.clone(), key.parse(val)?);
        }

        Ok(())
    }

    fn last_edge_set_data(&mut self, key: &Key, val: String) -> Result<(), Error> {
        if let Some(edge) = self.edges.last_mut() {
            edge.data.insert(key.name.clone(), key.parse(val)?);
        }

        Ok(())
    }
}

struct GraphElementInfo {
    attributes: DictMap<String, Value>,
    id: Option<String>,
}

impl Default for GraphElementInfo {
    fn default() -> Self {
        Self {
            attributes: DictMap::new(),
            id: None,
        }
    }
}

struct GraphElementInfos<Index> {
    vec: Vec<(Index, GraphElementInfo)>,
    id_taken: HashSet<String>,
}

impl<Index: std::cmp::Eq + std::hash::Hash> GraphElementInfos<Index> {
    fn new() -> Self {
        Self {
            vec: vec![],
            id_taken: HashSet::new(),
        }
    }
}

enum State {
    Start,
    Graph,
    Node,
    Edge,
    DataForNode,
    DataForEdge,
    DataForGraph,
    Key,
    DefaultForKey,
}

macro_rules! matches {
    ($expression:expr, $( $pattern:pat_param )|+) => {
        match $expression {
            $( $pattern )|+ => {},
            _ => {
                return Err(Error::InvalidDoc(String::from(
                    "The input xml document doesn't follow the syntax of GraphML language \
                    (or it has features that are not supported by the current version of the parser)."
                )));
            }
        }
    };
}

struct GraphML {
    graphs: Vec<Graph>,
    key_for_nodes: DictMap<String, Key>,
    key_for_edges: DictMap<String, Key>,
    key_for_graph: DictMap<String, Key>,
    key_for_all: DictMap<String, Key>,
}

impl Default for GraphML {
    fn default() -> Self {
        Self {
            graphs: Vec::new(),
            key_for_nodes: DictMap::new(),
            key_for_edges: DictMap::new(),
            key_for_graph: DictMap::new(),
            key_for_all: DictMap::new(),
        }
    }
}

/// Given maps from ids to keys, return a map from key name to ids and keys.
fn build_key_name_map<'a>(
    key_for_items: &'a DictMap<String, Key>,
    key_for_all: &'a DictMap<String, Key>,
) -> DictMap<String, (&'a String, &'a Key)> {
    // `key_for_items` is iterated before `key_for_all` since last
    // items take precedence in the collected map. Similarly,
    // the map `for_all` take precedence over kind-specific maps in
    // `last_node_set_data`, `last_edge_set_data`, and
    // `last_graph_set_attribute`
    key_for_all
        .iter()
        .chain(key_for_items.iter())
        .map(|(id, key)| (key.name.clone(), (id, key)))
        .collect()
}

fn infer_keys_for_attributes<'a>(
    target: &mut DictMap<String, Key>,
    attributes: impl Iterator<Item = (&'a String, &'a Value)>,
) -> Result<(), Error> {
    let mut inferred = DictMap::new();
    let mut counter = 0;
    for (name, value) in attributes {
        if let Some(ty) = value.ty() {
            match inferred.entry(name.clone()) {
                Entry::Vacant(entry) => {
                    counter += 1;
                    let id = format!("d{counter}");
                    entry.insert(id);
                    target.insert(
                        id,
                        Key {
                            name: name.to_string(),
                            ty,
                            default: Value::UnDefined,
                        },
                    );
                }
                Entry::Occupied(entry) => {
                    let other_ty = entry.get();
                    if *other_ty != ty {
                        return Err(Error::InvalidDoc(format!(
                            "Mismatch type for key {name}: {ty:?} and {other_ty:?}"
                        )));
                    }
                }
            }
        }
    }
    Ok(())
}

impl GraphML {
    fn create_graph<'a>(&mut self, element: &'a BytesStart<'a>) -> Result<(), Error> {
        let dir = match xml_attribute(element, b"edgedefault")?.as_bytes() {
            b"directed" => Direction::Directed,
            b"undirected" => Direction::Undirected,
            _ => {
                return Err(Error::InvalidDoc(String::from(
                    "Invalid 'edgedefault' attribute.",
                )));
            }
        };

        self.graphs.push(Graph::new(
            xml_attribute(element, b"id").ok(),
            dir,
            self.key_for_graph.values().chain(self.key_for_all.values()),
        ));

        Ok(())
    }

    fn add_node<'a>(&mut self, element: &'a BytesStart<'a>) -> Result<(), Error> {
        if let Some(graph) = self.graphs.last_mut() {
            graph.add_node(
                element,
                self.key_for_nodes.values().chain(self.key_for_all.values()),
            )?;
        }

        Ok(())
    }

    fn add_edge<'a>(&mut self, element: &'a BytesStart) -> Result<(), Error> {
        if let Some(graph) = self.graphs.last_mut() {
            graph.add_edge(
                element,
                self.key_for_edges.values().chain(self.key_for_all.values()),
            )?;
        }

        Ok(())
    }

    fn get_keys_mut(&mut self, domain: Domain) -> &mut DictMap<String, Key> {
        match domain {
            Domain::Node => &mut self.key_for_nodes,
            Domain::Edge => &mut self.key_for_edges,
            Domain::Graph => &mut self.key_for_graph,
            Domain::All => &mut self.key_for_all,
        }
    }

    fn add_graphml_key<'a>(&mut self, element: &'a BytesStart<'a>) -> Result<Domain, Error> {
        let id = xml_attribute(element, b"id")?;
        let ty = match xml_attribute(element, b"attr.type")?.as_bytes() {
            b"boolean" => Type::Boolean,
            b"int" => Type::Int,
            b"float" => Type::Float,
            b"double" => Type::Double,
            b"string" => Type::String,
            b"long" => Type::Long,
            _ => {
                return Err(Error::InvalidDoc(format!(
                    "Invalid 'attr.type' attribute in key with id={id}.",
                )));
            }
        };

        let key = Key {
            name: xml_attribute(element, b"attr.name")?,
            ty,
            default: Value::UnDefined,
        };

        let domain: Domain = xml_attribute(element, b"for")?
            .as_bytes()
            .try_into()
            .map_err(|()| {
                Error::InvalidDoc(format!("Invalid 'for' attribute in key with id={id}."))
            })?;

        self.get_keys_mut(domain).insert(id, key);
        Ok(domain)
    }

    fn last_key_set_value(&mut self, val: String, domain: Domain) -> Result<(), Error> {
        let elem = self.get_keys_mut(domain).last_mut();

        if let Some((_, key)) = elem {
            key.set_value(val)?;
        }

        Ok(())
    }

    fn last_node_set_data(&mut self, key: &str, val: String) -> Result<(), Error> {
        let key = match self.key_for_all.get(key) {
            Some(key) => key,
            None => self
                .key_for_nodes
                .get(key)
                .ok_or_else(|| Error::NotFound(format!("Key {key} for nodes not found.")))?,
        };

        if let Some(graph) = self.graphs.last_mut() {
            graph.last_node_set_data(key, val)?;
        }

        Ok(())
    }

    fn last_edge_set_data(&mut self, key: &str, val: String) -> Result<(), Error> {
        let key = match self.key_for_all.get(key) {
            Some(key) => key,
            None => self
                .key_for_edges
                .get(key)
                .ok_or_else(|| Error::NotFound(format!("Key {key} for edges not found.")))?,
        };

        if let Some(graph) = self.graphs.last_mut() {
            graph.last_edge_set_data(key, val)?;
        }

        Ok(())
    }
}
