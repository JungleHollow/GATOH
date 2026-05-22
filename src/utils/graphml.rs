use indexmap::map::Entry;

use core::any::Any;
use std::borrow::{Borrow, Cow};
use std::collections::HashSet;
use std::convert::From;
use std::ffi::OsStr;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter};
use std::iter::{FromIterator, Iterator};
use std::path::Path;

use flate2::Compression;
use flate2::bufread::GzDecoder;
use flate2::write::GzEncoder;

use crate::utils::utils::Error;

use quick_xml::events::{BytesDecl, BytesEnd, BytesStart, BytesText, Event};
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

    fn to_id(&self) -> Result<&str, Error> {
        match self {
            Value::String(value_str) => Ok(value_str),
            _ => Err(Error::ValueError(String::from(
                "Expected string value for id",
            ))),
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

    fn insert(
        &mut self,
        index: Index,
        weight: Option<DictMap<String, Value>>,
    ) -> Result<(), Error> {
        // TODO: KEEP AN EYE ON THIS FUNCTION...
        let element_info = weight
            .and_then(
                |mut data: DictMap<String, Value>| -> Option<Result<GraphElementInfo, Error>> {
                    let id = data
                        .shift_remove_entry("id")
                        .map(|(id, value)| -> Result<Option<String>, Error> {
                            let value_str = value.to_id()?;
                            if self.id_taken.contains(value_str) {
                                data.insert(id, value);
                                Ok(None)
                            } else {
                                self.id_taken.insert(value_str.to_string());
                                Ok(Some(value_str.to_string()))
                            }
                        })
                        .unwrap_or_else(|| Ok(None))
                        .ok()?;
                    Some(Ok(GraphElementInfo {
                        attributes: data.clone(),
                        id,
                    }))
                },
            )
            .unwrap_or_else(|| Ok(GraphElementInfo::default()))?;
        self.vec.push((index, element_info));
        Ok(())
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

    fn last_graph_set_attribute(&mut self, key: &str, val: String) -> Result<(), Error> {
        let key = match self.key_for_all.get(key) {
            Some(key) => key,
            None => self
                .key_for_graph
                .get(key)
                .ok_or_else(|| Error::NotFound(format!("Key '{key}' for graph not found.")))?,
        };

        if let Some(graph) = self.graphs.last_mut() {
            graph.attributes.insert(key.name.clone(), key.parse(val)?);
        }

        Ok(())
    }

    /// Open file compressed with gzip using GzDecoder
    /// Returns a quick_xml Reader instance
    fn open_file_gzip<P: AsRef<Path>>(
        path: P,
    ) -> Result<Reader<BufReader<GzDecoder<BufReader<File>>>>, quick_xml::Error> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let gzip_reader = BufReader::new(GzDecoder::new(reader));
        Ok(Reader::from_reader(gzip_reader))
    }

    /// Parse a file written in GraphML format from a BufReader
    ///
    /// The implementation is based on a state machine in order to
    /// accept only valid GraphML syntax (e.g. a '<data>' element should
    /// be nested inside a '<node>' element) where the internal state changes
    /// after handling each quick_xml event.
    fn read_graph_from_reader<R: BufRead>(mut reader: Reader<R>) -> Result<GraphML, Error> {
        let mut graphml = GraphML::default();

        let mut buf = Vec::new();
        let mut state = State::Start;
        let mut domain_of_last_key = Domain::Node;
        let mut last_data_key = String::new();

        loop {
            match reader.read_event_into(&mut buf)? {
                Event::Start(ref e) => match e.name() {
                    QName(b"key") => {
                        matches!(state, State::Start);
                        domain_of_last_key = graphml.add_graphml_key(e)?;
                        state = State::Key;
                    }
                    QName(b"default") => {
                        matches!(state, State::Key);
                        state = State::DefaultForKey;
                    }
                    QName(b"graph") => {
                        matches!(state, State::Start);
                        graphml.create_graph(e)?;
                        state = State::Graph;
                    }
                    QName(b"node") => {
                        matches!(state, State::Graph);
                        graphml.add_node(e)?;
                        state = State::Node;
                    }
                    QName(b"edge") => {
                        matches! {state, State::Graph};
                        graphml.add_edge(e)?;
                        state = State::Edge;
                    }
                    QName(b"data") => {
                        matches!(state, State::Node | State::Edge | State::Graph);
                        last_data_key = xml_attribute(e, b"key")?;
                        match state {
                            State::Node => state = State::DataForNode,
                            State::Edge => state = State::DataForEdge,
                            State::Graph => state = State::DataForGraph,
                            _ => {
                                // In all other cases we have already bailed out in `matches`.
                                unreachable!()
                            }
                        }
                    }
                    QName(b"hyperedge") => {
                        return Err(Error::Unsupported(String::from(
                            "Hyperedges are not supported.",
                        )));
                    }
                    QName(b"port") => {
                        return Err(Error::Unsupported(String::from("Ports are not supported.")));
                    }
                    _ => {}
                },
                Event::Empty(ref e) => match e.name() {
                    QName(b"key") => {
                        matches!(state, State::Start);
                        graphml.add_graphml_key(e)?;
                    }
                    QName(b"node") => {
                        matches!(state, State::Graph);
                        graphml.add_node(e)?;
                    }
                    QName(b"edge") => {
                        matches!(state, State::Graph);
                        graphml.add_edge(e)?;
                    }
                    QName(b"port") => {
                        return Err(Error::Unsupported(String::from("Ports are not supported.")));
                    }
                    _ => {}
                },
                Event::End(ref e) => match e.name() {
                    QName(b"key") => {
                        matches!(state, State::Key);
                        state = State::Start;
                    }
                    QName(b"default") => {
                        matches!(state, State::DefaultForKey);
                        state = State::Key;
                    }
                    QName(b"graph") => {
                        matches!(state, State::Graph);
                        state = State::Start;
                    }
                    QName(b"node") => {
                        matches!(state, State::Node);
                        state = State::Graph;
                    }
                    QName(b"edge") => {
                        matches!(state, State::Edge);
                        state = State::Graph;
                    }
                    QName(b"data") => {
                        matches!(
                            state,
                            State::DataForNode | State::DataForEdge | State::DataForGraph
                        );
                        match state {
                            State::DataForNode => state = State::Node,
                            State::DataForEdge => state = State::Edge,
                            State::DataForGraph => state = State::Graph,
                            _ => {
                                // In all other cases we have already bailed out in `matches`
                                unreachable!()
                            }
                        }
                    }
                    _ => {}
                },
                Event::Text(ref e) => match state {
                    State::DefaultForKey => {
                        graphml
                            .last_key_set_value((e.unescape()?).to_string(), domain_of_last_key)?;
                    }
                    State::DataForNode => {
                        graphml.last_node_set_data(&last_data_key, (e.unescape()?).to_string())?;
                    }
                    State::DataForEdge => {
                        graphml.last_edge_set_data(&last_data_key, (e.unescape()?).to_string())?;
                    }
                    State::DataForGraph => {
                        graphml.last_graph_set_attribute(
                            &last_data_key,
                            (e.unescape()?).to_string(),
                        )?;
                    }
                    _ => {}
                },
                Event::Eof => {
                    break;
                }
                _ => {}
            }

            buf.clear();
        }

        Ok(graphml)
    }

    /// Read a graph from a file in the GraphML format
    /// If the file extension is `graphmlz` or `gz`, decompress it on the fly
    fn from_file<P: AsRef<Path>>(path: P, compression: &str) -> Result<GraphML, Error> {
        let extension = path.as_ref().extension().unwrap_or(OsStr::new(""));

        let graph: Result<GraphML, Error> =
            if extension.eq("graphmlz") || extension.eq("gz") || compression.eq("gzip") {
                let reader = Self::open_file_gzip(path)?;
                Self::read_graph_from_reader(reader)
            } else {
                let reader = Reader::from_file(path)?;
                Self::read_graph_from_reader(reader)
            };

        graph
    }

    fn write_data<W: std::io::Write>(
        writer: &mut Writer<W>,
        keys: &DictMap<String, (&String, &Key)>,
        data: &DictMap<String, Value>,
    ) -> Result<(), Error> {
        for (key_name, value) in data {
            let (id, key) = keys
                .get(key_name)
                .ok_or_else(|| Error::NotFound(format!("Unknown key {key_name}")))?;
            if key.default == *value {
                continue;
            }

            let mut elem = BytesStart::new("data");
            elem.push_attribute(("key", id.as_str()));
            writer.write_event(Event::Start(elem.borrow()))?;
            if let Some(contents) = value.serialize() {
                writer.write_event(Event::Text(BytesText::new(contents.borrow())))?;
            }
            writer.write_event(Event::End(elem.to_end()))?;
        }
        Ok(())
    }

    fn write_elem_data<W: std::io::Write>(
        writer: &mut Writer<W>,
        keys: &DictMap<String, (&String, &Key)>,
        elem: BytesStart,
        data: &DictMap<String, Value>,
    ) -> Result<(), Error> {
        if data.is_empty() {
            writer.write_event(Event::Empty(elem))?;
            return Ok(());
        }

        writer.write_event(Event::Start(elem.borrow()))?;
        Self::write_data(writer, keys, data)?;
        writer.write_event(Event::End(elem.to_end()))?;
        Ok(())
    }

    fn write_keys<W: std::io::Write>(
        writer: &mut Writer<W>,
        key_for: &str,
        map: &DictMap<String, Key>,
    ) -> Result<(), quick_xml::Error> {
        for (id, key) in map {
            let mut elem = BytesStart::new("key");
            elem.push_attribute(("id", id.as_str()));
            elem.push_attribute(("for", key_for));
            elem.push_attribute(("attr.name", key.name.as_str()));
            let ty: &str = key.ty.into();
            elem.push_attribute(("attr.type", ty));
            writer.write_event(Event::Start(elem.borrow()))?;
            if let Some(contents) = key.default.serialize() {
                let elem = BytesStart::new("default");
                writer.write_event(Event::Start(elem.borrow()))?;
                writer.write_event(Event::Text(BytesText::new(contents.borrow())))?;
                writer.write_event(Event::End(elem.to_end()))?;
            };
            writer.write_event(Event::End(elem.to_end()))?;
        }
        Ok(())
    }

    fn write_graph_to_writer<W: std::io::Write>(
        &self,
        writer: &mut Writer<W>,
    ) -> Result<(), Error> {
        writer.write_event(Event::Decl(BytesDecl::new("1.0", Some("UTF-8"), None)))?;

        let mut elem = BytesStart::new("graphml");
        elem.push_attribute(("xmlns", "http://graphml.graphdrawing.org/xmlns"));
        elem.push_attribute(("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance"));
        elem.push_attribute(("xsi:schemaLocation", "http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd"));

        writer.write_event(Event::Start(elem.borrow()))?;

        Self::write_keys(writer, "node", &self.key_for_nodes)?;
        Self::write_keys(writer, "edge", &self.key_for_edges)?;
        Self::write_keys(writer, "graph", &self.key_for_graph)?;
        Self::write_keys(writer, "all", &self.key_for_all)?;

        let graph_keys: DictMap<String, (&String, &Key)> =
            build_key_name_map(&self.key_for_graph, &self.key_for_all);
        let node_keys: DictMap<String, (&String, &Key)> =
            build_key_name_map(&self.key_for_nodes, &self.key_for_all);
        let edge_keys: DictMap<String, (&String, &Key)> =
            build_key_name_map(&self.key_for_edges, &self.key_for_all);

        for graph in self.graphs.iter() {
            let mut elem = BytesStart::new("graph");
            if let Some(id) = &graph.id {
                elem.push_attribute(("id", id.as_str()));
            }
            let edgedefault = match graph.dir {
                Direction::Directed => "directed",
                Direction::Undirected => "undirected",
            };
            elem.push_attribute(("edgedefault", edgedefault));
            writer.write_event(Event::Start(elem.borrow()))?;
            Self::write_data(writer, &graph_keys, &graph.attributes)?;
            for node in &graph.nodes {
                let mut elem = BytesStart::new("node");
                elem.push_attribute(("id", node.id.as_str()));
                Self::write_elem_data(writer, &node_keys, elem, &node.data)?;
            }
            for edge in &graph.edges {
                let mut elem = BytesStart::new("edge");
                if let Some(id) = &edge.id {
                    elem.push_attribute(("id", id.as_str()));
                }
                elem.push_attribute(("source", edge.source.as_str()));
                elem.push_attribute(("target", edge.target.as_str()));
                Self::write_elem_data(writer, &edge_keys, elem, &edge.data)?;
            }
            writer.write_event(Event::End(elem.to_end()))?;
        }
        writer.write_event(Event::End(elem.to_end()))?;
        Ok(())
    }

    fn to_file(&self, path: impl AsRef<Path>, compression: &str) -> Result<(), Error> {
        let extension = path.as_ref().extension().unwrap_or(OsStr::new(""));
        if extension.eq("graphmlz") || extension.eq("gz") || compression.eq("gzip") {
            let file = File::create(path)?;
            let buf_writer = BufWriter::new(file);
            let gzip_encoder = GzEncoder::new(buf_writer, Compression::default());
            let mut writer = Writer::new(gzip_encoder);
            self.write_graph_to_writer(&mut writer)?;
            writer.into_inner().finish()?;
        } else {
            let file = File::create(path)?;
            let mut writer = Writer::new(file);
            self.write_graph_to_writer(&mut writer)?;
        }
        Ok(())
    }

    fn infer_keys(&mut self) -> Result<(), Error> {
        infer_keys_for_attributes(
            &mut self.key_for_graph,
            self.graphs.iter().flat_map(|graph| graph.attributes.iter()),
        )?;
        infer_keys_for_attributes(
            &mut self.key_for_nodes,
            self.graphs
                .iter()
                .flat_map(|graph| graph.nodes.iter())
                .flat_map(|nodes| nodes.data.iter()),
        )?;
        infer_keys_for_attributes(
            &mut self.key_for_edges,
            self.graphs
                .iter()
                .flat_map(|graph| graph.edges.iter())
                .flat_map(|edges| edges.data.iter()),
        )?;
        Ok(())
    }
}

pub struct KeySpec {
    id: String,
    domain: Domain,
    name: String,
    ty: Type,
    default: Value,
}

impl KeySpec {
    fn new(id: String, domain: Domain, name: String, ty: Type, default: Value) -> Self {
        KeySpec {
            id,
            domain,
            name,
            ty,
            default,
        }
    }
}
