// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

/// Header for the internal IR of the translator.

#ifndef SDFG_Translation_Node_H
#define SDFG_Translation_Node_H

#include "SDFG/Dialect/Dialect.h"
#include "SDFG/Translate/JsonEmitter.h"
#include "SDFG/Utils/Utils.h"
#include "mlir/IR/Location.h"

namespace mlir::sdfg::translation {
// Forward declarations.
class Emittable;

class Attribute;
class Condition;
class Assignment;
class Array;

class Node;
class NodeImpl;

class ConnectorNode;
class ConnectorNodeImpl;
class Connector;

class ScopeNode;
class ScopeNodeImpl;

class State;
class StateImpl;

class SDFG;
class SDFGImpl;

class NestedSDFG;
class NestedSDFGImpl;

class InterstateEdge;
class InterstateEdgeImpl;
class MultiEdge;

class Tasklet;
class TaskletImpl;

class Library;
class LibraryImpl;

class Access;
class AccessImpl;

class MapEntry;
class MapEntryImpl;
class MapExit;
class MapExitImpl;

class ConsumeEntry;
class ConsumeEntryImpl;
class ConsumeExit;
class ConsumeExitImpl;

//===----------------------------------------------------------------------===//
// Interfaces
//===----------------------------------------------------------------------===//

/// All classes that can be printed (emitted) implement this interface.
class Emittable {
public:
  virtual void emit(emitter::JsonEmitter &jemit) = 0;
};

//===----------------------------------------------------------------------===//
// DataClasses
//===----------------------------------------------------------------------===//

/// DaCe Datatypes.
enum class DType {
  null,
  boolean,
  int8,
  int16,
  int32,
  int64,
  float16,
  float32,
  float64
};

/// Node Types.
enum class NType { SDFG, State, Access, MapEntry, ConsumeEntry, Other };

/// Programming languages.
enum class CodeLanguage { Python, CPP, MLIR };

/// Stores an attribute for a node.
class Attribute {
public:
  std::string name;
  // Store attribute or string?
};

/// Stores a symbol with symbol name and data type.
class Symbol {
public:
  std::string name;
  DType type;
  Symbol(StringRef name, DType type) : name(name), type(type) {}
};

/// Represents a condition for an edge.
class Condition {
public:
  std::string condition;

  Condition(StringRef condition) : condition(condition) {}
};

/// Represents an assignment for an edge.
class Assignment {
public:
  std::string key;
  std::string value;

  Assignment(StringRef key, StringRef value) : key(key), value(value) {}
};

/// Stores code for tasklets with the associated programming language.
class Code {
public:
  std::string data;
  CodeLanguage language;

  Code() : data(""), language(CodeLanguage::Python) {}
  Code(StringRef data, CodeLanguage language)
      : data(data), language(language) {}
};

/// Represents a DaCe data container.
class Array : public Emittable {
public:
  std::string name;
  bool transient;
  bool stream;
  SizedType shape;

  Array(StringRef name, bool transient, bool stream, Type t)
      : name(name), transient(transient), stream(stream),
        shape(SizedType::get(t.getContext(), t, {}, {}, {})) {}

  Array(StringRef name, bool transient, bool stream, SizedType shape)
      : name(name), transient(transient), stream(stream), shape(shape) {}

  /// Emits this array to the output stream.
  void emit(emitter::JsonEmitter &jemit) override;
};

/// Represents a range for memlets.
class Range : public Emittable {
public:
  std::string start;
  std::string end;
  std::string step;
  std::string tile;

  Range(StringRef start, StringRef end, StringRef step, StringRef tile)
      : start(start), end(end), step(step), tile(tile) {}

  /// Emits this range to the output stream.
  void emit(emitter::JsonEmitter &jemit) override;
};

//===----------------------------------------------------------------------===//
// Node
//===----------------------------------------------------------------------===//

/// Base class for all SDFG nodes.
class Node : public Emittable {
protected:
  /// Pointer to the implementation (Pimpl idiom).
  std::shared_ptr<NodeImpl> ptr;
  /// Stores the type of this node.
  NType type;

public:
  Node(std::shared_ptr<NodeImpl> ptr) : ptr(ptr), type(NType::Other) {}

  bool operator==(const Node other) const { return other.ptr == ptr; }

  /// Sets the ID of the node.
  void setID(unsigned id);
  /// Returns the ID of the node.
  unsigned getID();

  /// Returns the source code location.
  Location getLocation();
  /// Returns the type of the node.
  NType getType();

  /// Sets the name of the node.
  void setName(StringRef name);
  /// Returns the name of the node.
  StringRef getName();

  /// Sets the parent of the node.
  void setParent(Node parent);
  /// Returns the parent of the node.
  Node getParent();
  /// Return true if this node has a parent node.
  bool hasParent();

  /// Returns the top-level SDFG.
  virtual SDFG getSDFG();
  /// Returns the surrounding state.
  virtual State getState();

  /// Adds an attribute to this node, replaces existing attributes with the same
  /// name.
  void addAttribute(Attribute attribute);
  /// Emits this node to the output stream.
  void emit(emitter::JsonEmitter &jemit) override;
};

/// Implementation of the base node class.
class NodeImpl : public Emittable {
protected:
  /// Unique node ID.
  unsigned id;
  /// Source code location.
  Location location;
  /// Name of this node.
  std::string name;
  /// An array of associated attributes.
  std::vector<Attribute> attributes;
  /// Pointer to the parent node.
  Node parent;

public:
  NodeImpl(Location location) : id(0), location(location), parent(nullptr) {}

  /// ID setter.
  void setID(unsigned id);
  /// ID getter.
  unsigned getID();

  /// Source code location getter.
  Location getLocation();

  /// Name setter.
  void setName(StringRef name);
  /// Name getter.
  StringRef getName();

  /// Parent node setter.
  void setParent(Node parent);
  /// Parent node getter.
  Node getParent();

  /// Adds an attribute to this node, replaces existing attributes with the same
  /// name.
  void addAttribute(Attribute attribute);
  /// Emits this node to the output stream.
  void emit(emitter::JsonEmitter &jemit) override;
};

//===----------------------------------------------------------------------===//
// ConnectorNode
//===----------------------------------------------------------------------===//

/// Special type of node capable of connecting to other nodes (memlets).
class ConnectorNode : public Node {
protected:
  /// Pointer to the implementation (Pimpl idiom).
  std::shared_ptr<ConnectorNodeImpl> ptr;

public:
  ConnectorNode(std::shared_ptr<ConnectorNodeImpl> ptr)
      : Node(std::static_pointer_cast<NodeImpl>(ptr)), ptr(ptr) {}

  ConnectorNode(Node n)
      : Node(n), ptr(std::static_pointer_cast<ConnectorNodeImpl>(Node::ptr)) {}

  /// Adds an incoming connector.
  void addInConnector(Connector connector);
  /// Adds an outgoing connector.
  void addOutConnector(Connector connector);
  /// Returns to number of incoming connectors.
  unsigned getInConnectorCount();
  /// Returns to number of outgoing connectors.
  unsigned getOutConnectorCount();

  /// Emits the connectors to the output stream.
  void emit(emitter::JsonEmitter &jemit) override;
};

/// Implementation of the connector node class.
class ConnectorNodeImpl : public NodeImpl {
protected:
  /// Array of incoming connectors.
  std::vector<Connector> inConnectors;
  /// Array of outgoing connectors.
  std::vector<Connector> outConnectors;

public:
  ConnectorNodeImpl(Location location) : NodeImpl(location) {}

  /// Adds an incoming connector.
  void addInConnector(Connector connector);
  /// Adds an outgoing connector.
  void addOutConnector(Connector connector);
  /// Returns to number of incoming connectors.
  unsigned getInConnectorCount();
  /// Returns to number of outgoing connectors.
  unsigned getOutConnectorCount();

  /// Emits the connectors to the output stream.
  void emit(emitter::JsonEmitter &jemit) override;
};

/// Represents a single connector.
class Connector {
public:
  /// The connector node this connector belongs to.
  ConnectorNode parent;
  /// The name of the connector.
  std::string name;
  /// Null type connector for unnamed connectors (e.g. access nodes).
  bool isNull;
  /// The ranges of the moved data (memlet).
  std::vector<Range> ranges;
  /// The name of the data being moved.
  std::string data;
  // IDEA: Add DType?

  Connector(ConnectorNode parent)
      : parent(parent), name("null"), isNull(true) {}
  Connector(ConnectorNode parent, StringRef name)
      : parent(parent), name(name), isNull(false) {}

  bool operator==(const Connector other) const {
    return other.parent == parent && other.name == name;
  }

  /// Adds a data range to the connector.
  void addRange(Range range) { ranges.push_back(range); }
  /// Sets the data ranges of the connector.
  void setRanges(std::vector<Range> ranges) { this->ranges = ranges; }
  /// Sets the name of the data being moved.
  void setData(StringRef data) { this->data = data.str(); }
};

//===----------------------------------------------------------------------===//
// MultiEdge
//===----------------------------------------------------------------------===//

// IDEA: Rewrite to use PImpl?
/// Represents an edge moving data between multiple connectors (memlet).
class MultiEdge : public Emittable {
private:
  /// Source code location.
  Location location;
  /// Source connector.
  Connector source;
  /// Destination connector.
  Connector destination;

public:
  MultiEdge(Location location, Connector source, Connector destination)
      : location(location), source(source), destination(destination) {}

  /// Emits this edge to the output stream.
  void emit(emitter::JsonEmitter &jemit) override;
};

//===----------------------------------------------------------------------===//
// ScopeNode
//===----------------------------------------------------------------------===//

/// Special type of connector node containing a scope.
class ScopeNode : public ConnectorNode {
protected:
  /// Pointer to the implementation (Pimpl idiom).
  std::shared_ptr<ScopeNodeImpl> ptr;

public:
  ScopeNode(std::shared_ptr<ScopeNodeImpl> ptr)
      : ConnectorNode(std::static_pointer_cast<ConnectorNodeImpl>(ptr)),
        ptr(ptr) {}

  ScopeNode(Node n)
      : ConnectorNode(n),
        ptr(std::static_pointer_cast<ScopeNodeImpl>(Node::ptr)) {}

  /// Adds a connector node to the scope.
  virtual void addNode(ConnectorNode node);
  /// Adds a multiedge from the source to the destination connector.
  virtual void routeWrite(Connector from, Connector to);
  /// Adds an edge to the scope.
  virtual void addEdge(MultiEdge edge);
  /// Maps the MLIR value to the specified connector.
  virtual void mapConnector(Value value, Connector connector);
  /// Returns the connector associated with a MLIR value.
  virtual Connector lookup(Value value);

  /// Emits all nodes and edges to the output stream.
  void emit(emitter::JsonEmitter &jemit) override;
};

/// Implementation of the scoped node class.
class ScopeNodeImpl : public ConnectorNodeImpl {
protected:
  /// Lookup table for Value-Connector mapping.
  std::map<std::string, Connector> lut;
  /// Array of all nodes in the scope.
  std::vector<ConnectorNode> nodes;
  /// Array of all edges in the scope.
  std::vector<MultiEdge> edges;

public:
  ScopeNodeImpl(Location location) : ConnectorNodeImpl(location) {}

  /// Adds a connector node to the scope.
  virtual void addNode(ConnectorNode node);
  /// Adds a multiedge from the source to the destination connector.
  virtual void routeWrite(Connector from, Connector to);
  /// Adds an edge to the scope.
  virtual void addEdge(MultiEdge edge);
  /// Maps the MLIR value to the specified connector.
  virtual void mapConnector(Value value, Connector connector);
  /// Returns the connector associated with a MLIR value.
  virtual Connector lookup(Value value);

  /// Emits all nodes and edges to the output stream.
  void emit(emitter::JsonEmitter &jemit) override;
};

//===----------------------------------------------------------------------===//
// State
//===----------------------------------------------------------------------===//

/// Represents a SDFG state.
class State : public ScopeNode {
private:
  /// Pointer to the implementation (Pimpl idiom).
  std::shared_ptr<StateImpl> ptr;

public:
  State(std::shared_ptr<StateImpl> ptr)
      : ScopeNode(std::static_pointer_cast<ScopeNodeImpl>(ptr)), ptr(ptr) {
    type = NType::State;
  }

  State(Location location)
      : ScopeNode(std::static_pointer_cast<ScopeNodeImpl>(
            std::make_shared<StateImpl>(location))),
        ptr(std::static_pointer_cast<StateImpl>(Node::ptr)) {
    type = NType::State;
  }

  /// Modified lookup function creates access nodes if the value could not be
  /// found.
  Connector lookup(Value value) override;

  /// Emits the state node to the output stream.
  void emit(emitter::JsonEmitter &jemit) override;
};

/// Implementation of the state node class.
class StateImpl : public ScopeNodeImpl {
public:
  StateImpl(Location location) : ScopeNodeImpl(location) {}

  /// Modified lookup function creates access nodes if the value could not be
  /// found.
  Connector lookup(Value value) override;

  /// Emits the state node to the output stream.
  void emit(emitter::JsonEmitter &jemit) override;
};

//===----------------------------------------------------------------------===//
// SDFG
//===----------------------------------------------------------------------===//

/// Represents the top-level SDFG.
class SDFG : public Node {
private:
  /// Pointer to the implementation (Pimpl idiom).
  std::shared_ptr<SDFGImpl> ptr;

public:
  SDFG(std::shared_ptr<SDFGImpl> ptr)
      : Node(std::static_pointer_cast<NodeImpl>(ptr)), ptr(ptr) {
    type = NType::SDFG;
  }

  SDFG(Location location)
      : Node(std::static_pointer_cast<NodeImpl>(
            std::make_shared<SDFGImpl>(location))),
        ptr(std::static_pointer_cast<SDFGImpl>(Node::ptr)) {
    type = NType::SDFG;
  }

  /// Returns the state associated with the provided name.
  State lookup(StringRef name);
  /// Adds a state to the SDFG.
  void addState(State state);
  /// Adds a state to the SDFG and marks it as the entry state.
  void setStartState(State state);
  /// Adds an interstate edge to the SDFG, connecting two states.
  void addEdge(InterstateEdge edge);
  /// Adds an array (data container) to the SDFG.
  void addArray(Array array);
  /// Adds an array (data container) to the SDFG and marks it as an argument.
  void addArg(Array arg);
  /// Adds a symbol to the SDFG.
  void addSymbol(Symbol symbol);
  /// Returns an array of all symbols in the SDFG.
  std::vector<Symbol> getSymbols();

  /// Emits the SDFG to the output stream.
  void emit(emitter::JsonEmitter &jemit) override;
  /// Emits the SDFG as a nested SDFG to the output stream.
  void emitNested(emitter::JsonEmitter &jemit);
};

/// Implementation of the SDFG node class.
class SDFGImpl : public NodeImpl {
private:
  /// Lookup table mapping names to states.
  std::map<std::string, State> lut;
  /// Array of states in the SDFG.
  std::vector<State> states;
  /// Array of interstate edges in the SDFG.
  std::vector<InterstateEdge> edges;
  /// Array of arrays (data containers) in the SDFG.
  std::vector<Array> arrays;
  /// Array of argument arrays (data containers) in the SDFG.
  std::vector<Array> args;
  /// Array of symbols in the SDFG.
  std::vector<Symbol> symbols;
  /// The entry state of the SDFG
  State startState;
  /// Global counter for the ID of SDFGs.
  static unsigned list_id;

  /// Emits the body of the SDFG to the output stream.
  void emitBody(emitter::JsonEmitter &jemit);

public:
  SDFGImpl(Location location) : NodeImpl(location), startState(location) {
    id = SDFGImpl::list_id++;
  }

  /// Returns the state associated with the provided name.
  State lookup(StringRef name);
  /// Adds a state to the SDFG.
  void addState(State state);
  /// Adds a state to the SDFG and marks it as the entry state.
  void setStartState(State state);
  /// Adds an interstate edge to the SDFG, connecting two states.
  void addEdge(InterstateEdge edge);
  /// Adds an array (data container) to the SDFG.
  void addArray(Array array);
  /// Adds an array (data container) to the SDFG and marks it as an argument.
  void addArg(Array arg);
  /// Adds a symbol to the SDFG.
  void addSymbol(Symbol symbol);
  /// Returns an array of all symbols in the SDFG.
  std::vector<Symbol> getSymbols();

  /// Emits the SDFG to the output stream.
  void emit(emitter::JsonEmitter &jemit) override;
  /// Emits the SDFG as a nested SDFG to the output stream.
  void emitNested(emitter::JsonEmitter &jemit);
};

//===----------------------------------------------------------------------===//
// NestedSDFG
//===----------------------------------------------------------------------===//

/// Represents a nested SDFG.
class NestedSDFG : public ConnectorNode {
private:
  /// Pointer to the implementation (Pimpl idiom).
  std::shared_ptr<NestedSDFGImpl> ptr;

public:
  NestedSDFG(Location location, SDFG sdfg)
      : ConnectorNode(std::static_pointer_cast<ConnectorNodeImpl>(
            std::make_shared<NestedSDFGImpl>(location, sdfg))),
        ptr(std::static_pointer_cast<NestedSDFGImpl>(ConnectorNode::ptr)) {}

  /// Emits the nested SDFG to the output stream.
  void emit(emitter::JsonEmitter &jemit) override;
};

/// Implementation of the nested SDFG node class.
class NestedSDFGImpl : public ConnectorNodeImpl {
private:
  /// The contained SDFG.
  SDFG sdfg;

public:
  NestedSDFGImpl(Location location, SDFG sdfg)
      : ConnectorNodeImpl(location), sdfg(sdfg) {}

  /// Emits the nested SDFG to the output stream.
  void emit(emitter::JsonEmitter &jemit) override;
};

//===----------------------------------------------------------------------===//
// InterstateEdge
//===----------------------------------------------------------------------===//

/// Represents an edge connecting muliple states.
class InterstateEdge : public Emittable {
private:
  /// Pointer to the implementation (Pimpl idiom).
  std::shared_ptr<InterstateEdgeImpl> ptr;

public:
  InterstateEdge(Location location, State source, State destination)
      : ptr(std::make_shared<InterstateEdgeImpl>(location, source,
                                                 destination)) {}

  /// Sets the condition of the interstate edge.
  void setCondition(Condition condition);
  /// Adds an assignment to the interstate edge.
  void addAssignment(Assignment assignment);

  /// Emits the interstate edge to the output stream.
  void emit(emitter::JsonEmitter &jemit) override;
};

/// Implementation of the interstate edge class.
class InterstateEdgeImpl : public Emittable {
private:
  /// Source code location.
  Location location;
  /// The source state of this edge.
  State source;
  /// The destination state of this edge.
  State destination;

  /// The condition of this edge.
  Condition condition;
  /// Array of assignments on this edge.
  std::vector<Assignment> assignments;

public:
  InterstateEdgeImpl(Location location, State source, State destination)
      : location(location), source(source), destination(destination),
        condition("1") {}

  /// Sets the condition of the interstate edge.
  void setCondition(Condition condition);
  /// Adds an assignment to the interstate edge.
  void addAssignment(Assignment assignment);

  /// Emits the interstate edge to the output stream.
  void emit(emitter::JsonEmitter &jemit) override;
};

//===----------------------------------------------------------------------===//
// Tasklet
//===----------------------------------------------------------------------===//

/// Represents a SDFG tasklet.
class Tasklet : public ConnectorNode {
private:
  /// Pointer to the implementation (Pimpl idiom).
  std::shared_ptr<TaskletImpl> ptr;

public:
  Tasklet(Location location)
      : ConnectorNode(std::static_pointer_cast<ConnectorNodeImpl>(
            std::make_shared<TaskletImpl>(location))),
        ptr(std::static_pointer_cast<TaskletImpl>(ConnectorNode::ptr)) {}

  /// Sets the code of the tasklet.
  void setCode(Code code);
  /// Sets the global code of the tasklet.
  void setGlobalCode(Code code_global);
  /// Sets the side effect flag of the tasklet.
  void setHasSideEffect(bool hasSideEffect);

  /// Emits the tasklet to the output stream.
  void emit(emitter::JsonEmitter &jemit) override;
};

/// Implementation of the tasklet node class.
class TaskletImpl : public ConnectorNodeImpl {
private:
  /// The code in the tasklet.
  Code code;
  /// The global code required to run the tasklet.
  Code code_global;
  /// Flag indicating if this tasklet has side effects.
  bool hasSideEffect;

public:
  TaskletImpl(Location location)
      : ConnectorNodeImpl(location), hasSideEffect(false) {}

  /// Sets the code of the tasklet.
  void setCode(Code code);
  /// Sets the global code of the tasklet.
  void setGlobalCode(Code code_global);
  /// Sets the side effect flag of the tasklet.
  void setHasSideEffect(bool hasSideEffect);

  /// Emits the tasklet to the output stream.
  void emit(emitter::JsonEmitter &jemit) override;
};

//===----------------------------------------------------------------------===//
// Library
//===----------------------------------------------------------------------===//

/// Represents a SDFG libary node.
class Library : public ConnectorNode {
private:
  /// Pointer to the implementation (Pimpl idiom).
  std::shared_ptr<LibraryImpl> ptr;

public:
  Library(Location location)
      : ConnectorNode(std::static_pointer_cast<ConnectorNodeImpl>(
            std::make_shared<LibraryImpl>(location))),
        ptr(std::static_pointer_cast<LibraryImpl>(ConnectorNode::ptr)) {}

  /// Sets the library code path.
  void setClasspath(StringRef classpath);
  /// Emits the library node to the output stream.
  void emit(emitter::JsonEmitter &jemit) override;
};

/// Implementation of the library node class.
class LibraryImpl : public ConnectorNodeImpl {
private:
  /// The path to the library code.
  std::string classpath;

public:
  LibraryImpl(Location location) : ConnectorNodeImpl(location) {}

  /// Sets the library code path.
  void setClasspath(StringRef classpath);
  /// Emits the library node to the output stream.
  void emit(emitter::JsonEmitter &jemit) override;
};

//===----------------------------------------------------------------------===//
// Access
//===----------------------------------------------------------------------===//

/// Represents an access node in the SDFG.
class Access : public ConnectorNode {
private:
  /// Pointer to the implementation (Pimpl idiom).
  std::shared_ptr<AccessImpl> ptr;

public:
  Access(Location location)
      : ConnectorNode(std::static_pointer_cast<ConnectorNodeImpl>(
            std::make_shared<AccessImpl>(location))),
        ptr(std::static_pointer_cast<AccessImpl>(ConnectorNode::ptr)) {
    type = NType::Access;
  }

  /// Emits the access node to the output stream.
  void emit(emitter::JsonEmitter &jemit) override;
};

/// Implementation of the access node class.
class AccessImpl : public ConnectorNodeImpl {
private:
public:
  AccessImpl(Location location) : ConnectorNodeImpl(location) {}

  /// Emits the access node to the output stream.
  void emit(emitter::JsonEmitter &jemit) override;
};

//===----------------------------------------------------------------------===//
// Map
//===----------------------------------------------------------------------===//

/// Represents a map entry node in the SDFG.
class MapEntry : public ScopeNode {
private:
  /// Pointer to the implementation (Pimpl idiom).
  std::shared_ptr<MapEntryImpl> ptr;

public:
  MapEntry(Location location)
      : ScopeNode(std::static_pointer_cast<ScopeNodeImpl>(
            std::make_shared<MapEntryImpl>(location))),
        ptr(std::static_pointer_cast<MapEntryImpl>(Node::ptr)) {
    type = NType::MapEntry;
  }

  MapEntry(Node n)
      : ScopeNode(n), ptr(std::static_pointer_cast<MapEntryImpl>(Node::ptr)) {}

  MapEntry() : ScopeNode(nullptr) {}

  /// Adds a parameter to the map entry.
  void addParam(StringRef param);
  /// Adds a range for a parameter.
  void addRange(Range range);
  /// Sets the map exit this map entry belongs to.
  void setExit(MapExit exit);
  /// Returns the matching map exit.
  MapExit getExit();
  /// Adds a connector node to the scope.
  void addNode(ConnectorNode node) override;
  /// Adds a multiedge from the source to the destination connector.
  void routeWrite(Connector from, Connector to) override;
  /// Adds an edge to the scope.
  void addEdge(MultiEdge edge) override;
  /// Maps the MLIR value to the specified connector.
  void mapConnector(Value value, Connector connector) override;
  /// Returns the connector associated with a MLIR value, inserting map
  /// connectors when needed.
  Connector lookup(Value value) override;

  /// Emits the map entry to the output stream.
  void emit(emitter::JsonEmitter &jemit) override;
};

/// Represents a map exit node in the SDFG.
class MapExit : public ConnectorNode {
private:
  /// Pointer to the implementation (Pimpl idiom).
  std::shared_ptr<MapExitImpl> ptr;

public:
  MapExit(Location location)
      : ConnectorNode(std::static_pointer_cast<ConnectorNodeImpl>(
            std::make_shared<MapExitImpl>(location))),
        ptr(std::static_pointer_cast<MapExitImpl>(ConnectorNode::ptr)) {}

  MapExit() : ConnectorNode(nullptr) {}

  /// Sets the map entry this map exit belongs to.
  void setEntry(MapEntry entry);

  /// Emits the map exit to the output stream.
  void emit(emitter::JsonEmitter &jemit) override;
};

/// Implementation of the map entry node class.
class MapEntryImpl : public ScopeNodeImpl {
private:
  /// The matching map exit.
  MapExit exit;
  /// Array of parameters.
  std::vector<std::string> params;
  /// Array of ranges for the parameters.
  std::vector<Range> ranges;

public:
  MapEntryImpl(Location location) : ScopeNodeImpl(location) {}

  /// Adds a parameter to the map entry.
  void addParam(StringRef param);
  /// Adds a range for a parameter.
  void addRange(Range range);
  /// Sets the map exit this map entry belongs to.
  void setExit(MapExit exit);
  /// Returns the matching map exit.
  MapExit getExit();
  /// Adds a connector node to the scope.
  void addNode(ConnectorNode node) override;
  /// Adds a multiedge from the source to the destination connector.
  void routeWrite(Connector from, Connector to) override;
  /// Adds an edge to the scope.
  void addEdge(MultiEdge edge) override;
  /// Maps the MLIR value to the specified connector.
  void mapConnector(Value value, Connector connector) override;
  /// Returns the connector associated with a MLIR value, inserting map
  /// connectors when needed.
  Connector lookup(Value value, MapEntry mapEntry);

  /// Emits the map entry to the output stream.
  void emit(emitter::JsonEmitter &jemit) override;
};

/// Implementation of the map exit node class.
class MapExitImpl : public ConnectorNodeImpl {
private:
  /// The matching map entry.
  MapEntry entry;

public:
  MapExitImpl(Location location) : ConnectorNodeImpl(location) {}

  /// Sets the map entry this map exit belongs to.
  void setEntry(MapEntry entry);

  /// Emits the map exit to the output stream.
  void emit(emitter::JsonEmitter &jemit) override;
};

//===----------------------------------------------------------------------===//
// Consume
//===----------------------------------------------------------------------===//

/// Represents a consume entry node in the SDFG.
class ConsumeEntry : public ScopeNode {
private:
  /// Pointer to the implementation (Pimpl idiom).
  std::shared_ptr<ConsumeEntryImpl> ptr;

public:
  ConsumeEntry(Location location)
      : ScopeNode(std::static_pointer_cast<ScopeNodeImpl>(
            std::make_shared<ConsumeEntryImpl>(location))),
        ptr(std::static_pointer_cast<ConsumeEntryImpl>(Node::ptr)) {
    type = NType::ConsumeEntry;
  }

  ConsumeEntry(Node n)
      : ScopeNode(n),
        ptr(std::static_pointer_cast<ConsumeEntryImpl>(Node::ptr)) {}

  ConsumeEntry() : ScopeNode(nullptr) {}

  /// Sets the consume exit this consume entry belongs to.
  void setExit(ConsumeExit exit);
  /// Returns the matching consume exit.
  ConsumeExit getExit();
  /// Adds a connector node to the scope.
  void addNode(ConnectorNode node) override;
  /// Adds a multiedge from the source to the destination connector.
  void routeWrite(Connector from, Connector to) override;

  /// Adds an edge to the scope.
  void addEdge(MultiEdge edge) override;
  /// Maps the MLIR value to the specified connector.
  void mapConnector(Value value, Connector connector) override;
  /// Returns the connector associated with a MLIR value, inserting consume
  /// connectors when needed.
  Connector lookup(Value value) override;

  /// Sets the number of processing elements.
  void setNumPes(StringRef pes);
  /// Sets the name of the processing element index.
  void setPeIndex(StringRef pe);
  /// Sets the condition to continue stream consumption.
  void setCondition(Code condition);

  /// Emits the consume entry to the output stream.
  void emit(emitter::JsonEmitter &jemit) override;
};

/// Represents a consume exit node in the SDFG.
class ConsumeExit : public ConnectorNode {
private:
  /// Pointer to the implementation (Pimpl idiom).
  std::shared_ptr<ConsumeExitImpl> ptr;

public:
  ConsumeExit(Location location)
      : ConnectorNode(std::static_pointer_cast<ConnectorNodeImpl>(
            std::make_shared<ConsumeExitImpl>(location))),
        ptr(std::static_pointer_cast<ConsumeExitImpl>(ConnectorNode::ptr)) {}

  ConsumeExit() : ConnectorNode(nullptr) {}

  /// Sets the consume entry this consume exit belongs to.
  void setEntry(ConsumeEntry entry);

  /// Emits the consume exit to the output stream.
  void emit(emitter::JsonEmitter &jemit) override;
};

/// Implementation of the consume entry node class.
class ConsumeEntryImpl : public ScopeNodeImpl {
private:
  /// The matching consume exit.
  ConsumeExit exit;
  /// The number of processing elements.
  std::string num_pes;
  /// The name of the processing element index.
  std::string pe_index;
  /// The condition to continue stream consumption.
  Code condition;

public:
  ConsumeEntryImpl(Location location) : ScopeNodeImpl(location) {}

  /// Sets the consume exit this consume entry belongs to.
  void setExit(ConsumeExit exit);
  /// Returns the matching consume exit.
  ConsumeExit getExit();
  /// Adds a connector node to the scope.
  void addNode(ConnectorNode node) override;
  /// Adds a multiedge from the source to the destination connector.
  void routeWrite(Connector from, Connector to) override;

  /// Adds an edge to the scope.
  void addEdge(MultiEdge edge) override;
  /// Maps the MLIR value to the specified connector.
  void mapConnector(Value value, Connector connector) override;
  /// Returns the connector associated with a MLIR value, inserting consume
  /// connectors when needed.
  Connector lookup(Value value, ConsumeEntry entry);

  /// Sets the number of processing elements.
  void setNumPes(StringRef pes);
  /// Sets the name of the processing element index.
  void setPeIndex(StringRef pe);
  /// Sets the condition to continue stream consumption.
  void setCondition(Code condition);

  /// Emits the consume entry to the output stream.
  void emit(emitter::JsonEmitter &jemit) override;
};

/// Implementation of the consume exit node class.
class ConsumeExitImpl : public ConnectorNodeImpl {
private:
  /// The matching consume entry.
  ConsumeEntry entry;

public:
  ConsumeExitImpl(Location location) : ConnectorNodeImpl(location) {}

  /// Sets the consume entry this consume exit belongs to.
  void setEntry(ConsumeEntry entry);

  /// Emits the consume exit to the output stream.
  void emit(emitter::JsonEmitter &jemit) override;
};

} // namespace mlir::sdfg::translation

#endif // SDFG_Translation_Node_H
