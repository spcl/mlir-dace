#ifndef SDFG_Translation_Node_H
#define SDFG_Translation_Node_H

#include "SDFG/Dialect/Dialect.h"
#include "SDFG/Translate/JsonEmitter.h"
#include "SDFG/Utils/Utils.h"
#include "mlir/IR/Location.h"

namespace mlir::sdfg::translation {
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

class Emittable {
public:
  virtual void emit(emitter::JsonEmitter &jemit) = 0;
};

//===----------------------------------------------------------------------===//
// DataClasses
//===----------------------------------------------------------------------===//

enum class DType { int32, int64, float32, float64 };
enum class NType { SDFG, State, Access, MapEntry, ConsumeEntry, Other };

class Attribute {
public:
  std::string name;
  // Store attribute or string?
};

class Condition {
public:
  std::string condition;

  Condition(StringRef condition) : condition(condition) {}
};

class Assignment {
public:
  std::string key;
  std::string value;

  Assignment(StringRef key, StringRef value) : key(key), value(value) {}
};

class Array : public Emittable {
public:
  std::string name;
  bool transient;
  SizedType shape;

  Array(StringRef name, bool transient, Type t)
      : name(name), transient(transient),
        shape(SizedType::get(t.getContext(), t, {}, {}, {})) {}

  Array(StringRef name, bool transient, SizedType shape)
      : name(name), transient(transient), shape(shape) {}

  void emit(emitter::JsonEmitter &jemit) override;
};

class Range : public Emittable {
public:
  std::string start;
  std::string end;
  std::string step;
  std::string tile;

  Range(StringRef start, StringRef end, StringRef step, StringRef tile)
      : start(start), end(end), step(step), tile(tile) {}

  void emit(emitter::JsonEmitter &jemit) override;
};

//===----------------------------------------------------------------------===//
// Node
//===----------------------------------------------------------------------===//

class Node : public Emittable {
protected:
  std::shared_ptr<NodeImpl> ptr;
  NType type;

public:
  Node(std::shared_ptr<NodeImpl> ptr) : ptr(ptr), type(NType::Other) {}

  bool operator==(const Node other) const { return other.ptr == ptr; }

  void setID(unsigned id);
  unsigned getID();

  Location getLocation();
  NType getType();

  void setName(StringRef name);
  StringRef getName();

  void setParent(Node parent);
  Node getParent();
  bool hasParent();

  virtual SDFG getSDFG();
  virtual State getState();
  virtual MapEntry getMapEntry();
  virtual ConsumeEntry getConsumeEntry();

  void addAttribute(Attribute attribute);
  void emit(emitter::JsonEmitter &jemit) override;
};

class NodeImpl : public Emittable {
protected:
  unsigned id;
  Location location;
  std::string name;
  std::vector<Attribute> attributes;
  Node parent;

public:
  NodeImpl(Location location) : id(0), location(location), parent(nullptr) {}

  void setID(unsigned id);
  unsigned getID();

  Location getLocation();

  void setName(StringRef name);
  StringRef getName();

  void setParent(Node parent);
  Node getParent();

  // check for existing attribtues
  // Replace or add to list
  void addAttribute(Attribute attribute);

  void emit(emitter::JsonEmitter &jemit) override;
};

//===----------------------------------------------------------------------===//
// ConnectorNode
//===----------------------------------------------------------------------===//

class ConnectorNode : public Node {
protected:
  std::shared_ptr<ConnectorNodeImpl> ptr;

public:
  ConnectorNode(std::shared_ptr<ConnectorNodeImpl> ptr)
      : Node(std::static_pointer_cast<NodeImpl>(ptr)), ptr(ptr) {}

  ConnectorNode(Node n)
      : Node(n), ptr(std::static_pointer_cast<ConnectorNodeImpl>(Node::ptr)) {}

  void addInConnector(Connector connector);
  void addOutConnector(Connector connector);
  unsigned getInConnectorCount();
  unsigned getOutConnectorCount();

  void emit(emitter::JsonEmitter &jemit) override;
};

class ConnectorNodeImpl : public NodeImpl {
protected:
  std::vector<Connector> inConnectors;
  std::vector<Connector> outConnectors;

public:
  ConnectorNodeImpl(Location location) : NodeImpl(location) {}

  void addInConnector(Connector connector);
  void addOutConnector(Connector connector);
  unsigned getInConnectorCount();
  unsigned getOutConnectorCount();

  // Emits connectors
  void emit(emitter::JsonEmitter &jemit) override;
};

class Connector {
public:
  ConnectorNode parent;
  std::string name;
  bool isNull;
  // DType?

  Connector(ConnectorNode parent)
      : parent(parent), name("null"), isNull(true) {}
  Connector(ConnectorNode parent, StringRef name)
      : parent(parent), name(name), isNull(false) {}

  bool operator==(const Connector other) const {
    return other.parent == parent && other.name == name;
  }
};

//===----------------------------------------------------------------------===//
// MultiEdge
//===----------------------------------------------------------------------===//

class MultiEdge : public Emittable {

private:
  Connector source;
  Connector destination;
  mlir::sdfg::SizedType shape;

public:
  MultiEdge(Connector source, Connector destination)
      : source(source), destination(destination) {}

  void emit(emitter::JsonEmitter &jemit) override;
};

//===----------------------------------------------------------------------===//
// ScopeNode
//===----------------------------------------------------------------------===//

class ScopeNode : public ConnectorNode {
protected:
  std::shared_ptr<ScopeNodeImpl> ptr;

public:
  ScopeNode(std::shared_ptr<ScopeNodeImpl> ptr)
      : ConnectorNode(std::static_pointer_cast<ConnectorNodeImpl>(ptr)),
        ptr(ptr) {}

  ScopeNode(Node n)
      : ConnectorNode(n),
        ptr(std::static_pointer_cast<ScopeNodeImpl>(Node::ptr)) {}

  virtual void addNode(ConnectorNode node);
  virtual void routeWrite(Connector from, Connector to);
  virtual void addEdge(MultiEdge edge);
  virtual void mapConnector(Value value, Connector connector);
  virtual Connector lookup(Value value);

  void emit(emitter::JsonEmitter &jemit) override;
};

class ScopeNodeImpl : public ConnectorNodeImpl {
protected:
  std::map<std::string, Connector> lut;
  std::vector<ConnectorNode> nodes;
  std::vector<MultiEdge> edges;

public:
  ScopeNodeImpl(Location location) : ConnectorNodeImpl(location) {}

  virtual void addNode(ConnectorNode node);
  virtual void routeWrite(Connector from, Connector to);
  virtual void addEdge(MultiEdge edge);
  virtual void mapConnector(Value value, Connector connector);
  virtual Connector lookup(Value value);

  // Emits nodes & edges
  void emit(emitter::JsonEmitter &jemit) override;
};

//===----------------------------------------------------------------------===//
// State
//===----------------------------------------------------------------------===//

class State : public ScopeNode {
private:
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

  Connector lookup(Value value) override;
  void emit(emitter::JsonEmitter &jemit) override;
};

class StateImpl : public ScopeNodeImpl {
public:
  StateImpl(Location location) : ScopeNodeImpl(location) {}

  Connector lookup(Value value) override;
  void emit(emitter::JsonEmitter &jemit) override;
};

//===----------------------------------------------------------------------===//
// SDFG
//===----------------------------------------------------------------------===//

class SDFG : public Node {
private:
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

  State lookup(StringRef name);
  void addState(State state);
  void setStartState(State state);
  void addEdge(InterstateEdge edge);
  void addArray(Array array);
  void addArg(Array arg);

  void emit(emitter::JsonEmitter &jemit) override;
  void emitNested(emitter::JsonEmitter &jemit);
};

class SDFGImpl : public NodeImpl {
private:
  std::map<std::string, State> lut;
  std::vector<State> states;
  std::vector<InterstateEdge> edges;
  std::vector<Array> arrays;
  std::vector<Array> args;
  State startState;
  static unsigned list_id;

  void emitBody(emitter::JsonEmitter &jemit);

public:
  SDFGImpl(Location location) : NodeImpl(location), startState(location) {
    id = SDFGImpl::list_id++;
  }

  State lookup(StringRef name);
  void addState(State state);
  void setStartState(State state);
  void addEdge(InterstateEdge edge);
  void addArray(Array array);
  void addArg(Array arg);

  void emit(emitter::JsonEmitter &jemit) override;
  void emitNested(emitter::JsonEmitter &jemit);
};

//===----------------------------------------------------------------------===//
// NestedSDFG
//===----------------------------------------------------------------------===//

class NestedSDFG : public ConnectorNode {
private:
  std::shared_ptr<NestedSDFGImpl> ptr;

public:
  NestedSDFG(Location location, SDFG sdfg)
      : ConnectorNode(std::static_pointer_cast<ConnectorNodeImpl>(
            std::make_shared<NestedSDFGImpl>(location, sdfg))),
        ptr(std::static_pointer_cast<NestedSDFGImpl>(ConnectorNode::ptr)) {}

  void emit(emitter::JsonEmitter &jemit) override;
};

class NestedSDFGImpl : public ConnectorNodeImpl {
private:
  SDFG sdfg;

public:
  NestedSDFGImpl(Location location, SDFG sdfg)
      : ConnectorNodeImpl(location), sdfg(sdfg) {}

  void emit(emitter::JsonEmitter &jemit) override;
};

//===----------------------------------------------------------------------===//
// InterstateEdge
//===----------------------------------------------------------------------===//

class InterstateEdge : public Emittable {
private:
  std::shared_ptr<InterstateEdgeImpl> ptr;

public:
  InterstateEdge(State source, State destination)
      : ptr(std::make_shared<InterstateEdgeImpl>(source, destination)) {}

  void setCondition(Condition condition);
  void addAssignment(Assignment assignment);

  void emit(emitter::JsonEmitter &jemit) override;
};

class InterstateEdgeImpl : public Emittable {
private:
  State source;
  State destination;

  Condition condition;
  std::vector<Assignment> assignments;

public:
  InterstateEdgeImpl(State source, State destination)
      : source(source), destination(destination), condition("1") {}

  void setCondition(Condition condition);
  // Check for duplicates
  void addAssignment(Assignment assignment);

  void emit(emitter::JsonEmitter &jemit) override;
};

//===----------------------------------------------------------------------===//
// Tasklet
//===----------------------------------------------------------------------===//

class Tasklet : public ConnectorNode {
private:
  std::shared_ptr<TaskletImpl> ptr;

public:
  Tasklet(Location location)
      : ConnectorNode(std::static_pointer_cast<ConnectorNodeImpl>(
            std::make_shared<TaskletImpl>(location))),
        ptr(std::static_pointer_cast<TaskletImpl>(ConnectorNode::ptr)) {}

  void setCode(StringRef code);
  void setLanguage(StringRef language);

  void emit(emitter::JsonEmitter &jemit) override;
};

class TaskletImpl : public ConnectorNodeImpl {
private:
  std::string code;
  std::string language;

public:
  TaskletImpl(Location location)
      : ConnectorNodeImpl(location), language("Python") {}

  void setCode(StringRef code);
  void setLanguage(StringRef language);

  void emit(emitter::JsonEmitter &jemit) override;
};

//===----------------------------------------------------------------------===//
// Access
//===----------------------------------------------------------------------===//

class Access : public ConnectorNode {
private:
  std::shared_ptr<AccessImpl> ptr;

public:
  Access(Location location)
      : ConnectorNode(std::static_pointer_cast<ConnectorNodeImpl>(
            std::make_shared<AccessImpl>(location))),
        ptr(std::static_pointer_cast<AccessImpl>(ConnectorNode::ptr)) {
    type = NType::Access;
  }

  void emit(emitter::JsonEmitter &jemit) override;
};

class AccessImpl : public ConnectorNodeImpl {
private:
public:
  AccessImpl(Location location) : ConnectorNodeImpl(location) {}

  void emit(emitter::JsonEmitter &jemit) override;
};

//===----------------------------------------------------------------------===//
// Map
//===----------------------------------------------------------------------===//

class MapEntry : public ScopeNode {
private:
  std::shared_ptr<MapEntryImpl> ptr;

public:
  MapEntry(Location location)
      : ScopeNode(std::static_pointer_cast<ScopeNodeImpl>(
            std::make_shared<MapEntryImpl>(location))),
        ptr(std::static_pointer_cast<MapEntryImpl>(Node::ptr)) {
    type = NType::MapEntry;
  }

  MapEntry() : ScopeNode(nullptr) {}

  void addParam(StringRef param);
  void addRange(Range range);
  void setExit(MapExit exit);
  MapExit getExit();
  void addNode(ConnectorNode node) override;
  void routeWrite(Connector from, Connector to) override;
  void addEdge(MultiEdge edge) override;
  void mapConnector(Value value, Connector connector) override;
  Connector lookup(Value value) override;
  void emit(emitter::JsonEmitter &jemit) override;
};

class MapExit : public ConnectorNode {
private:
  std::shared_ptr<MapExitImpl> ptr;

public:
  MapExit(Location location)
      : ConnectorNode(std::static_pointer_cast<ConnectorNodeImpl>(
            std::make_shared<MapExitImpl>(location))),
        ptr(std::static_pointer_cast<MapExitImpl>(ConnectorNode::ptr)) {}

  MapExit() : ConnectorNode(nullptr) {}

  void setEntry(MapEntry entry);
  void emit(emitter::JsonEmitter &jemit) override;
};

class MapEntryImpl : public ScopeNodeImpl {
private:
  MapExit exit;
  std::vector<std::string> params;
  std::vector<Range> ranges;

public:
  MapEntryImpl(Location location) : ScopeNodeImpl(location) {}

  void setExit(MapExit exit);
  MapExit getExit();
  void addParam(StringRef param);
  void addRange(Range range);
  void addNode(ConnectorNode node) override;
  void routeWrite(Connector from, Connector to) override;
  void addEdge(MultiEdge edge) override;
  void mapConnector(Value value, Connector connector) override;
  Connector lookup(Value value, MapEntry mapEntry);
  void emit(emitter::JsonEmitter &jemit) override;
};

class MapExitImpl : public ConnectorNodeImpl {
private:
  MapEntry entry;

public:
  MapExitImpl(Location location) : ConnectorNodeImpl(location) {}

  void setEntry(MapEntry entry);
  void emit(emitter::JsonEmitter &jemit) override;
};

//===----------------------------------------------------------------------===//
// Consume
//===----------------------------------------------------------------------===//

class ConsumeEntry : public ScopeNode {
private:
  std::shared_ptr<ConsumeEntryImpl> ptr;

public:
  ConsumeEntry(Location location)
      : ScopeNode(std::static_pointer_cast<ScopeNodeImpl>(
            std::make_shared<ConsumeEntryImpl>(location))),
        ptr(std::static_pointer_cast<ConsumeEntryImpl>(Node::ptr)) {
    type = NType::ConsumeEntry;
  }

  ConsumeEntry() : ScopeNode(nullptr) {}

  ConsumeEntry(std::shared_ptr<ConsumeEntryImpl> ptr)
      : ScopeNode(std::static_pointer_cast<ScopeNodeImpl>(ptr)), ptr(ptr) {}

  void setExit(ConsumeExit exit);
  ConsumeExit getExit();
  void addNode(ConnectorNode node) override;
  void addEdge(MultiEdge edge) override;
  void mapConnector(Value value, Connector connector) override;
  Connector lookup(Value value) override;
  void emit(emitter::JsonEmitter &jemit) override;
};

class ConsumeExit : public ConnectorNode {
private:
  std::shared_ptr<ConsumeExitImpl> ptr;

public:
  ConsumeExit(Location location)
      : ConnectorNode(std::static_pointer_cast<ConnectorNodeImpl>(
            std::make_shared<ConsumeExitImpl>(location))),
        ptr(std::static_pointer_cast<ConsumeExitImpl>(ConnectorNode::ptr)) {}

  ConsumeExit() : ConnectorNode(nullptr) {}

  void setEntry(ConsumeEntry entry);
  void emit(emitter::JsonEmitter &jemit) override;
};

class ConsumeEntryImpl : public ScopeNodeImpl {
private:
  ConsumeExit exit;
  // Condition
  // pe index
  // num pes

public:
  ConsumeEntryImpl(Location location) : ScopeNodeImpl(location) {}

  void setExit(ConsumeExit exit);
  ConsumeExit getExit();
  void addNode(ConnectorNode node) override;
  void addEdge(MultiEdge edge) override;
  void mapConnector(Value value, Connector connector) override;
  Connector lookup(Value value, ConsumeEntry entry);
  void emit(emitter::JsonEmitter &jemit) override;
};

class ConsumeExitImpl : public ConnectorNodeImpl {
private:
  ConsumeEntry entry;

public:
  ConsumeExitImpl(Location location) : ConnectorNodeImpl(location) {}

  void setEntry(ConsumeEntry entry);
  void emit(emitter::JsonEmitter &jemit) override;
};

} // namespace mlir::sdfg::translation

#endif // SDFG_Translation_Node_H
