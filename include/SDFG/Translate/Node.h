#ifndef SDFG_Translation_Node_H
#define SDFG_Translation_Node_H

#include "SDFG/Dialect/Dialect.h"
#include "SDFG/Translate/JsonEmitter.h"
#include "SDFG/Utils/Utils.h"
#include "mlir/IR/Location.h"

namespace mlir::sdfg::translation {
class Emittable;

class Attribute;
class Connector;
class Assignment;

class Node;
class NodeImpl;

class SDFG;
class SDFGImpl;

class State;
class StateImpl;

class InterstateEdge;
class MultiEdge;

class ConnectorNode;
class ConnectorNodeImpl;

class Tasklet;
class TaskletImpl;

class Access;

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

//===----------------------------------------------------------------------===//
// Node
//===----------------------------------------------------------------------===//

class Node {
protected:
  std::shared_ptr<NodeImpl> ptr;

public:
  Node(std::shared_ptr<NodeImpl> ptr) : ptr(ptr) {}

  bool operator==(const Node other) const { return other.ptr == ptr; }

  void setID(unsigned id);
  unsigned getID();

  Location getLocation();

  void setName(StringRef name);
  StringRef getName();

  void setParent(Node parent);
  Node getParent();

  void addAttribute(Attribute attribute);
};

class NodeImpl {
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
};

//===----------------------------------------------------------------------===//
// ConnectorNode
//===----------------------------------------------------------------------===//

class ConnectorNode : public Node, public Emittable {
protected:
  std::shared_ptr<ConnectorNodeImpl> ptr;

public:
  ConnectorNode(std::shared_ptr<ConnectorNodeImpl> ptr)
      : Node(std::static_pointer_cast<NodeImpl>(ptr)), ptr(ptr) {}

  void addInConnector(Connector connector);
  void addOutConnector(Connector connector);
  void emit(emitter::JsonEmitter &jemit) override;
};

class ConnectorNodeImpl : public NodeImpl, public Emittable {
protected:
  std::vector<Connector> inConnectors;
  std::vector<Connector> outConnectors;

public:
  ConnectorNodeImpl(Location location) : NodeImpl(location) {}

  void addInConnector(Connector connector);
  void addOutConnector(Connector connector);

  // Emits connectors
  void emit(emitter::JsonEmitter &jemit) override;
};

class Connector {
public:
  ConnectorNode parent;
  std::string name;

  Connector(ConnectorNode parent, StringRef name)
      : parent(parent), name(name) {}

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
// State
//===----------------------------------------------------------------------===//

class State : public Node, public Emittable {
private:
  std::shared_ptr<StateImpl> ptr;

public:
  State(Location location)
      : Node(std::static_pointer_cast<NodeImpl>(
            std::make_shared<StateImpl>(location))),
        ptr(std::static_pointer_cast<StateImpl>(Node::ptr)) {}

  void addNode(unsigned id, ConnectorNode node);
  void addEdge(MultiEdge edge);
  void emit(emitter::JsonEmitter &jemit) override;
};

class StateImpl : public NodeImpl, public Emittable {
private:
  std::map<unsigned, ConnectorNode> lut;
  std::vector<ConnectorNode> nodes;
  std::vector<MultiEdge> edges;

public:
  StateImpl(Location location) : NodeImpl(location) {}

  void addNode(unsigned id, ConnectorNode node);
  void addEdge(MultiEdge edge);
  void emit(emitter::JsonEmitter &jemit) override;
};

//===----------------------------------------------------------------------===//
// SDFG
//===----------------------------------------------------------------------===//

class SDFG : public Node, public Emittable {
private:
  std::shared_ptr<SDFGImpl> ptr;

public:
  SDFG(Location location)
      : Node(std::static_pointer_cast<NodeImpl>(
            std::make_shared<SDFGImpl>(location))),
        ptr(std::static_pointer_cast<SDFGImpl>(Node::ptr)) {}

  State lookup(unsigned id);
  void addState(unsigned id, State state);
  void setStartState(State state);
  void addEdge(InterstateEdge edge);
  void emit(emitter::JsonEmitter &jemit) override;
};

class SDFGImpl : public NodeImpl, public Emittable {
private:
  std::map<unsigned, State> lut;
  std::vector<State> states;
  std::vector<InterstateEdge> edges;
  State startState;

public:
  SDFGImpl(Location location) : NodeImpl(location), startState(location) {}

  State lookup(unsigned id);
  void addState(unsigned id, State state);
  void setStartState(State state);
  void addEdge(InterstateEdge edge);

  void emit(emitter::JsonEmitter &jemit) override;
};

//===----------------------------------------------------------------------===//
// InterstateEdge
//===----------------------------------------------------------------------===//

class InterstateEdge : public Emittable {
protected:
  State source;
  State destination;

  Condition condition;
  std::vector<Assignment> assignments;

public:
  InterstateEdge(State source, State destination)
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

// class Access : public ConnectorNode {};

/* class MapBegin : public ConnectorNode {};
class MapEnd : public ConnectorNode {};

class ConsumeBegin : public ConnectorNode {};
class ConsumeEnd : public ConnectorNode {}; */

} // namespace mlir::sdfg::translation

#endif // SDFG_Translation_Node_H
