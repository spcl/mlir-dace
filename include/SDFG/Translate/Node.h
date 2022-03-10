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
class SDFG;
class State;
class StateImpl;

class InterstateEdge;
class MultiEdge;

class ConnectorNode;
class Access;
class Tasklet;

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

class Connector {
public:
  // TODO: Change to shared_ptr
  ConnectorNode *parent;
  std::string name;
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
  unsigned id;
  Location location;
  std::string name;
  std::vector<Attribute> attributes;
  // TODO: Change to shared_ptr
  Node *parent;

  Node(Location location) : id(0), location(location) {}

public:
  void setID(unsigned id) { this->id = id; }
  unsigned getID() { return id; }

  Location getLocation() { return location; }

  void setName(StringRef name) {
    this->name = name.str();
    utils::sanitizeName(this->name);
  }
  StringRef getName() { return name; }

  void setParent(Node *parent) { this->parent = parent; }
  Node *getParent() { return parent; }

  // check for existing attribtues
  // Replace or add to list
  void addAttribute(Attribute attribute);
};

//===----------------------------------------------------------------------===//
// State
//===----------------------------------------------------------------------===//

class StateImpl : public Node, public Emittable {

private:
  std::map<unsigned, ConnectorNode *> lut;
  std::vector<ConnectorNode> nodes;
  std::vector<MultiEdge> edges;

public:
  StateImpl(Location location) : Node(location) {}

  void addNode(ConnectorNode node);
  void addEdge(MultiEdge edge);

  void emit(emitter::JsonEmitter &jemit) override;
};

class State : public Emittable {
private:
  std::shared_ptr<StateImpl> ptr;

public:
  State() : ptr(nullptr) {}
  State(Location location) : ptr(std::make_shared<StateImpl>(location)) {}

  // TODO: Avoid -> overloading
  std::shared_ptr<StateImpl> operator->() const { return ptr; }
  bool operator==(const State other) const { return other.ptr == ptr; }

  void emit(emitter::JsonEmitter &jemit) override { ptr->emit(jemit); }
};

//===----------------------------------------------------------------------===//
// SDFG
//===----------------------------------------------------------------------===//

class SDFG : public Node, public Emittable {
private:
  std::map<unsigned, State> lut;
  std::vector<State> states;
  std::vector<InterstateEdge> edges;
  State startState;

public:
  SDFG(Location location) : Node(location) {}

  State lookup(unsigned id);
  void addState(unsigned id, State state);
  void setStartState(State state);
  void addEdge(InterstateEdge edge);

  void emit(emitter::JsonEmitter &jemit) override;
};

//===----------------------------------------------------------------------===//
// Edges
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

class MultiEdge : public Emittable {

private:
  // TODO: Change to shared_ptr
  Connector *source;
  // TODO: Change to shared_ptr
  Connector *destination;
  mlir::sdfg::SizedType shape;

public:
  MultiEdge(Connector *source, Connector *destination);

  void emit(emitter::JsonEmitter &jemit) override;
};

//===----------------------------------------------------------------------===//
// ConnectorNode
//===----------------------------------------------------------------------===//

class ConnectorNode : public Node {

protected:
  std::vector<Connector> connectors;
};

class Tasklet : public ConnectorNode {};
class Access : public ConnectorNode {};

/* class MapBegin : public ConnectorNode {};
class MapEnd : public ConnectorNode {};

class ConsumeBegin : public ConnectorNode {};
class ConsumeEnd : public ConnectorNode {}; */

} // namespace mlir::sdfg::translation

#endif // SDFG_Translation_Node_H
