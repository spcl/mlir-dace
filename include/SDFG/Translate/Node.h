#ifndef SDFG_Translation_Node_H
#define SDFG_Translation_Node_H

#include "SDFG/Dialect/Dialect.h"
#include "SDFG/Translate/JsonEmitter.h"
#include "mlir/IR/Location.h"

namespace mlir::sdfg::translation {
class Attribute;
class Connector;
class Assignment;

// class Edge;
class InterstateEdge;
class MultiEdge;

class Node;
class ContainerNode;
class SDFG;
class State;

class ConnectorNode;
class Access;
class Tasklet;

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
  ConnectorNode *parent;
  std::string name;
};

class Assignment {
public:
  std::string key;
  std::string value;
};

//===----------------------------------------------------------------------===//
// Interfaces
//===----------------------------------------------------------------------===//

class Emittable {
public:
  virtual void emit(emitter::JsonEmitter &jemit) = 0;
};

//===----------------------------------------------------------------------===//
// Edges
//===----------------------------------------------------------------------===//

class InterstateEdge {
protected:
  std::shared_ptr<State> source;
  std::shared_ptr<State> destination;

  std::string condition;
  std::vector<Assignment> assignments;

public:
  InterstateEdge(std::shared_ptr<State> source,
                 std::shared_ptr<State> destination)
      : source(source), destination(destination) {}

  void setCondition(StringRef condition);
  void addAssignment(Assignment assignment);

  void emit(emitter::JsonEmitter &jemit);
};

class MultiEdge {

private:
  Connector *source;
  Connector *destination;
  mlir::sdfg::SizedType shape;

public:
  MultiEdge(Connector *source, Connector *destination);

  void emit(emitter::JsonEmitter &jemit);
};

//===----------------------------------------------------------------------===//
// Nodes
//===----------------------------------------------------------------------===//

class Node {
protected:
  unsigned id;
  Location location;
  std::string label;
  std::vector<Attribute> attributes;
  Node *parent;

  Node(Location location) : id(0), location(location) {}

public:
  void setID(unsigned id) { this->id = id; }
  unsigned getID() { return id; }

  Location getLocation() { return location; }

  void setLabel(StringRef label) { this->label = label.str(); }
  StringRef getLabel() { return label; }

  void setParent(Node *parent) { this->parent = parent; }
  Node *getParent() { return parent; }

  virtual void emit(emitter::JsonEmitter &jemit) {}

  // check for existing attribtues
  // Replace or add to list
  void addAttribute(Attribute attribute);
};

class SDFG : public Node {
private:
  std::map<unsigned, std::shared_ptr<State>> lut;
  std::vector<State> nodes;
  std::vector<InterstateEdge> edges;

public:
  SDFG(Location location) : Node(location) {}

  std::shared_ptr<State> lookup(unsigned id);
  void addState(unsigned id, std::shared_ptr<State> state);
  void addEdge(InterstateEdge edge);

  void emit(emitter::JsonEmitter &jemit) override;
};

class State : public Node {

private:
  std::map<unsigned, ConnectorNode *> lut;
  std::vector<ConnectorNode> nodes;
  std::vector<MultiEdge> edges;

public:
  State(Location location) : Node(location) {}

  void addNode(ConnectorNode node);
  void addEdge(MultiEdge edge);

  void emit(emitter::JsonEmitter &jemit) override;
};

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
