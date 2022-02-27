#ifndef SDFG_Translation_Node_H
#define SDFG_Translation_Node_H

#include "SDFG/Dialect/Dialect.h"
#include "SDFG/Translate/JsonEmitter.h"
#include "mlir/IR/Location.h"

namespace mlir::sdfg::translation {
class Attribute;
class Connector;
class Assignment;

class Edge;
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
private:
  std::string name;
  // Store attribute or string?
};

class Connector {
private:
  ConnectorNode *parent;
  std::string name;
};

class Assignment {
private:
  std::string key;
  std::string value;
};

//===----------------------------------------------------------------------===//
// Edges
//===----------------------------------------------------------------------===//

class Edge {
protected:
  Node *source;
  Node *destination;

  Edge(Node *source, Node *destination)
      : source(source), destination(destination) {}

public:
  virtual void emit(emitter::JsonEmitter &jemit) {}
};

class InterstateEdge : public Edge {
protected:
  State *source;
  State *destination;

  std::string condition;
  std::vector<Assignment> assignments;

public:
  InterstateEdge(State *source, State *destination);
  void setCondition(std::string condition);
  void addAssignment(Assignment assignment);
};

class MultiEdge : public Edge {

private:
  Connector source;
  Connector destination;
  mlir::sdfg::SizedType shape;

public:
  MultiEdge(Connector source, Connector destination);
};

//===----------------------------------------------------------------------===//
// Nodes
//===----------------------------------------------------------------------===//

class Node {
protected:
  unsigned id;
  Location location;
  std::vector<Attribute> attributes;
  Node *parent;

  Node(Location location) : id(0), location(location) {}

public:
  void setID(unsigned id);
  unsigned getID();
  void setParent(Node *parent);
  Node *getParent();

  virtual void emit(emitter::JsonEmitter &jemit) {}

  // check for existing attribtues
  // Replace or add to list
  void addAttribute(Attribute attribute);
};

class SDFG : public Node {
private:
  // LUT (id -> node)
  std::vector<State> nodes;
  std::vector<InterstateEdge> edges;

public:
  SDFG(Location location) : Node(location) {}
  void emit(emitter::JsonEmitter &jemit) override;
  void addNode(State &state);
  void addEdge(InterstateEdge edge);
};

class State : public Node {

private:
  // LUT (id -> node)
  std::vector<ConnectorNode> nodes;
  std::vector<MultiEdge> edges;

public:
  State(Location location) : Node(location) {}
  void emit(emitter::JsonEmitter &jemit) override;
  void addNode(ConnectorNode node);
  void addEdge(MultiEdge edge);
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
