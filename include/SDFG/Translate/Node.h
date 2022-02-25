#ifndef SDFG_Translation_Node_H
#define SDFG_Translation_Node_H

#include "SDFG/Dialect/Dialect.h"
#include "SDFG/Translate/JsonEmitter.h"
#include "mlir/IR/Location.h"

namespace mlir::sdfg::translation {
class Attribute;
class Connector;

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

//===----------------------------------------------------------------------===//
// Edges
//===----------------------------------------------------------------------===//

class Edge {
public:
  Edge(Node *source, Node *destination);
  virtual void emit(emitter::JsonEmitter &jemit);

private:
  Node *source;
  Node *destination;
};

class InterstateEdge : public Edge {
public:
  InterstateEdge(StateNode *source, StateNode *destination);

private:
  StateNode *source;
  StateNode *destination;

  // Condition
  // assignments
};

class MultiEdge : public Edge {
public:
  MultiEdge(Connector source, Connector destination);

private:
  Connector source;
  Connector destination;

  mlir::sdfg::SizedType shape;
};

//===----------------------------------------------------------------------===//
// Nodes
//===----------------------------------------------------------------------===//

class Node {
public:
  virtual void emit(emitter::JsonEmitter &jemit);

  // check for existing attribtues
  // Replace or add to list
  void addAttribute(Attribute attribute);

protected:
  Node(Location location);

private:
  unsigned ID;
  Location location;
  std::vector<Attribute> attributes;
};

class ContainerNode : public Node {
public:
  void addNode(Node node);
  // Check EdgeNodes are inside this Node
  virtual void addEdge(Edge edge);

protected:
  ContainerNode(Location location);

private:
  std::vector<Node> children;
  std::vector<Edge> edges;
};

class SDFG : public ContainerNode {
public:
  SDFG(Location location);
  void addEdge(InterstateEdge edge);

private:
  // LUT (id -> node)
};

class State : public ContainerNode {
public:
  void addEdge(MultiEdge edge);

private:
  // LUT (id -> node)
};

class ConnectorNode : public Node {

private:
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
