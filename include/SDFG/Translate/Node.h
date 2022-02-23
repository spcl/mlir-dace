#ifndef SDFG_Translation_Node_H
#define SDFG_Translation_Node_H

#include "SDFG/Dialect/Dialect.h"
#include "SDFG/Translate/JsonEmitter.h"
#include "mlir/IR/Location.h"

namespace mlir::sdfg::translation {
class Node;
class Edge;
class Attribute;

class Node {
public:
  enum NodeType { SDFG, STATE, ACCESS, TASKLET };

  Node(NodeType type, Location location);
  void emit(emitter::JsonEmitter &jemit);
  void addEdge(Edge edge);
  void addAttribute(Attribute attribute);

private:
  // LUT (id -> access node)

  int ID;
  NodeType type;
  Location location;
  std::vector<Node> children;
  std::vector<Edge> edges;
  std::vector<Attribute> attributes;
};

class Edge {
public:
  enum EdgeType { InterState, MultiEdge };

private:
  Node source;
  Node destination;

  // connectors
  mlir::sdfg::SizedType shape;
  EdgeType type;

  std::vector<Attribute> attributes;
};

class Attribute {
private:
  std::string name;
  // Store attribute or string?
};

} // namespace mlir::sdfg::translation

#endif // SDFG_Translation_Node_H
