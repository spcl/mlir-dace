#include "SDFG/Translate/Node.h"

using namespace mlir;
using namespace sdfg;
using namespace translation;

//===----------------------------------------------------------------------===//
// Edges
//===----------------------------------------------------------------------===//

InterstateEdge::InterstateEdge(State *source, State *destination)
    : Edge(source, destination) {
  condition = nullptr;
  assignments.clear();
}

void InterstateEdge::setCondition(std::string condition) {
  this->condition = condition;
}

void InterstateEdge::addAssignment(Assignment assignment) {
  assignments.push_back(assignment);
}

//===----------------------------------------------------------------------===//
// Nodes
//===----------------------------------------------------------------------===//

void Node::setID(unsigned id) { this->id = id; }
unsigned Node::getID() { return id; }
void Node::setParent(Node *parent) { this->parent = parent; }
Node *Node::getParent() { return parent; }

void Node::addAttribute(Attribute attribute) {
  attributes.push_back(attribute);
}

void SDFG::emit(emitter::JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("type", "SDFG");
  jemit.printKVPair("sdfg_list_id", id, /*stringify=*/false);
  // jemit.printKVPair("start_state", entryState.ID(), /*stringify=*/false);

  jemit.startNamedObject("attributes");
  jemit.endObject(); // attributes

  jemit.startNamedList("nodes");
  for (State s : nodes)
    s.emit(jemit);
  jemit.endList(); // nodes

  jemit.startNamedList("edges");
  jemit.endList(); // edges

  jemit.endObject();
}

void SDFG::addNode(State &state) {
  state.setParent(this);
  state.setID(nodes.size());
  nodes.push_back(state);
}

void State::emit(emitter::JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("type", "SDFGState");
  // jemit.printKVPair("label", op.sym_name());
  jemit.printKVPair("id", id, /*stringify=*/false);

  jemit.startNamedObject("attributes");
  jemit.endObject(); // attributes

  jemit.startNamedList("nodes");
  jemit.endList(); // nodes

  jemit.startNamedList("edges");
  jemit.endList(); // edges

  jemit.endObject();
}
