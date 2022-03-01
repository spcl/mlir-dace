#include "SDFG/Translate/Node.h"

using namespace mlir;
using namespace sdfg;
using namespace translation;

//===----------------------------------------------------------------------===//
// Edges
//===----------------------------------------------------------------------===//

void InterstateEdge::emit(emitter::JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("type", "Edge");
  jemit.printKVPair("src", source->getID());
  jemit.printKVPair("dst", destination->getID());

  jemit.startNamedObject("attributes");
  jemit.endObject(); // attributes

  jemit.endObject();
}

void InterstateEdge::setCondition(StringRef condition) {
  this->condition = condition.str();
}

void InterstateEdge::addAssignment(Assignment assignment) {
  assignments.push_back(assignment);
}

//===----------------------------------------------------------------------===//
// Node
//===----------------------------------------------------------------------===//

void Node::addAttribute(Attribute attribute) {
  attributes.push_back(attribute);
}

//===----------------------------------------------------------------------===//
// SDFG
//===----------------------------------------------------------------------===//

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
  for (InterstateEdge e : edges)
    e.emit(jemit);
  jemit.endList(); // edges

  jemit.endObject();
}

void SDFG::addState(unsigned id, std::shared_ptr<State> state) {
  state->setParent(this);
  state->setID(nodes.size());
  nodes.push_back(*state);

  if (!lut.insert({id, state}).second)
    emitError(location, "Duplicate ID in SDFG::addState");
}

void SDFG::addEdge(InterstateEdge edge) { edges.push_back(edge); }

std::shared_ptr<State> SDFG::lookup(unsigned id) {
  return lut.find(id)->second;
}

//===----------------------------------------------------------------------===//
// State
//===----------------------------------------------------------------------===//

void State::emit(emitter::JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("type", "SDFGState");
  jemit.printKVPair("label", label);
  jemit.printKVPair("id", id, /*stringify=*/false);

  jemit.startNamedObject("attributes");
  jemit.endObject(); // attributes

  jemit.startNamedList("nodes");
  jemit.endList(); // nodes

  jemit.startNamedList("edges");
  jemit.endList(); // edges

  jemit.endObject();
}
