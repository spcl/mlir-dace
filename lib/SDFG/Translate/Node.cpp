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
  jemit.startNamedObject("data");
  jemit.printKVPair("type", "InterstateEdge");
  // label

  jemit.startNamedObject("attributes");

  jemit.startNamedObject("assignments");
  for (Assignment a : assignments)
    jemit.printKVPair(a.key, a.value);
  jemit.endObject(); // assignments

  jemit.startNamedObject("condition");
  jemit.printKVPair("string_data", condition.condition);
  jemit.printKVPair("language", "Python");
  jemit.endObject(); // condition

  jemit.endObject(); // attributes
  jemit.endObject(); // data
  jemit.endObject(); // attributes

  jemit.endObject();
}

void InterstateEdge::setCondition(Condition condition) {
  this->condition = condition;
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
  jemit.printKVPair("start_state", startState->getID(),
                    /*stringify=*/false);

  jemit.startNamedObject("attributes");
  jemit.printKVPair("name", name);

  jemit.startNamedList("arg_names");
  jemit.endList(); // arg_names

  jemit.startNamedObject("constants_prop");
  jemit.endObject(); // constants_prop

  jemit.startNamedObject("_arrays");
  jemit.endObject(); // _arrays

  jemit.startNamedObject("symbols");
  jemit.endObject(); // symbols

  jemit.endObject(); // attributes

  jemit.startNamedList("nodes");
  for (State s : states)
    s->emit(jemit);
  jemit.endList(); // nodes

  jemit.startNamedList("edges");
  for (InterstateEdge e : edges)
    e.emit(jemit);
  jemit.endList(); // edges

  jemit.endObject();
}

void SDFG::addState(unsigned id, State state) {
  state->setParent(this);
  state->setID(states.size());
  states.push_back(state);

  if (!lut.insert({id, state}).second)
    emitError(location, "Duplicate ID in SDFG::addState");
}

void SDFG::setStartState(State state) {
  if (std::find(states.begin(), states.end(), state) == states.end())
    emitError(location,
              "Non-existent state assigned as start in SDFG::setStartState");
  else
    this->startState = state;
}

void SDFG::addEdge(InterstateEdge edge) { edges.push_back(edge); }

State SDFG::lookup(unsigned id) { return lut.find(id)->second; }

//===----------------------------------------------------------------------===//
// State
//===----------------------------------------------------------------------===//

void StateImpl::emit(emitter::JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("type", "SDFGState");
  jemit.printKVPair("label", name);
  jemit.printKVPair("id", id, /*stringify=*/false);

  jemit.startNamedObject("attributes");
  jemit.endObject(); // attributes

  jemit.startNamedList("nodes");
  jemit.endList(); // nodes

  jemit.startNamedList("edges");
  jemit.endList(); // edges

  jemit.endObject();
}
