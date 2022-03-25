#include "SDFG/Translate/Node.h"

using namespace mlir;
using namespace sdfg;
using namespace translation;

//===----------------------------------------------------------------------===//
// InterstateEdge
//===----------------------------------------------------------------------===//

void InterstateEdge::emit(emitter::JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("type", "Edge");
  jemit.printKVPair("src", source.getID());
  jemit.printKVPair("dst", destination.getID());

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
// MultiEdge
//===----------------------------------------------------------------------===//

void MultiEdge::emit(emitter::JsonEmitter &jemit) {}

//===----------------------------------------------------------------------===//
// Node
//===----------------------------------------------------------------------===//

void Node::setID(unsigned id) { ptr->setID(id); }
unsigned Node::getID() { return ptr->getID(); }

Location Node::getLocation() { return ptr->getLocation(); }

void Node::setName(StringRef name) { ptr->setName(name); }
StringRef Node::getName() { return ptr->getName(); }

void Node::setParent(Node parent) { ptr->setParent(parent); }
Node Node::getParent() { return ptr->getParent(); }

void Node::addAttribute(Attribute attribute) { ptr->addAttribute(attribute); }

void NodeImpl::addAttribute(Attribute attribute) {
  attributes.push_back(attribute);
}

//===----------------------------------------------------------------------===//
// ConnectorNode
//===----------------------------------------------------------------------===//

void ConnectorNode::emit(emitter::JsonEmitter &jemit) { ptr->emit(jemit); }
void ConnectorNodeImpl::emit(emitter::JsonEmitter &jemit) {}

//===----------------------------------------------------------------------===//
// SDFG
//===----------------------------------------------------------------------===//

void SDFG::addState(unsigned id, State state) {
  state.setParent(*this);
  ptr->addState(id, state);
}

State SDFG::lookup(unsigned id) { return ptr->lookup(id); }
void SDFG::setStartState(State state) { ptr->setStartState(state); }
void SDFG::addEdge(InterstateEdge edge) { ptr->addEdge(edge); }
void SDFG::emit(emitter::JsonEmitter &jemit) { ptr->emit(jemit); };

void SDFGImpl::emit(emitter::JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("type", "SDFG");
  jemit.printKVPair("sdfg_list_id", id, /*stringify=*/false);
  jemit.printKVPair("start_state", startState.getID(),
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
    s.emit(jemit);
  jemit.endList(); // nodes

  jemit.startNamedList("edges");
  for (InterstateEdge e : edges)
    e.emit(jemit);
  jemit.endList(); // edges

  jemit.endObject();
}

void SDFGImpl::addState(unsigned id, State state) {
  state.setID(states.size());
  states.push_back(state);

  if (!lut.insert({id, state}).second)
    emitError(location, "Duplicate ID in SDFG::addState");
}

void SDFGImpl::setStartState(State state) {
  if (std::find(states.begin(), states.end(), state) == states.end())
    emitError(location,
              "Non-existent state assigned as start in SDFG::setStartState");
  else
    this->startState = state;
}

void SDFGImpl::addEdge(InterstateEdge edge) { edges.push_back(edge); }

State SDFGImpl::lookup(unsigned id) { return lut.find(id)->second; }

//===----------------------------------------------------------------------===//
// State
//===----------------------------------------------------------------------===//

void State::addNode(ConnectorNode node) { ptr->addNode(node); }
void State::addEdge(MultiEdge edge) { ptr->addEdge(edge); }
void State::emit(emitter::JsonEmitter &jemit) { ptr->emit(jemit); }

void StateImpl::addNode(ConnectorNode node) { nodes.push_back(node); }

void StateImpl::emit(emitter::JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("type", "SDFGState");
  jemit.printKVPair("label", name);
  jemit.printKVPair("id", id, /*stringify=*/false);

  jemit.startNamedObject("attributes");
  jemit.endObject(); // attributes

  jemit.startNamedList("nodes");
  for (ConnectorNode cn : nodes)
    cn.emit(jemit);
  jemit.endList(); // nodes

  jemit.startNamedList("edges");
  jemit.endList(); // edges

  jemit.endObject();
}

//===----------------------------------------------------------------------===//
// Tasklet
//===----------------------------------------------------------------------===//

void Tasklet::emit(emitter::JsonEmitter &jemit) { ptr->emit(jemit); }

void TaskletImpl::emit(emitter::JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("type", "Tasklet");
  jemit.printKVPair("label", name);
  jemit.printKVPair("id", id, /*stringify=*/false);

  jemit.startNamedObject("attributes");
  jemit.printKVPair("label", name);

  jemit.startNamedObject("code");
  jemit.printKVPair("string_data", "");
  jemit.printKVPair("language", "Python");
  jemit.endObject(); // code

  // Superclass should print these
  jemit.startNamedObject("in_connectors");
  jemit.endObject(); // in_connectors

  jemit.startNamedObject("out_connectors");
  jemit.endObject(); // out_connectors

  jemit.endObject(); // attributes

  jemit.endObject();
}
