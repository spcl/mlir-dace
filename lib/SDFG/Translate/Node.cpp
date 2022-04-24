#include "SDFG/Translate/Node.h"

using namespace mlir;
using namespace sdfg;
using namespace translation;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

std::string typeToString(Type t) {
  if (t.isInteger(32))
    return "int32";

  if (t.isInteger(64))
    return "int64";

  if (t.isF32())
    return "float32";

  if (t.isF64())
    return "float64";

  if (t.isIndex())
    return "int64";

  std::string type;
  llvm::raw_string_ostream typeStream(type);
  t.print(typeStream);

  return "Unsupported: " + type;
}

//===----------------------------------------------------------------------===//
// Array
//===----------------------------------------------------------------------===//

void Array::emit(emitter::JsonEmitter &jemit) {
  jemit.startNamedObject(name);

  if (shape.getShape().empty()) {
    jemit.printKVPair("type", "Scalar");
  } else {
    jemit.printKVPair("type", "Array");
  }

  jemit.startNamedObject("attributes");

  jemit.printKVPair("transient", transient ? "true" : "false",
                    /*stringify=*/false);
  jemit.printKVPair("dtype", typeToString(shape.getElementType()));

  jemit.startNamedList("shape");

  if (shape.getShape().empty()) {
    jemit.startEntry();
    jemit.printString("1");
  }

  unsigned intIdx = 0;
  unsigned symIdx = 0;
  SmallVector<std::string> strideList;

  for (unsigned i = 0; i < shape.getShape().size(); ++i) {
    jemit.startEntry();
    if (shape.getShape()[i]) {
      jemit.printString(std::to_string(shape.getIntegers()[intIdx]));
      strideList.push_back(std::to_string(shape.getIntegers()[intIdx]));
      ++intIdx;
    } else {
      jemit.printString(shape.getSymbols()[symIdx].str());
      strideList.push_back(shape.getSymbols()[symIdx].str());
      ++symIdx;
    }
  }

  jemit.endList(); // shape

  if (!shape.getShape().empty()) {
    jemit.startNamedList("strides");

    for (int i = strideList.size() - 1; i >= 0; --i) {
      jemit.startEntry();
      jemit.printString(i == 0 ? "1" : strideList[i]);
    }

    jemit.endList(); // strides
  }

  jemit.endObject(); // attributes
  jemit.endObject();
}

//===----------------------------------------------------------------------===//
// Range
//===----------------------------------------------------------------------===//

void Range::emit(emitter::JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("start", start);
  jemit.printKVPair("end", end);
  jemit.printKVPair("step", step);
  jemit.printKVPair("tile", tile);
  jemit.endObject();
}

//===----------------------------------------------------------------------===//
// InterstateEdge
//===----------------------------------------------------------------------===//

void InterstateEdge::setCondition(Condition condition) {
  ptr->setCondition(condition);
}

void InterstateEdge::addAssignment(Assignment assignment) {
  ptr->addAssignment(assignment);
}

void InterstateEdge::emit(emitter::JsonEmitter &jemit) { ptr->emit(jemit); }

void InterstateEdgeImpl::setCondition(Condition condition) {
  this->condition = condition;
}

void InterstateEdgeImpl::addAssignment(Assignment assignment) {
  assignments.push_back(assignment);
}

void InterstateEdgeImpl::emit(emitter::JsonEmitter &jemit) {
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

//===----------------------------------------------------------------------===//
// MultiEdge
//===----------------------------------------------------------------------===//

void MultiEdge::emit(emitter::JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("type", "MultiConnectorEdge");

  jemit.printKVPair("src", source.parent.getID());
  jemit.printKVPair("dst", destination.parent.getID());

  jemit.printKVPair("src_connector", source.name,
                    /*stringify=*/!source.isNull);

  jemit.printKVPair("dst_connector", destination.name,
                    /*stringify=*/!destination.isNull);

  jemit.startNamedObject("attributes");
  jemit.startNamedObject("data");
  jemit.printKVPair("type", "Memlet");
  jemit.startNamedObject("attributes");

  // more data

  jemit.endObject(); // attributes
  jemit.endObject(); // data
  jemit.endObject(); // attributes

  jemit.endObject();
}

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
void Node::emit(emitter::JsonEmitter &jemit) { ptr->emit(jemit); }

void NodeImpl::setID(unsigned id) { this->id = id; }
unsigned NodeImpl::getID() { return id; }

Location NodeImpl::getLocation() { return location; }

void NodeImpl::setName(StringRef name) {
  this->name = name.str();
  utils::sanitizeName(this->name);
}

StringRef NodeImpl::getName() { return name; }

void NodeImpl::setParent(Node parent) { this->parent = parent; }
Node NodeImpl::getParent() { return parent; }

void NodeImpl::addAttribute(Attribute attribute) {
  attributes.push_back(attribute);
}

void NodeImpl::emit(emitter::JsonEmitter &jemit) {}

//===----------------------------------------------------------------------===//
// ConnectorNode
//===----------------------------------------------------------------------===//

void ConnectorNode::addInConnector(Connector connector) {
  ptr->addInConnector(connector);
}
void ConnectorNode::addOutConnector(Connector connector) {
  ptr->addOutConnector(connector);
}
void ConnectorNode::emit(emitter::JsonEmitter &jemit) { ptr->emit(jemit); }

void ConnectorNodeImpl::addInConnector(Connector connector) {
  if (std::find(inConnectors.begin(), inConnectors.end(), connector) !=
      inConnectors.end()) {
    emitError(location,
              "Duplicate connector in ConnectorNodeImpl::addInConnector");
  }

  inConnectors.push_back(connector);
}

void ConnectorNodeImpl::addOutConnector(Connector connector) {
  if (std::find(outConnectors.begin(), outConnectors.end(), connector) !=
      outConnectors.end()) {
    emitError(location,
              "Duplicate connector in ConnectorNodeImpl::addOutConnector");
  }

  outConnectors.push_back(connector);
}

void ConnectorNodeImpl::emit(emitter::JsonEmitter &jemit) {
  jemit.startNamedObject("in_connectors");
  for (Connector c : inConnectors) {
    if (c.isNull)
      continue;
    jemit.printKVPair(c.name, "null", /*stringify=*/false);
  }
  jemit.endObject(); // in_connectors

  jemit.startNamedObject("out_connectors");
  for (Connector c : outConnectors) {
    if (c.isNull)
      continue;
    jemit.printKVPair(c.name, "null", /*stringify=*/false);
  }
  jemit.endObject(); // out_connectors
}

//===----------------------------------------------------------------------===//
// ScopeNode
//===----------------------------------------------------------------------===//

SDFG ScopeNode::getSDFG() { return ptr->getSDFG(); }
void ScopeNode::addNode(ConnectorNode node) { ptr->addNode(node); }
void ScopeNode::addEdge(MultiEdge edge) { ptr->addEdge(edge); }

void ScopeNode::mapConnector(Value value, Connector connector) {
  ptr->mapConnector(value, connector);
}

Connector ScopeNode::lookup(Value value) { return ptr->lookup(value); }
void ScopeNode::emit(emitter::JsonEmitter &jemit) { ptr->emit(jemit); }

SDFG ScopeNodeImpl::getSDFG() { return static_cast<SDFG>(getParent()); }

void ScopeNodeImpl::addNode(ConnectorNode node) {
  node.setParent(ScopeNode(std::make_shared<ScopeNodeImpl>(*this)));
  node.setID(nodes.size());
  nodes.push_back(node);
}

void ScopeNodeImpl::addEdge(MultiEdge edge) { edges.push_back(edge); }

void ScopeNodeImpl::mapConnector(Value value, Connector connector) {
  auto res = lut.insert({utils::valueToString(value), connector});

  if (!res.second)
    res.first->second = connector;
}

Connector ScopeNodeImpl::lookup(Value value) {
  return lut.find(utils::valueToString(value))->second;
}

void ScopeNodeImpl::emit(emitter::JsonEmitter &jemit) {
  jemit.startNamedList("nodes");
  for (ConnectorNode cn : nodes)
    cn.emit(jemit);
  jemit.endList(); // nodes

  jemit.startNamedList("edges");
  for (MultiEdge me : edges)
    me.emit(jemit);
  jemit.endList(); // edges
}

//===----------------------------------------------------------------------===//
// SDFG
//===----------------------------------------------------------------------===//

void SDFG::addState(State state) {
  state.setParent(*this);
  ptr->addState(state);
}

State SDFG::lookup(StringRef name) { return ptr->lookup(name); }
void SDFG::setStartState(State state) { ptr->setStartState(state); }
void SDFG::addEdge(InterstateEdge edge) { ptr->addEdge(edge); }
void SDFG::addArray(Array array) { ptr->addArray(array); }
void SDFG::addArg(Array arg) { ptr->addArg(arg); }
void SDFG::emit(emitter::JsonEmitter &jemit) { ptr->emit(jemit); };
void SDFG::emitNested(emitter::JsonEmitter &jemit) { ptr->emitNested(jemit); };

unsigned SDFGImpl::list_id = 0;

void SDFGImpl::addState(State state) {
  state.setID(states.size());
  states.push_back(state);

  if (!lut.insert({state.getName().str(), state}).second)
    emitError(location, "Duplicate ID in SDFGImpl::addState");
}

void SDFGImpl::setStartState(State state) {
  if (std::find(states.begin(), states.end(), state) == states.end())
    emitError(
        location,
        "Non-existent state assigned as start in SDFGImpl::setStartState");
  else
    this->startState = state;
}

void SDFGImpl::addEdge(InterstateEdge edge) { edges.push_back(edge); }

State SDFGImpl::lookup(StringRef name) { return lut.find(name.str())->second; }

void SDFGImpl::addArray(Array array) { arrays.push_back(array); }
void SDFGImpl::addArg(Array arg) {
  args.push_back(arg);
  addArray(arg);
}

void SDFGImpl::emit(emitter::JsonEmitter &jemit) {
  jemit.startObject();
  emitBody(jemit);
}

void SDFGImpl::emitNested(emitter::JsonEmitter &jemit) {
  jemit.startNamedObject("sdfg");
  emitBody(jemit);
}

void SDFGImpl::emitBody(emitter::JsonEmitter &jemit) {
  jemit.printKVPair("type", "SDFG");
  jemit.printKVPair("sdfg_list_id", id, /*stringify=*/false);
  jemit.printKVPair("start_state", startState.getID(),
                    /*stringify=*/false);

  jemit.startNamedObject("attributes");
  jemit.printKVPair("name", name);

  jemit.startNamedList("arg_names");
  for (Array a : args) {
    jemit.startEntry();
    jemit.printString(a.name);
  }
  jemit.endList(); // arg_names

  jemit.startNamedObject("constants_prop");
  jemit.endObject(); // constants_prop

  jemit.startNamedObject("_arrays");
  for (Array a : arrays)
    a.emit(jemit);
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

//===----------------------------------------------------------------------===//
// NestedSDFG
//===----------------------------------------------------------------------===//

void NestedSDFG::emit(emitter::JsonEmitter &jemit) { ptr->emit(jemit); }

void NestedSDFGImpl::emit(emitter::JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("type", "NestedSDFG");
  jemit.printKVPair("id", id, /*stringify=*/false);

  jemit.startNamedObject("attributes");
  jemit.printKVPair("label", name);
  ConnectorNodeImpl::emit(jemit);
  sdfg.emitNested(jemit);

  jemit.endObject(); // attributes
  jemit.endObject();
}

//===----------------------------------------------------------------------===//
// State
//===----------------------------------------------------------------------===//

Connector State::lookup(Value value) { return ptr->lookup(value); }
void State::emit(emitter::JsonEmitter &jemit) { ptr->emit(jemit); }

Connector StateImpl::lookup(Value value) {
  if (lut.find(utils::valueToString(value)) == lut.end()) {
    Access access(location);
    access.setName(utils::valueToString(value));
    addNode(access);

    Connector accOut(access);
    access.addOutConnector(accOut);

    return accOut;
  }

  return ScopeNodeImpl::lookup(value);
}

void StateImpl::emit(emitter::JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("type", "SDFGState");
  jemit.printKVPair("label", name);
  jemit.printKVPair("id", id, /*stringify=*/false);

  jemit.startNamedObject("attributes");
  jemit.endObject(); // attributes

  ScopeNodeImpl::emit(jemit);
  jemit.endObject();
}

//===----------------------------------------------------------------------===//
// Tasklet
//===----------------------------------------------------------------------===//

void Tasklet::setCode(StringRef code) { ptr->setCode(code); }
void Tasklet::setLanguage(StringRef language) { ptr->setLanguage(language); }
void Tasklet::emit(emitter::JsonEmitter &jemit) { ptr->emit(jemit); }

void TaskletImpl::setCode(StringRef code) { this->code = code.str(); }
void TaskletImpl::setLanguage(StringRef language) {
  this->language = language.str();
}

void TaskletImpl::emit(emitter::JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("type", "Tasklet");
  jemit.printKVPair("label", name);
  jemit.printKVPair("id", id, /*stringify=*/false);

  jemit.startNamedObject("attributes");
  jemit.printKVPair("label", name);

  jemit.startNamedObject("code");
  jemit.printKVPair("string_data", code);
  jemit.printKVPair("language", language);
  jemit.endObject(); // code

  ConnectorNodeImpl::emit(jemit);
  jemit.endObject(); // attributes

  jemit.endObject();
}

//===----------------------------------------------------------------------===//
// Access
//===----------------------------------------------------------------------===//

void Access::emit(emitter::JsonEmitter &jemit) { ptr->emit(jemit); }

void AccessImpl::emit(emitter::JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("type", "AccessNode");
  jemit.printKVPair("label", name);
  jemit.printKVPair("id", id, /*stringify=*/false);

  jemit.startNamedObject("attributes");
  jemit.printKVPair("data", name);
  ConnectorNodeImpl::emit(jemit);
  jemit.endObject(); // attributes */

  jemit.endObject();
}

//===----------------------------------------------------------------------===//
// Map
//===----------------------------------------------------------------------===//

void MapEntry::addParam(StringRef param) { ptr->addParam(param); }
void MapEntry::addRange(Range range) { ptr->addRange(range); }
void MapEntry::setExit(MapExit exit) { ptr->setExit(exit); }
void MapEntry::emit(emitter::JsonEmitter &jemit) { ptr->emit(jemit); }

void MapEntryImpl::addParam(StringRef param) { params.push_back(param.str()); }
void MapEntryImpl::addRange(Range range) { ranges.push_back(range); }
void MapEntryImpl::setExit(MapExit exit) { this->exit = exit; }

void MapEntryImpl::emit(emitter::JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("type", "MapEntry");
  jemit.printKVPair("label", name);
  jemit.printKVPair("scope_exit", exit.getID());
  jemit.printKVPair("id", id, /*stringify=*/false);

  jemit.startNamedObject("attributes");
  jemit.printKVPair("label", name);

  jemit.startNamedList("params");
  for (std::string s : params) {
    jemit.startEntry();
    jemit.printString(s);
  }
  jemit.endList(); // params

  jemit.startNamedObject("range");
  jemit.printKVPair("type", "Range");
  jemit.startNamedList("ranges");
  for (Range r : ranges)
    r.emit(jemit);
  jemit.endList();   // ranges
  jemit.endObject(); // range

  ConnectorNodeImpl::emit(jemit);
  jemit.endObject(); // attributes */

  jemit.endObject();
}

void MapExit::setEntry(MapEntry entry) { ptr->setEntry(entry); }
void MapExit::emit(emitter::JsonEmitter &jemit) { ptr->emit(jemit); }

void MapExitImpl::setEntry(MapEntry entry) { this->entry = entry; }

void MapExitImpl::emit(emitter::JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("type", "MapExit");
  jemit.printKVPair("label", name);
  jemit.printKVPair("scope_entry", entry.getID());
  jemit.printKVPair("scope_exit", id);
  jemit.printKVPair("id", id, /*stringify=*/false);

  jemit.startNamedObject("attributes");
  ConnectorNodeImpl::emit(jemit);
  jemit.endObject(); // attributes */

  jemit.endObject();
}
