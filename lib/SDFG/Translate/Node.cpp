#include "SDFG/Translate/Node.h"
#include "SDFG/Utils/Utils.h"

using namespace mlir;
using namespace sdfg;
using namespace translation;

//===----------------------------------------------------------------------===//
// Edges
//===----------------------------------------------------------------------===//

Edge::Edge(Node *source, Node *destination)
    : source(source), destination(destination) {}

//===----------------------------------------------------------------------===//
// Nodes
//===----------------------------------------------------------------------===//

Node::Node(Location location) : location(location) { ID = utils::generateID(); }

void Node::emit(emitter::JsonEmitter &jemit) {}

void Node::addAttribute(Attribute attribute) {
  attributes.push_back(attribute);
}

ContainerNode::ContainerNode(Location location) : Node(location) {}

SDFG::SDFG(Location location) : ContainerNode(location) {}
