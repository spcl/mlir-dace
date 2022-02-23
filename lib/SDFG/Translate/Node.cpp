#include "SDFG/Translate/Node.h"

using namespace mlir;
using namespace sdfg;
using namespace translation;

Node::Node(NodeType type, Location location) : type(type), location(location) {
  ID = 5;
}

void Node::emit(emitter::JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("type", ID);
  jemit.endObject();
}
