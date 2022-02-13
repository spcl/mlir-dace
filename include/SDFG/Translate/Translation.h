#ifndef SDFG_Translation_H
#define SDFG_Translation_H

#include "SDFG/Dialect/Dialect.h"
#include "SDFG/Translate/JsonEmitter.h"

using namespace mlir::sdfg::emitter;

namespace mlir::sdfg::translation {

void registerToSDFGTranslation();

LogicalResult translateToSDFG(ModuleOp &op, JsonEmitter &jemit);
LogicalResult translateToSDFG(SDFGNode &op, JsonEmitter &jemit);
LogicalResult translateToSDFG(StateNode &op, JsonEmitter &jemit);

LogicalResult translateToSDFG(TaskletNode &op, JsonEmitter &jemit);
LogicalResult translateToSDFG(MapNode &op, JsonEmitter &jemit);
LogicalResult translateToSDFG(ConsumeNode &op, JsonEmitter &jemit);
LogicalResult translateToSDFG(EdgeOp &op, JsonEmitter &jemit);

LogicalResult translateToSDFG(AllocOp &op, JsonEmitter &jemit);
LogicalResult translateToSDFG(LoadOp &op, JsonEmitter &jemit);
LogicalResult translateToSDFG(StoreOp &op, JsonEmitter &jemit);
LogicalResult translateToSDFG(CopyOp &op, JsonEmitter &jemit);

LogicalResult translateToSDFG(MemletCastOp &op, JsonEmitter &jemit);
LogicalResult translateToSDFG(ViewCastOp &op, JsonEmitter &jemit);
LogicalResult translateToSDFG(SubviewOp &op, JsonEmitter &jemit);

LogicalResult translateToSDFG(StreamPopOp &op, JsonEmitter &jemit);
LogicalResult translateToSDFG(StreamPushOp &op, JsonEmitter &jemit);
LogicalResult translateToSDFG(StreamLengthOp &op, JsonEmitter &jemit);

LogicalResult translateToSDFG(sdfg::CallOp &op, JsonEmitter &jemit);
LogicalResult translateToSDFG(LibCallOp &op, JsonEmitter &jemit);

LogicalResult translateToSDFG(AllocSymbolOp &op, JsonEmitter &jemit);
LogicalResult translateToSDFG(SymOp &op, JsonEmitter &jemit);

StringRef translateTypeToSDFG(Type &t, Location &loc);
inline void printDebuginfo(Operation &op, JsonEmitter &jemit);

void prepForTranslation(SDFGNode &op);
void prepForTranslation(StateNode &op);
void prepForTranslation(sdfg::CallOp &op);

} // namespace mlir::sdfg::translation

#endif // SDFG_Translation_H
