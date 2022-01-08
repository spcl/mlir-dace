#ifndef SDIR_Translation_H
#define SDIR_Translation_H

#include "SDIR/Dialect/Dialect.h"
#include "SDIR/Translate/JsonEmitter.h"

using namespace mlir::sdir::emitter;

namespace mlir::sdir::translation {

void registerToSDFGTranslation();

LogicalResult translateToSDFG(ModuleOp &op, JsonEmitter &jemit);
LogicalResult translateToSDFG(SDFGNode &op, JsonEmitter &jemit);
LogicalResult translateToSDFG(StateNode &op, JsonEmitter &jemit);

LogicalResult translateToSDFG(TaskletNode &op, JsonEmitter &jemit);
LogicalResult translateToSDFG(MapNode &op, JsonEmitter &jemit);
LogicalResult translateToSDFG(ConsumeNode &op, JsonEmitter &jemit);
LogicalResult translateToSDFG(EdgeOp &op, JsonEmitter &jemit);

LogicalResult translateToSDFG(AllocOp &op, JsonEmitter &jemit);
LogicalResult translateToSDFG(AllocTransientOp &op, JsonEmitter &jemit);
LogicalResult translateToSDFG(GetAccessOp &op, JsonEmitter &jemit);
LogicalResult translateToSDFG(LoadOp &op, JsonEmitter &jemit);

LogicalResult translateToSDFG(StoreOp &op, JsonEmitter &jemit);
LogicalResult translateToSDFG(CopyOp &op, JsonEmitter &jemit);
LogicalResult translateToSDFG(MemletCastOp &op, JsonEmitter &jemit);
LogicalResult translateToSDFG(ViewCastOp &op, JsonEmitter &jemit);
LogicalResult translateToSDFG(SubviewOp &op, JsonEmitter &jemit);

LogicalResult translateToSDFG(AllocStreamOp &op, JsonEmitter &jemit);
LogicalResult translateToSDFG(AllocTransientStreamOp &op, JsonEmitter &jemit);
LogicalResult translateToSDFG(StreamPopOp &op, JsonEmitter &jemit);
LogicalResult translateToSDFG(StreamPushOp &op, JsonEmitter &jemit);

LogicalResult translateToSDFG(StreamLengthOp &op, JsonEmitter &jemit);
LogicalResult translateToSDFG(sdir::CallOp &op, JsonEmitter &jemit);
LogicalResult translateToSDFG(LibCallOp &op, JsonEmitter &jemit);

LogicalResult translateToSDFG(AllocSymbolOp &op, JsonEmitter &jemit);
LogicalResult translateToSDFG(SymOp &op, JsonEmitter &jemit);

StringRef translateTypeToSDFG(Type &t, Location &loc, JsonEmitter &jemit);
inline void printDebuginfo(Operation &op, JsonEmitter &jemit);

} // namespace mlir::sdir::translation

#endif // SDIR_Translation_H
