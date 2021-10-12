#ifndef SDIR_Translation_H
#define SDIR_Translation_H

#include "SDIR/Dialect/Dialect.h"
#include "SDIR/Translate/JsonEmitter.h"

using namespace mlir;
using namespace sdir;

void registerToSDFGTranslation();

LogicalResult translateToSDFG(Operation &op, JsonEmitter &jemit);
LogicalResult translateModuleToSDFG(ModuleOp &op, JsonEmitter &jemit);
LogicalResult translateSDFGToSDFG(SDFGNode &op, JsonEmitter &jemit);
LogicalResult translateStateToSDFG(StateNode &op, JsonEmitter &jemit);

LogicalResult translateTaskletToSDFG(TaskletNode &op, JsonEmitter &jemit);
LogicalResult translateMapToSDFG(MapNode &op, JsonEmitter &jemit);
LogicalResult translateConsumeToSDFG(ConsumeNode &op, JsonEmitter &jemit);
LogicalResult translateEdgeToSDFG(EdgeOp &op, JsonEmitter &jemit);

LogicalResult translateAllocToSDFG(AllocOp &op, JsonEmitter &jemit);
LogicalResult translateAllocTransientToSDFG(AllocTransientOp &op,
                                            JsonEmitter &jemit);
LogicalResult translateGetAccessToSDFG(GetAccessOp &op, JsonEmitter &jemit);
LogicalResult translateLoadToSDFG(LoadOp &op, JsonEmitter &jemit);

LogicalResult translateStoreToSDFG(StoreOp &op, JsonEmitter &jemit);
LogicalResult translateCopyToSDFG(CopyOp &op, JsonEmitter &jemit);
LogicalResult translateMemletCastToSDFG(MemletCastOp &op, JsonEmitter &jemit);
LogicalResult translateViewCastToSDFG(ViewCastOp &op, JsonEmitter &jemit);
LogicalResult translateSubviewToSDFG(SubviewOp &op, JsonEmitter &jemit);

LogicalResult translateAllocStreamToSDFG(AllocStreamOp &op, JsonEmitter &jemit);
LogicalResult translateAllocTransientStreamToSDFG(AllocTransientStreamOp &op,
                                                  JsonEmitter &jemit);
LogicalResult translateStreamPopToSDFG(StreamPopOp &op, JsonEmitter &jemit);
LogicalResult translateStreamPushToSDFG(StreamPushOp &op, JsonEmitter &jemit);

LogicalResult translateStreamLengthToSDFG(StreamLengthOp &op,
                                          JsonEmitter &jemit);
LogicalResult translateCallToSDFG(sdir::CallOp &op, JsonEmitter &jemit);
LogicalResult translateLibCallToSDFG(LibCallOp &op, JsonEmitter &jemit);

LogicalResult translateAllocSymbolToSDFG(AllocSymbolOp &op, JsonEmitter &jemit);
LogicalResult translateSymbolExprToSDFG(SymOp &op, JsonEmitter &jemit);

bool containsAttr(Operation &op, StringRef attrName);

#endif // SDIR_Translation_H
