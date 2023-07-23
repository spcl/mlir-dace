// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

#ifndef SDFG_Translation_H
#define SDFG_Translation_H

#include "SDFG/Dialect/Dialect.h"
#include "SDFG/Translate/JsonEmitter.h"
#include "SDFG/Translate/Node.h"

using namespace mlir::sdfg::emitter;

namespace mlir::sdfg::translation {
void registerToSDFGTranslation();

LogicalResult translateToSDFG(ModuleOp &op, JsonEmitter &jemit);

LogicalResult collect(StateNode &op, SDFG &sdfg);
LogicalResult collect(EdgeOp &op, SDFG &sdfg);
LogicalResult collect(AllocOp &op, SDFG &sdfg);
LogicalResult collect(AllocSymbolOp &op, SDFG &sdfg);

LogicalResult collect(AllocOp &op, ScopeNode &scope);
LogicalResult collect(TaskletNode &op, ScopeNode &scope);
LogicalResult collect(NestedSDFGNode &op, ScopeNode &scope);
LogicalResult collect(LibCallOp &op, ScopeNode &scope);

LogicalResult collect(MapNode &op, ScopeNode &scope);
LogicalResult collect(ConsumeNode &op, ScopeNode &scope);
LogicalResult collect(CopyOp &op, ScopeNode &scope);
LogicalResult collect(StoreOp &op, ScopeNode &scope);

LogicalResult collect(LoadOp &op, ScopeNode &scope);
LogicalResult collect(AllocSymbolOp &op, ScopeNode &scope);
LogicalResult collect(SymOp &op, ScopeNode &scope);
LogicalResult collect(StreamPushOp &op, ScopeNode &scope);

LogicalResult collect(StreamPopOp &op, ScopeNode &scope);

} // namespace mlir::sdfg::translation

#endif // SDFG_Translation_H
