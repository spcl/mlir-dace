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

LogicalResult collect(AllocOp &op, ScopeNode &scope);
LogicalResult collect(TaskletNode &op, ScopeNode &scope);
LogicalResult collect(NestedSDFGNode &op, ScopeNode &scope);
LogicalResult collect(MapNode &op, ScopeNode &scope);

LogicalResult collect(ConsumeNode &op, ScopeNode &scope);
LogicalResult collect(CopyOp &op, ScopeNode &scope);
LogicalResult collect(StoreOp &op, ScopeNode &scope);
LogicalResult collect(LoadOp &op, ScopeNode &scope);

LogicalResult collect(AllocSymbolOp &op, ScopeNode &scope);
LogicalResult collect(SymOp &op, ScopeNode &scope);

} // namespace mlir::sdfg::translation

#endif // SDFG_Translation_H
