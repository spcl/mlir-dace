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
LogicalResult collect(AllocOp &op, State &state);

LogicalResult collect(TaskletNode &op, State &state);
LogicalResult collect(CopyOp &op, State &state);

} // namespace mlir::sdfg::translation

#endif // SDFG_Translation_H
