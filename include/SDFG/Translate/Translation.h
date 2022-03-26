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
LogicalResult collect(TaskletNode &op, State &state);

} // namespace mlir::sdfg::translation

#endif // SDFG_Translation_H
