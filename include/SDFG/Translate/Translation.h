// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

#ifndef SDFG_Translation_H
#define SDFG_Translation_H

#include "SDFG/Dialect/Dialect.h"
#include "SDFG/Translate/JsonEmitter.h"
#include "SDFG/Translate/Node.h"

using namespace mlir::sdfg::emitter;

namespace mlir::sdfg::translation {
/// Registers SDFG to SDFG IR translation.
void registerToSDFGTranslation();

/// Translates a module containing SDFG dialect to SDFG IR, outputs the result
/// to the provided output stream.
LogicalResult translateToSDFG(ModuleOp &op, JsonEmitter &jemit);

/// Collects state node information in a top-level SDFG.
LogicalResult collect(StateNode &op, SDFG &sdfg);
/// Collects edge information in a top-level SDFG.
LogicalResult collect(EdgeOp &op, SDFG &sdfg);
/// Collects array/stream allocation information in a top-level SDFG.
LogicalResult collect(AllocOp &op, SDFG &sdfg);
/// Collects symbol allocation information in a top-level SDFG.
LogicalResult collect(AllocSymbolOp &op, SDFG &sdfg);

/// Collects array/stream allocation information in a scope.
LogicalResult collect(AllocOp &op, ScopeNode &scope);
/// Collects tasklet information in a scope.
LogicalResult collect(TaskletNode &op, ScopeNode &scope);
/// Collects nested SDFG node information in a scope.
LogicalResult collect(NestedSDFGNode &op, ScopeNode &scope);
/// Collects library call information in a scope.
LogicalResult collect(LibCallOp &op, ScopeNode &scope);

/// Collects map node information in a scope.
LogicalResult collect(MapNode &op, ScopeNode &scope);
/// Collects consume node information in a scope.
LogicalResult collect(ConsumeNode &op, ScopeNode &scope);
/// Collects copy operation information in a scope.
LogicalResult collect(CopyOp &op, ScopeNode &scope);
/// Collects store operation information in a scope.
LogicalResult collect(StoreOp &op, ScopeNode &scope);

/// Collects load operation information in a scope.
LogicalResult collect(LoadOp &op, ScopeNode &scope);
/// Collects symbol allocation information in a scope.
LogicalResult collect(AllocSymbolOp &op, ScopeNode &scope);
/// Collects symbolic expression information in a scope.
LogicalResult collect(SymOp &op, ScopeNode &scope);
/// Collects stream push operation information in a scope.
LogicalResult collect(StreamPushOp &op, ScopeNode &scope);

/// Collects stream pop operation information in a scope.
LogicalResult collect(StreamPopOp &op, ScopeNode &scope);
} // namespace mlir::sdfg::translation

#endif // SDFG_Translation_H
