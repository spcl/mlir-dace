#ifndef SDFG_OPS
#define SDFG_OPS

include "SDFG/Dialect/Dialect.td"

//===----------------------------------------------------------------------===//
// Base class for all SDFG ops
//===----------------------------------------------------------------------===//

class SDFG_Op<string mnemonic, list<Trait> traits = []> : 
                                            Op<SDFG_Dialect, mnemonic, traits>{
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Nodes
//===----------------------------------------------------------------------===//

include "SDFG/Dialect/nodes/sdfg.td"
include "SDFG/Dialect/nodes/state.td"
include "SDFG/Dialect/nodes/tasklet.td"
include "SDFG/Dialect/nodes/map.td"
include "SDFG/Dialect/nodes/consume.td"

//===----------------------------------------------------------------------===//
// Ops
//===----------------------------------------------------------------------===//

include "SDFG/Dialect/ops/tasklet.td"
include "SDFG/Dialect/ops/memlet.td"
include "SDFG/Dialect/ops/stream.td"
include "SDFG/Dialect/ops/symbol.td"
include "SDFG/Dialect/ops/edge.td"

#endif // SDFG_OPS
