#ifndef SDIR_OPS
#define SDIR_OPS

include "SDIR/Dialect/Dialect.td"

//===----------------------------------------------------------------------===//
// Base class for all SDIR ops
//===----------------------------------------------------------------------===//

class SDIR_Op<string mnemonic, list<OpTrait> traits = []> : 
                                            Op<SDIR_Dialect, mnemonic, traits>{
    let parser = [{ return ::parse$cppClass(parser, result); }];
    let printer = [{ ::print(p, *this); }];
    let verifier = [{ return ::verify(*this); }];
}

//===----------------------------------------------------------------------===//
// Nodes
//===----------------------------------------------------------------------===//

include "SDIR/Dialect/nodes/sdfg.td"
include "SDIR/Dialect/nodes/state.td"
include "SDIR/Dialect/nodes/tasklet.td"
include "SDIR/Dialect/nodes/map.td"
include "SDIR/Dialect/nodes/consume.td"

//===----------------------------------------------------------------------===//
// Ops
//===----------------------------------------------------------------------===//

include "SDIR/Dialect/ops/tasklet.td"
include "SDIR/Dialect/ops/memlet.td"
include "SDIR/Dialect/ops/stream.td"
include "SDIR/Dialect/ops/symbol.td"
include "SDIR/Dialect/ops/edge.td"

#endif // SDIR_OPS
