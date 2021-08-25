#ifndef SDIR_OPS
#define SDIR_OPS

include "SDIR/SDIR_Dialect.td"

//===----------------------------------------------------------------------===//
// Base class for all SDIR ops
//===----------------------------------------------------------------------===//

class SDIR_Op<string mnemonic, list<OpTrait> traits = []> : Op<SDIR_Dialect, mnemonic, traits>{}

//===----------------------------------------------------------------------===//
// Nodes
//===----------------------------------------------------------------------===//

include "SDIR/nodes/sdfg.td"
include "SDIR/nodes/state.td"
include "SDIR/nodes/tasklet.td"
include "SDIR/nodes/map.td"
include "SDIR/nodes/consume.td"

//===----------------------------------------------------------------------===//
// Ops
//===----------------------------------------------------------------------===//

include "SDIR/ops/edge.td"
include "SDIR/ops/tasklet.td"
include "SDIR/ops/memlet.td"
include "SDIR/ops/stream.td"
include "SDIR/ops/get_access.td"

#endif // SDIR_OPS
