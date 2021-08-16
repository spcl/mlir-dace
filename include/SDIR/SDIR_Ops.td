#ifndef SDIR_OPS
#define SDIR_OPS

include "SDIR/SDIR_Dialect.td"

//===----------------------------------------------------------------------===//
// Base class for all SDIR ops
//===----------------------------------------------------------------------===//

class SDIR_Op<string mnemonic, list<OpTrait> traits = []> : Op<SDIR_Dialect, mnemonic, traits>{}

//===----------------------------------------------------------------------===//
// Region ops
//===----------------------------------------------------------------------===//

include "SDIR/ops/sdfg.td"
include "SDIR/ops/edge.td"
include "SDIR/ops/state.td"

//===----------------------------------------------------------------------===//
// Tasklet ops
//===----------------------------------------------------------------------===//

include "SDIR/ops/tasklet.td"
include "SDIR/ops/return.td"
include "SDIR/ops/call.td"
include "SDIR/ops/libcall.td"

//===----------------------------------------------------------------------===//
// Memlet ops
//===----------------------------------------------------------------------===//

include "SDIR/ops/memlet.td"
include "SDIR/ops/get_access.td"


//===----------------------------------------------------------------------===//
// Stream ops
//===----------------------------------------------------------------------===//

include "SDIR/ops/stream.td"

#endif // SDIR_OPS
