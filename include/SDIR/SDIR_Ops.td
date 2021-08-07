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

include "SDIR/ops/alloc.td"
include "SDIR/ops/alloc_transient.td"
include "SDIR/ops/get_access.td"
include "SDIR/ops/load.td"
include "SDIR/ops/store.td"
include "SDIR/ops/copy.td"

//===----------------------------------------------------------------------===//
// Stream ops
//===----------------------------------------------------------------------===//

include "SDIR/ops/alloc_stream.td"
include "SDIR/ops/alloc_transient_stream.td"
include "SDIR/ops/stream_pop.td"
include "SDIR/ops/stream_push.td"

#endif // SDIR_OPS