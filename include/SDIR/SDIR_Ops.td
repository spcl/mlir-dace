#ifndef SDIR_OPS
#define SDIR_OPS

include "SDIR/SDIR_Dialect.td"

//===----------------------------------------------------------------------===//
// Base class for all SDIR ops
//===----------------------------------------------------------------------===//

class SDIR_Op<string mnemonic, list<OpTrait> traits = []> : Op<SDIR_Dialect, mnemonic, traits>{}

include "SDIR/ops/state.td"
include "SDIR/ops/tasklet.td"
include "SDIR/ops/return.td"
include "SDIR/ops/alloc.td"
include "SDIR/ops/alloc_transient.td"
include "SDIR/ops/get_access.td"
include "SDIR/ops/load.td"
include "SDIR/ops/store.td"
include "SDIR/ops/copy.td"

include "SDIR/ops/alloc_stream.td"
include "SDIR/ops/stream_pop.td"
include "SDIR/ops/stream_push.td"

#endif // SDIR_OPS