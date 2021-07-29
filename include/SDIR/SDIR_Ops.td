#ifndef SDIR_OPS
#define SDIR_OPS

include "SDIR/SDIR_Dialect.td"
include "SDIR/SDIR_Types.td"

//===----------------------------------------------------------------------===//
// Base class for all SDIR ops
//===----------------------------------------------------------------------===//

class SDIR_Op<string mnemonic, list<OpTrait> traits = []> : Op<SDIR_Dialect, mnemonic, traits>{}

include "SDIR/ops/state.td"
include "SDIR/ops/tasklet.td"
include "SDIR/ops/return.td"

#endif // SDIR_OPS