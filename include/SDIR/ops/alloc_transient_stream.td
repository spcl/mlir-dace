#ifndef SDIR_AllocTransientStreamOp
#define SDIR_AllocTransientStreamOp

def SDIR_AllocTransientStreamOp : SDIR_Op<"alloc_transient_stream", [
    HasParent<"StateOp">
]> {
  let summary = "Transient stream allocation operation";
  let description = [{
    Alloc operation to create transient streams.
    For example:

    ```mlir
        %A = sdir.alloc_transient_stream() : !sdir.stream<i32>
    ```
  }];

    let results = (outs SDIR_StreamType);
    let assemblyFormat = [{attr-dict `(` `)` `:` type(results)}];
}

#endif // SDIR_AllocTransientStreamOp