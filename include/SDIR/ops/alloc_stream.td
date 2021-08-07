#ifndef SDIR_AllocStreamOp
#define SDIR_AllocStreamOp

def SDIR_AllocStreamOp : SDIR_Op<"alloc_stream", [
    ParentOneOf<["SDFGOp", "StateOp"]>
]> {
  let summary = "Stream allocation operation";
  let description = [{
    Alloc operation to create streams.
    For example:

    ```mlir
        %A = sdir.alloc_stream() : !sdir.stream_array<i32>
    ```
  }];

    let results = (outs SDIR_StreamArrayType);
    let assemblyFormat = [{attr-dict `(` `)` `:` type(results)}];
}

#endif // SDIR_AllocStreamOp