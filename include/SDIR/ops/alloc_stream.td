#ifndef SDIR_AllocStreamOp
#define SDIR_AllocStreamOp

def SDIR_AllocStreamOp : SDIR_Op<"alloc_stream", [
    HasParent<"ModuleOp">
]> {
  let summary = "Stream allocation operation";
  let description = [{
    Alloc operation to create streams.
    For example:

    ```mlir
        %A = sdir.alloc_stream() : !sdir.stream<i32>
    ```
  }];

    let results = (outs SDIR_StreamType);
    let assemblyFormat = [{attr-dict `(` `)` `:` type(results)}];
}

#endif // SDIR_AllocStreamOp