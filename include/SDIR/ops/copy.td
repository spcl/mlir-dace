#ifndef SDIR_CopyOp
#define SDIR_CopyOp

def SDIR_CopyOp : SDIR_Op<"copy", [
    HasParent<"StateOp">,
    SameTypeOperands
]> {
  let summary = "Memlet copy operation";
  let description = [{
    Allows a state to copy the contents from one memlet to another.
    For example:

    ```mlir
    sdir.state @state_0 {
        ...
        sdir.copy %a -> %c : !sdir.memlet<i32>
        ...
    }
    ```
  }];

    let arguments = (ins SDIR_MemletType:$src, SDIR_MemletType:$dest);
    let assemblyFormat = [{attr-dict $src `->` $dest `:` type($src)}];
}

#endif // SDIR_CopyOp