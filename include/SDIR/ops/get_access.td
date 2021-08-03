#ifndef SDIR_GetAccessOp
#define SDIR_GetAccessOp

def SDIR_GetAccessOp : SDIR_Op<"get_access", [
    HasParent<"StateOp">
]> {
  let summary = "Array access operation";
  let description = [{
    Allows a state to access the specified array.
    For example:

    ```mlir
    sdir.state @state_0 {
        ...
        %a = sdir.get_access %A : !sdir.memlet<i32>
        ...
    }
    ```
  }];

    let arguments = (ins SDIR_MemletType:$arr);
    let results = (outs SDIR_MemletType);
    let assemblyFormat = [{attr-dict $arr `:` type($arr) `->` type(results)}];
}

#endif // SDIR_GetAccessOp