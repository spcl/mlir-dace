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

  let arguments = (ins SDIR_ArrayType:$arr);
  let results = (outs SDIR_MemletType:$res);
  let assemblyFormat = [{attr-dict $arr `:` type($arr) `->` type($res)}];

  let verifier = [{
    ::mlir::sdir::ArrayType arr = GetAccessOp::arr().getType().cast<ArrayType>();
    ::mlir::sdir::MemletType mem = GetAccessOp::res().getType().cast<MemletType>();
    if( arr.getElementType() != mem.getElementType() )
      return emitOpError("failed to verify that result element type matches element type of 'array'");

    return ::mlir::success();
  }];
}

#endif // SDIR_GetAccessOp