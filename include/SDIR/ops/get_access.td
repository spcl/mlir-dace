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

  let arguments = (ins AnyTypeOf<[SDIR_ArrayType, SDIR_StreamArrayType]>:$arr);
  let results = (outs AnyTypeOf<[SDIR_MemletType, SDIR_StreamType]>:$res);
  let assemblyFormat = [{attr-dict $arr `:` type($arr) `->` type($res)}];

  let verifier = [{
    ::mlir::Type arr = GetAccessOp::arr().getType();
    ::mlir::Type res = GetAccessOp::res().getType();

    if( arr.isa<ArrayType>() && res.isa<MemletType>() )
      if( arr.cast<ArrayType>().getElementType() != res.cast<MemletType>().getElementType() )
        return emitOpError("failed to verify that result element type matches element type of 'array'");

    if( arr.isa<StreamArrayType>() && res.isa<StreamType>() )
      if( arr.cast<StreamArrayType>().getElementType() != res.cast<StreamType>().getElementType() )
        return emitOpError("failed to verify that result element type matches element type of 'stream_array'");

    if( arr.isa<ArrayType>() && res.isa<StreamType>() )
      return emitOpError("failed to verify that result type matches derived type of 'array'");

    if( arr.isa<StreamArrayType>() && res.isa<MemletType>() )
      return emitOpError("failed to verify that result type matches derived type of 'stream_array'");

    return ::mlir::success();
  }];
}

#endif // SDIR_GetAccessOp
