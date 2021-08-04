#ifndef SDIR_StreamPushOp
#define SDIR_StreamPushOp

def SDIR_StreamPushOp : SDIR_Op<"stream_push", [
    HasParent<"StateOp">,
    TypesMatchWith<"value type matches element type of 'stream'",
                  "str", "val",
                  "$_self.cast<StreamType>().getElementType()">
]> {
  let summary = "Stream push operation";
  let description = [{
    Allows a state to push a value into a stream.
    For example:

    ```mlir
    sdir.state @state_0 {
        ...
        %1 = constant 1 : i32
        sdir.stream_push %1, %A : !sdir.stream<i32>
        ...
    }
    ```
  }];

    let arguments = (ins SDIR_StreamType:$str, AnyType:$val);
    let assemblyFormat = [{attr-dict $val `,` $str `:` type($str)}];
}

#endif // SDIR_StreamPushOp