#ifndef SDIR_StreamPopOp
#define SDIR_StreamPopOp

def SDIR_StreamPopOp : SDIR_Op<"stream_pop", [
    HasParent<"StateOp">,
    TypesMatchWith<"result type matches element type of 'stream'",
                     "str", "res",
                     "$_self.cast<StreamType>().getElementType()">
]> {
  let summary = "Stream pop operation";
  let description = [{
    Allows a state to pop a value from a stream.
    For example:

    ```mlir
    sdir.state @state_0 {
        ...
        %a = sdir.stream_pop %A : !sdir.stream<i32> -> i32
        ...
    }
    ```
  }];

    let arguments = (ins SDIR_StreamType:$str);
    let results = (outs AnyType:$res);
    let assemblyFormat = [{attr-dict $str `:` type($str) `->` type($res)}];
}

#endif // SDIR_StreamPopOp
