#ifndef SDIR_StreamOps
#define SDIR_StreamOps

//===----------------------------------------------------------------------===//
// AllocStreamOp
//===----------------------------------------------------------------------===//

def SDIR_AllocStreamOp : SDIR_Op<"alloc_stream", [
    ParentOneOf<["SDFGNode", "StateNode"]>
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

//===----------------------------------------------------------------------===//
// AllocTransientStreamOp
//===----------------------------------------------------------------------===//

def SDIR_AllocTransientStreamOp : SDIR_Op<"alloc_transient_stream", [
    ParentOneOf<["SDFGNode", "StateNode"]>
]> {
  let summary = "Transient stream allocation operation";
  let description = [{
    Alloc operation to create transient streams.
    For example:

    ```mlir
        %A = sdir.alloc_transient_stream() : !sdir.stream_array<i32>
    ```
  }];

    let results = (outs SDIR_StreamArrayType);
    let assemblyFormat = [{attr-dict `(` `)` `:` type(results)}];
}

//===----------------------------------------------------------------------===//
// StreamPopOp
//===----------------------------------------------------------------------===//

def SDIR_StreamPopOp : SDIR_Op<"stream_pop", [
    HasParent<"StateNode">,
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

//===----------------------------------------------------------------------===//
// StreamPushOp
//===----------------------------------------------------------------------===//

def SDIR_StreamPushOp : SDIR_Op<"stream_push", [
    HasParent<"StateNode">,
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

#endif // SDIR_StreamOps