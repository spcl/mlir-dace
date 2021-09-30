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
}

//===----------------------------------------------------------------------===//
// StreamPopOp
//===----------------------------------------------------------------------===//

def SDIR_StreamPopOp : SDIR_Op<"stream_pop", [
    ParentOneOf<["StateNode", "MapNode"]>,
    TypesMatchWith<"result type matches element type of 'stream'", "str", "res",
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
}

//===----------------------------------------------------------------------===//
// StreamPushOp
//===----------------------------------------------------------------------===//

def SDIR_StreamPushOp : SDIR_Op<"stream_push", [
    ParentOneOf<["StateNode", "MapNode"]>,
    TypesMatchWith<"value type matches element type of 'stream'", "str", "val",
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
            sdir.stream_push %1, %A : i32 -> !sdir.stream<i32>
            ...
        }
        ```
    }];

    let arguments = (ins AnyType:$val, SDIR_StreamType:$str);
}

//===----------------------------------------------------------------------===//
// StreamLengthOp
//===----------------------------------------------------------------------===//

def SDIR_StreamLengthOp : SDIR_Op<"stream_length", [
    ParentOneOf<["StateNode", "MapNode", "FuncOp"]>
]> {
    let summary = "Stream length operation";
    let description = [{
        Returns the length of the stream.
        For example:

        ```mlir
        sdir.state @state_0 {
            ...
            %l = sdir.stream_length %A : !sdir.stream<i32> -> i32
            ...
        }
        ```
    }];

    let arguments = (ins SDIR_StreamType:$str);
    let results = (outs I32);
}

#endif // SDIR_StreamOps
