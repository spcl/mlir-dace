// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

/// Table-driven file for stream operations.

#ifndef SDFG_StreamOps
#define SDFG_StreamOps

//===----------------------------------------------------------------------===//
// StreamPopOp
//===----------------------------------------------------------------------===//

/// Defining part of the SDFG stream memlet.
def SDFG_StreamPopOp : SDFG_Op<"stream_pop", [
    ParentOneOf<["StateNode", "MapNode"]>,
    TypesMatchWith<"result type matches element type of 'stream'", "str", "res",
                    "$_self.cast<StreamType>().getDimensions().getElementType()">
]> {
    let summary = "Stream pop operation";
    let description = [{
        Allows a state to pop a value from a stream.
        For example:

        ```mlir
        sdfg.state @state_0 {
            ...
            %a = sdfg.stream_pop %A : !sdfg.stream<i32> -> i32
            ...
        }
        ```
    }];

    let arguments = (ins SDFG_StreamType:$str);
    let results = (outs AnyType:$res);
}

//===----------------------------------------------------------------------===//
// StreamPushOp
//===----------------------------------------------------------------------===//

/// Defining part of the SDFG stream memlet.
def SDFG_StreamPushOp : SDFG_Op<"stream_push", [
    ParentOneOf<["StateNode", "MapNode"]>,
    TypesMatchWith<"value type matches element type of 'stream'", "str", "val",
                  "$_self.cast<StreamType>().getDimensions().getElementType()">
]> {
    let summary = "Stream push operation";
    let description = [{
        Allows a state to push a value into a stream.
        For example:

        ```mlir
        sdfg.state @state_0 {
            ...
            %1 = arith.constant 1 : i32
            sdfg.stream_push %1, %A : i32 -> !sdfg.stream<i32>
            ...
        }
        ```
    }];

    let arguments = (ins AnyType:$val, SDFG_StreamType:$str);
}

//===----------------------------------------------------------------------===//
// StreamLengthOp
//===----------------------------------------------------------------------===//

/// Defining the SDFG stream length operation.
def SDFG_StreamLengthOp : SDFG_Op<"stream_length"> {
    let summary = "Stream length operation";
    let description = [{
        Returns the length of the stream.
        For example:

        ```mlir
        sdfg.state @state_0 {
            ...
            %l = sdfg.stream_length %A : !sdfg.stream<i32> -> i32
            ...
        }
        ```
    }];

    let arguments = (ins SDFG_StreamType:$str);
    let results = (outs I32);
}

#endif // SDFG_StreamOps
