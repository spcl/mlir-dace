// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK-NEXT: [[NAMEA:%[a-zA-Z0-9_]*]] = sdir.alloc_stream
%A = sdir.alloc_stream() : !sdir.stream<i32>

// CHECK: sdir.state
// CHECK-SAME: @state_0
sdir.state @state_0 {
    // CHECK-NEXT: [[NAMEB:%[a-zA-Z0-9_]*]] = constant 
    // CHECK-SAME: i32
    %1 = constant 0 : i32
    // CHECK-NEXT: sdir.stream_push [[NAMEB]], [[NAMEA]]
    // CHECK-SAME: !sdir.stream<i32>
    sdir.stream_push %1, %A : !sdir.stream<i32>
}
