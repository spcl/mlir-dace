// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK: sdir.sdfg
sdir.sdfg{entry=@state_0} @sdfg_0 {
    // CHECK-NEXT: [[NAMEA:%[a-zA-Z0-9_]*]] = sdir.alloc_stream()
    // CHECK-SAME: !sdir.stream_array<i32>
    %A = sdir.alloc_stream() : !sdir.stream_array<i32>
    // CHECK: sdir.state
    // CHECK-SAME: @state_0
    sdir.state @state_0 {
        // CHECK: sdir.tasklet @zero
        sdir.tasklet @zero() -> i32{
            %0 = arith.constant 0 : i32
            sdir.return %0 : i32
        }
        // CHECK: [[NAMEB:%[a-zA-Z0-9_]*]] = sdir.call @zero()
        %0 = sdir.call @zero() : () -> i32
        // CHECK-NEXT: [[NAMEC:%[a-zA-Z0-9_]*]] = sdir.get_access [[NAMEA]] 
        // CHECK-SAME: !sdir.stream_array<i32> -> !sdir.stream<i32>
        %a = sdir.get_access %A : !sdir.stream_array<i32> -> !sdir.stream<i32>
        // CHECK-NEXT: sdir.stream_push [[NAMEB]], [[NAMEC]]
        // CHECK-SAME: i32 -> !sdir.stream<i32>
        sdir.stream_push %0, %a : i32 -> !sdir.stream<i32>
    }
}
