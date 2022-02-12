// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK: sdir.sdfg
sdir.sdfg{entry=@state_0} @sdfg_0 {
    // CHECK-NEXT: [[NAMEA:%[a-zA-Z0-9_]*]] = sdir.alloc()
    // CHECK-SAME: !sdir.stream<i32>
    %A = sdir.alloc() : !sdir.stream<i32>
    // CHECK: sdir.state
    // CHECK-SAME: @state_0
    sdir.state @state_0 {
        // CHECK: [[NAMEB:%[a-zA-Z0-9_]*]] = sdir.tasklet @zero
        %0 = sdir.tasklet @zero() -> i32{
            %0 = arith.constant 0 : i32
            sdir.return %0 : i32
        }
        // CHECK: sdir.stream_push [[NAMEB]], [[NAMEA]]
        // CHECK-SAME: i32 -> !sdir.stream<i32>
        sdir.stream_push %0, %A : i32 -> !sdir.stream<i32>
    }
}
