// RUN: sdfg-opt %s | sdfg-opt | FileCheck %s

// CHECK: module
// CHECK: sdfg.sdfg
sdfg.sdfg{entry=@state_0} {
    // CHECK-NEXT: [[NAMEA:%[a-zA-Z0-9_]*]] = sdfg.alloc
    // CHECK-SAME: !sdfg.stream<i32>
    %A = sdfg.alloc() : !sdfg.stream<i32>
    // CHECK: sdfg.state
    // CHECK-SAME: @state_0
    sdfg.state @state_0 {
        // CHECK: [[NAMEB:%[a-zA-Z0-9_]*]] = sdfg.tasklet
        %0 = sdfg.tasklet() -> (i32) {
            %0 = arith.constant 0 : i32
            sdfg.return %0 : i32
        }
        // CHECK: sdfg.stream_push [[NAMEB]], [[NAMEA]]
        // CHECK-SAME: i32 -> !sdfg.stream<i32>
        sdfg.stream_push %0, %A : i32 -> !sdfg.stream<i32>
    }
}
