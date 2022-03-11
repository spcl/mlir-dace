// RUN: sdfg-opt %s | sdfg-opt | FileCheck %s

// CHECK: module
// CHECK: sdfg.sdfg
sdfg.sdfg{entry=@state_0} {
    // CHECK-NEXT: [[NAMEA:%[a-zA-Z0-9_]*]] = sdfg.alloc()
    // CHECK-SAME: !sdfg.stream<i32>
    %A = sdfg.alloc() : !sdfg.stream<i32>
    // CHECK: sdfg.state
    // CHECK-SAME: @state_0
    sdfg.state @state_0 {
        // CHECK-NEXT: {{%[a-zA-Z0-9_]*}} = sdfg.stream_pop [[NAMEA]]
        // CHECK-SAME: !sdfg.stream<i32> -> i32
        %a_1 = sdfg.stream_pop %A : !sdfg.stream<i32> -> i32
    }
}
