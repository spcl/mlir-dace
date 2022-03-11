// RUN: sdfg-opt %s | sdfg-opt | FileCheck %s

// CHECK: module
// CHECK: sdfg.sdfg
sdfg.sdfg{entry=@state_0} {
    // CHECK-NEXT: [[NAMEA:%[a-zA-Z0-9_]*]] = sdfg.alloc()
    // CHECK-SAME: !sdfg.array<12x45xi32>
    %A = sdfg.alloc() : !sdfg.array<12x45xi32>
    // CHECK: sdfg.state
    // CHECK-SAME: @state_0
    sdfg.state @state_0 {
        // CHECK: [[NAME0:%[a-zA-Z0-9_]*]] = sdfg.tasklet
        %0 = sdfg.tasklet() -> index{
            %0 = arith.constant 0 : index
            sdfg.return %0 : index
        }
        // CHECK: {{%[a-zA-Z0-9_]*}} = sdfg.load [[NAMEA]]
        // CHECK-SAME: [[NAME0]], [[NAME0]]
        // CHECK-SAME: !sdfg.array<12x45xi32> -> i32
        %a_1 = sdfg.load %A[%0, %0] : !sdfg.array<12x45xi32> -> i32
    }
} 
