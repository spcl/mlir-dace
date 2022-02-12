// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK: sdir.sdfg
sdir.sdfg{entry=@state_0} @sdfg_0 {
    // CHECK-NEXT: [[NAMEA:%[a-zA-Z0-9_]*]] = sdir.alloc()
    // CHECK-SAME: !sdir.array<12x45xi32>
    %A = sdir.alloc() : !sdir.array<12x45xi32>
    // CHECK: sdir.state
    // CHECK-SAME: @state_0
    sdir.state @state_0 {
        // CHECK: [[NAME0:%[a-zA-Z0-9_]*]] = sdir.tasklet @zero
        %0 = sdir.tasklet @zero() -> index{
            %0 = arith.constant 0 : index
            sdir.return %0 : index
        }
        // CHECK: {{%[a-zA-Z0-9_]*}} = sdir.load [[NAMEA]]
        // CHECK-SAME: [[NAME0]], [[NAME0]]
        // CHECK-SAME: !sdir.array<12x45xi32> -> i32
        %a_1 = sdir.load %A[%0, %0] : !sdir.array<12x45xi32> -> i32
    }
} 
