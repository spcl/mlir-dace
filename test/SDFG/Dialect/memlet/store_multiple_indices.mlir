// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK: sdir.sdfg
sdir.sdfg{entry=@state_0} @sdfg_0 {
    // CHECK-NEXT: [[NAMEA:%[a-zA-Z0-9_]*]] = sdir.alloc()
    // CHECK-SAME: !sdir.array<56x45xi32>
    %A = sdir.alloc() : !sdir.array<56x45xi32>
    // CHECK: sdir.state
    // CHECK-SAME: @state_0
    sdir.state @state_0 {
        // CHECK: [[NAME0:%[a-zA-Z0-9_]*]] = sdir.tasklet @zero
        %0 = sdir.tasklet @zero() -> index{
            %0 = arith.constant 0 : index
            sdir.return %0 : index
        }
        // CHECK: [[NAMEC:%[a-zA-Z0-9_]*]] = sdir.tasklet @one
        %1 = sdir.tasklet @one() -> i32{
            %1 = arith.constant 1 : i32
            sdir.return %1 : i32
        }
        // CHECK: sdir.store [[NAMEC]], [[NAMEA]]
        // CHECK-SAME: [[NAME0]], [[NAME0]]
        // CHECK-SAME: i32 -> !sdir.array<56x45xi32>
        sdir.store %1, %A[%0, %0] : i32 -> !sdir.array<56x45xi32>
    }
} 
