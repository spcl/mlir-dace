// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK: sdir.sdfg
sdir.sdfg{entry=@state_0} @sdfg_0 {
    // CHECK-NEXT: [[NAMEA:%[a-zA-Z0-9_]*]] = sdir.alloc()
    // CHECK-SAME: !sdir.array<12x12xi32>
    %A = sdir.alloc() : !sdir.array<12x12xi32>
    // CHECK: sdir.state
    // CHECK-SAME: @state_0
    sdir.state @state_0 {
        // CHECK-NEXT: sdir.alloc_symbol("N")
        sdir.alloc_symbol("N")
        // CHECK: sdir.tasklet @one
        sdir.tasklet @one() -> i32{
            %1 = arith.constant 1 : i32
            sdir.return %1 : i32
        }
        // CHECK: [[NAME1:%[a-zA-Z0-9_]*]] = sdir.call @one()
        %1 = sdir.call @one() : () -> i32
        // CHECK-NEXT: sdir.store [[NAME1]], [[NAMEA]][0, sym("N")]
        // CHECK-SAME: i32 -> !sdir.array<12x12xi32>
        sdir.store %1, %A[0, sym("N")] : i32 -> !sdir.array<12x12xi32>
    }
}
