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
        // CHECK-NEXT: [[NAME1:%[a-zA-Z0-9_]*]]
        %1 = arith.constant 1 : i32
        // CHECK-NEXT: [[NAMEa:%[a-zA-Z0-9_]*]] = sdir.get_access [[NAMEA]] 
        // CHECK-SAME: !sdir.array<12x12xi32> -> !sdir.memlet<12x12xi32>
        %a = sdir.get_access %A : !sdir.array<12x12xi32> -> !sdir.memlet<12x12xi32>
        // CHECK-NEXT: sdir.store [[NAME1]], [[NAMEa]][0, sym("N")]
        // CHECK-SAME: i32 -> !sdir.memlet<12x12xi32>
        sdir.store %1, %a[0, sym("N")] : i32 -> !sdir.memlet<12x12xi32>
    }
}
