// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK: sdir.sdfg
sdir.sdfg{entry=@state_0} @sdfg_0 {
    // CHECK-NEXT: [[NAMEA:%[a-zA-Z0-9_]*]] = sdir.alloc()
    // CHECK-SAME: !sdir.array<5x3xi32>
    %A = sdir.alloc() : !sdir.array<5x3xi32>
    // CHECK: sdir.state
    // CHECK-SAME: @state_0
    sdir.state @state_0 {
        // CHECK-NEXT: sdir.alloc_symbol("N")
        sdir.alloc_symbol("N")
        // CHECK-NEXT: {{%[a-zA-Z0-9_]*}} = sdir.load [[NAMEA]][sym("N"), 0]
        // CHECK-SAME: !sdir.array<5x3xi32> -> i32
        %a_1 = sdir.load %A[sym("N"), 0] : !sdir.array<5x3xi32> -> i32
    }
}
