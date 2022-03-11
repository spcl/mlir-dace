// RUN: sdfg-opt %s | sdfg-opt | FileCheck %s

// CHECK: module
// CHECK: sdfg.sdfg
sdfg.sdfg{entry=@state_0} {
    // CHECK-NEXT: [[NAMEA:%[a-zA-Z0-9_]*]] = sdfg.alloc()
    // CHECK-SAME: !sdfg.array<5x3xi32>
    %A = sdfg.alloc() : !sdfg.array<5x3xi32>
    // CHECK: sdfg.state
    // CHECK-SAME: @state_0
    sdfg.state @state_0 {
        // CHECK-NEXT: sdfg.alloc_symbol("N")
        sdfg.alloc_symbol("N")
        // CHECK-NEXT: {{%[a-zA-Z0-9_]*}} = sdfg.load [[NAMEA]][sym("N"), 0]
        // CHECK-SAME: !sdfg.array<5x3xi32> -> i32
        %a_1 = sdfg.load %A[sym("N"), 0] : !sdfg.array<5x3xi32> -> i32
    }
}
