// RUN: sdfg-opt %s | sdfg-opt | FileCheck %s

// CHECK: module
// CHECK: sdfg.sdfg
sdfg.sdfg{entry=@state_0} @sdfg_0 {
    // CHECK: sdfg.state
    sdfg.state @state_0{
        // CHECK-NEXT: sdfg.alloc_symbol("N")
        sdfg.alloc_symbol("N")
        // CHECK-NEXT: {{%[a-zA-Z0-9_]*}} = sdfg.alloc()
        // CHECK-SAME: !sdfg.array<sym("N")xi32>
        %a = sdfg.alloc() : !sdfg.array<sym("N")xi32>
    }
}
