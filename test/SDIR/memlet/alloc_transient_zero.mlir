// XFAIL: *
// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK: sdir.sdfg
sdir.sdfg{entry=@state_0} @sdfg_0 {
    // CHECK-NEXT: {{%[a-zA-Z0-9_]*}} = sdir.alloc_transient()
    // CHECK-SAME: !sdir.array<0xi32>
    %a = sdir.alloc_transient() : !sdir.array<0xi32>

    sdir.state @state_0{

    }
}
