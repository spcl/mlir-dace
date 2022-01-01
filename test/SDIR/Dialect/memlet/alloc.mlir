// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK: sdir.sdfg
sdir.sdfg{entry=@state_0} @sdfg_0 {
    // CHECK-NEXT: {{%[a-zA-Z0-9_]*}} = sdir.alloc()
    // CHECK-SAME: !sdir.array<i32>
    %a = sdir.alloc() : !sdir.array<i32>

    sdir.state @state_0{

    }
}
