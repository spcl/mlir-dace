// XFAIL: *
// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK: sdir.sdfg 
// CHECK-SAME: [[SDFG0:@[a-zA-Z0-9_]*]]
// CHECK-SAME: ({{%[a-zA-Z0-9_]*}}: !sdir.memlet<i32>, {{%[a-zA-Z0-9_]*}}: !sdir.memlet<i32>) -> !sdir.memlet<i32>
sdir.sdfg{entry=@state_0} @sdfg_0(%a: !sdir.memlet<i32>, %b: !sdir.memlet<i32>) -> !sdir.memlet<i32> {
    // CHECK: sdir.state {{@[a-zA-Z0-9_]*}}
    sdir.state @state_0 {
        // CHECK: sdir.sdfg
        sdir.sdfg{entry=@state_1} @sdfg_1 {
            // CHECK: sdir.state {{@[a-zA-Z0-9_]*}}
            sdir.state @state_1 {
                // CHECK: [[ARRAYN:%[a-zA-Z0-9_]*]] = sdir.alloc()
                // CHECK-SAME: !sdir.array<i32>
                %N = sdir.alloc() : !sdir.array<i32>
                // CHECK-NEXT: [[MEMLETn:%[a-zA-Z0-9_]*]] = sdir.get_access [[ARRAYN]] 
                // CHECK-SAME: !sdir.array<i32> -> !sdir.memlet<i32>
                %n = sdir.get_access %N : !sdir.array<i32> -> !sdir.memlet<i32>
                // CHECK-NEXT: {{%[a-zA-Z0-9_]*}} = sdir.call [[SDFG0]]
                // CHECK-SAME: ([[MEMLETn]], [[MEMLETn]])
                // CHECK-SAME: (!sdir.memlet<i32>, !sdir.memlet<i32>) -> !sdir.memlet<i32>
                %c = sdir.call @sdfg_0(%n, %n) : (!sdir.memlet<i32>, !sdir.memlet<i32>) -> !sdir.memlet<i32>
            }
        }

    }
} 
