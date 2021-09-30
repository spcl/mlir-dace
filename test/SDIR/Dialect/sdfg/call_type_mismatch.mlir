// XFAIL: *
// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK: sdir.sdfg 
// CHECK-SAME: {{@[a-zA-Z0-9_]*}}
sdir.sdfg{entry=@state_0} @sdfg_0 {
    // CHECK: sdir.state {{@[a-zA-Z0-9_]*}}
    sdir.state @state_0 {
        // CHECK: sdir.sdfg
        // CHECK-SAME: entry = {{@[a-zA-Z0-9_]*}}
        // CHECK-SAME: [[SDFG1:@[a-zA-Z0-9_]*]]
        // CHECK-SAME: ({{%[a-zA-Z0-9_]*}}: !sdir.memlet<i64>, {{%[a-zA-Z0-9_]*}}: !sdir.memlet<i32>) -> !sdir.memlet<i32>
        sdir.sdfg{entry=@state_1} @sdfg_1(%a: !sdir.memlet<i64>, %b: !sdir.memlet<i32>) -> !sdir.memlet<i32> {
            // CHECK: sdir.state {{@[a-zA-Z0-9_]*}}
            sdir.state @state_1 {
            }
        }
        // CHECK: [[ARRAYN:%[a-zA-Z0-9_]*]] = sdir.alloc()
        // CHECK-SAME: !sdir.array<i32>
        %N = sdir.alloc() : !sdir.array<i32>
        // CHECK-NEXT: [[MEMLETn:%[a-zA-Z0-9_]*]] = sdir.get_access [[ARRAYN]] 
        // CHECK-SAME: !sdir.array<i32> -> !sdir.memlet<i32>
        %n = sdir.get_access %N : !sdir.array<i32> -> !sdir.memlet<i32>
        // CHECK-NEXT: {{%[a-zA-Z0-9_]*}} = sdir.call [[SDFG1]]
        // CHECK-SAME: ([[MEMLETn]], [[MEMLETn]])
        // CHECK-SAME: (!sdir.memlet<i32>, !sdir.memlet<i32>) -> !sdir.memlet<i32>
        %c = sdir.call @sdfg_1(%n, %n) : (!sdir.memlet<i32>, !sdir.memlet<i32>) -> !sdir.memlet<i32>
    }
} 
