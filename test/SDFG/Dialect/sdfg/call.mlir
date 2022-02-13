// RUN: sdfg-opt %s | sdfg-opt | FileCheck %s

// CHECK: module
// CHECK: sdfg.sdfg 
// CHECK-SAME: {{@[a-zA-Z0-9_]*}}
sdfg.sdfg{entry=@state_0} @sdfg_0 {
    // CHECK: sdfg.state {{@[a-zA-Z0-9_]*}}
    sdfg.state @state_0 {
        // CHECK: sdfg.sdfg
        // CHECK-SAME: entry = {{@[a-zA-Z0-9_]*}}
        // CHECK-SAME: [[SDFG1:@[a-zA-Z0-9_]*]]
        // CHECK-SAME: ({{%[a-zA-Z0-9_]*}}: !sdfg.array<i32>, {{%[a-zA-Z0-9_]*}}: !sdfg.array<i32>) -> !sdfg.array<i32>
        sdfg.sdfg{entry=@state_1} @sdfg_1(%a: !sdfg.array<i32>, %b: !sdfg.array<i32>) -> !sdfg.array<i32> {
            // CHECK: sdfg.state {{@[a-zA-Z0-9_]*}}
            sdfg.state @state_1 {
            }
        }
        // CHECK: [[ARRAYN:%[a-zA-Z0-9_]*]] = sdfg.alloc()
        // CHECK-SAME: !sdfg.array<i32>
        %N = sdfg.alloc() : !sdfg.array<i32>
        // CHECK-NEXT: {{%[a-zA-Z0-9_]*}} = sdfg.call [[SDFG1]]
        // CHECK-SAME: ([[ARRAYN]], [[ARRAYN]])
        // CHECK-SAME: (!sdfg.array<i32>, !sdfg.array<i32>) -> !sdfg.array<i32>
        %c = sdfg.call @sdfg_1(%N, %N) : (!sdfg.array<i32>, !sdfg.array<i32>) -> !sdfg.array<i32>
    }
} 
