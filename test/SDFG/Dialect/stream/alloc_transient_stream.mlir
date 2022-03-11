// RUN: sdfg-opt %s | sdfg-opt | FileCheck %s

// CHECK: module
// CHECK: sdfg.sdfg
sdfg.sdfg{entry=@state_0} {
    // CHECK: sdfg.state
    // CHECK-SAME: @state_0
    sdfg.state @state_0 {
        // CHECK-NEXT: {{%[a-zA-Z0-9_]*}} = sdfg.alloc {transient}()
        // CHECK-SAME: !sdfg.stream<i32>
        %A = sdfg.alloc{transient}() : !sdfg.stream<i32>
    }
}

