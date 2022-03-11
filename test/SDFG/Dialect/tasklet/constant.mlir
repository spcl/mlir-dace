// RUN: sdfg-opt %s | sdfg-opt | FileCheck %s

// CHECK: module
// CHECK: sdfg.sdfg
sdfg.sdfg{entry=@state_0} {
    // CHECK: sdfg.state @state_0
    sdfg.state @state_0{
        // CHECK: sdfg.tasklet 
        // CHECK-SAME: i32
        %res = sdfg.tasklet() -> (i32) {
            // CHECK-NEXT: [[NAME:%[a-zA-Z0-9_]*]]
            %c = arith.constant 0 : i32
            // CHECK-NEXT: sdfg.return [[NAME]]
            sdfg.return %c : i32
        }
    }
}
