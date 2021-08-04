// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK-NEXT: [[NAMEA:%[a-zA-Z0-9_]*]] = sdir.alloc
%A = sdir.alloc() : !sdir.memlet<i32>
// CHECK-NEXT: [[NAMEB:%[a-zA-Z0-9_]*]] = sdir.alloc
%B = sdir.alloc() : !sdir.memlet<i32>

// CHECK: sdir.state
// CHECK-SAME: @state_0
sdir.state @state_0 {
    // CHECK-NEXT: [[NAMEa:%[a-zA-Z0-9_]*]] = sdir.get_access [[NAMEA]] 
    // CHECK-SAME: !sdir.memlet<i32>
    %a = sdir.get_access %A : !sdir.memlet<i32>
    // CHECK-NEXT: [[NAMEb:%[a-zA-Z0-9_]*]] = sdir.get_access [[NAMEB]] 
    // CHECK-SAME: !sdir.memlet<i32>
    %b = sdir.get_access %B : !sdir.memlet<i32>
    // CHECK-NEXT: sdir.copy [[NAMEa]] -> [[NAMEb]]
    // CHECK-SAME: !sdir.memlet<i32>
    sdir.copy %a -> %b : !sdir.memlet<i32>
}
