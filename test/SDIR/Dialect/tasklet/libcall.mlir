// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK: sdir.sdfg
sdir.sdfg{entry=@state_0} @sdfg_0 {
    // CHECK-NEXT: [[NAMEA:%[a-zA-Z0-9_]*]] = sdir.alloc()
    // CHECK-SAME: !sdir.array<i32>
    %A = sdir.alloc() : !sdir.array<i32>
    // CHECK-NEXT: [[NAMEB:%[a-zA-Z0-9_]*]] = sdir.alloc()
    // CHECK-SAME: !sdir.array<i32>
    %B = sdir.alloc() : !sdir.array<i32>
    // CHECK: sdir.state @state_0
    sdir.state @state_0{
        // CHECK-NEXT: [[NAMEa:%[a-zA-Z0-9_]*]] = sdir.get_access [[NAMEA]] 
        // CHECK-SAME: !sdir.array<i32> -> !sdir.memlet<i32>
        %a = sdir.get_access %A : !sdir.array<i32> -> !sdir.memlet<i32>
        // CHECK-NEXT: [[NAMEb:%[a-zA-Z0-9_]*]] = sdir.get_access [[NAMEB]] 
        // CHECK-SAME: !sdir.array<i32> -> !sdir.memlet<i32>
        %b = sdir.get_access %B : !sdir.array<i32> -> !sdir.memlet<i32>
        // CHECK-NEXT: {{%[a-zA-Z0-9_]*}} = sdir.libcall "dace.libraries.blas.nodes.Gemm"
        // CHECK-SAME: ([[NAMEa]], [[NAMEb]])
        // CHECK-SAME: (!sdir.memlet<i32>, !sdir.memlet<i32>) -> f32
        %c = sdir.libcall "dace.libraries.blas.nodes.Gemm" (%a, %b) : (!sdir.memlet<i32>, !sdir.memlet<i32>) -> f32
    }
}
