// RUN: sdir-translate --mlir-to-sdfg %s | not python %S/../import_translation_test.py %s 2>&1 | FileCheck %s
// CHECK: Isolated node
sdir.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdir.alloc() : !sdir.array<i32>
    %B = sdir.alloc() : !sdir.array<i32>
    sdir.state @state_0{
        %a = sdir.get_access %A : !sdir.array<i32> -> !sdir.memlet<i32>
        %b = sdir.get_access %B : !sdir.array<i32> -> !sdir.memlet<i32>
        %c = sdir.libcall "dace.libraries.blas.nodes.Gemm" (%a, %b) : (!sdir.memlet<i32>, !sdir.memlet<i32>) -> f32
    }
}
