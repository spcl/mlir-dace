// RUN: sdfg-translate --mlir-to-sdfg %s | not python %S/../import_translation_test.py 2>&1 | FileCheck %s
// CHECK: Isolated node
sdfg.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdfg.alloc() : !sdfg.array<i32>
    %B = sdfg.alloc() : !sdfg.array<i32>
    sdfg.state @state_0{
        %a = sdfg.get_access %A : !sdfg.array<i32> -> !sdfg.memlet<i32>
        %b = sdfg.get_access %B : !sdfg.array<i32> -> !sdfg.memlet<i32>
        %c = sdfg.libcall "dace.libraries.blas.nodes.Gemm" (%a, %b) : (!sdfg.memlet<i32>, !sdfg.memlet<i32>) -> f32
    }
}
