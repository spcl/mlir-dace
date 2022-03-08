// RUN: sdfg-translate --mlir-to-sdfg %s | not python3 %S/../import_translation_test.py 2>&1 | FileCheck %s
// CHECK: Isolated node

sdfg.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdfg.alloc() : !sdfg.array<i32>
    %B = sdfg.alloc() : !sdfg.array<i32>

    sdfg.state @state_0{
        %c = sdfg.libcall "dace.libraries.blas.nodes.Gemm" (%A, %B) : (!sdfg.array<i32>, !sdfg.array<i32>) -> f32
    }
}
