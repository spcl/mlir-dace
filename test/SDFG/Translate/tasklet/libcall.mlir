// RUN: sdfg-translate --mlir-to-sdfg %s | python3 %S/../import_translation_test.py

sdfg.sdfg () -> (%r: !sdfg.array<2x2xi32>) {
  %A = sdfg.alloc() : !sdfg.array<2x2xi32>
  %B = sdfg.alloc() : !sdfg.array<2x2xi32>

  sdfg.state @state_0{
    %c = sdfg.libcall{inputs=["_a", "_b"], outputs=["_c"]} "dace.libraries.blas.nodes.MatMul" (%A, %B) : (!sdfg.array<2x2xi32>, !sdfg.array<2x2xi32>) -> !sdfg.array<2x2xi32>
    sdfg.copy %c -> %r : !sdfg.array<2x2xi32>
  }
}
