// RUN: sdfg-translate --mlir-to-sdfg %s | not python3 %S/../import_translation_test.py 2>&1 | FileCheck %s
// CHECK: Isolated node

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  %A = sdfg.alloc() : !sdfg.stream<2x6xi32>

  sdfg.state @state_0 {
    sdfg.consume{num_pes=5} (%A : !sdfg.stream<2x6xi32>) -> (pe: %p, elem: %e) {
    }
  }
}
