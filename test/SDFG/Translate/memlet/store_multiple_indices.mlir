// RUN: sdfg-translate --mlir-to-sdfg %s | python3 %S/../import_translation_test.py

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  %A = sdfg.alloc() : !sdfg.array<56x45xi32>

  sdfg.state @state_0 {
    %0 = sdfg.tasklet() -> (index) {
      %0 = arith.constant 0 : index
      sdfg.return %0 : index
    }

    %1 = sdfg.tasklet() -> (i32) {
      %1 = arith.constant 1 : i32
      sdfg.return %1 : i32
    }

    sdfg.store %1, %A[%0, %0] : i32 -> !sdfg.array<56x45xi32>
  }
} 
