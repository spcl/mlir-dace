// RUN: sdfg-translate --mlir-to-sdfg %s | python3 %S/../import_translation_test.py

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  %A = sdfg.alloc() : !sdfg.array<i32>

  sdfg.state @state_0 {
    %a_1 = sdfg.load %A[] : !sdfg.array<i32> -> i32
    sdfg.store %a_1, %A[] : i32 -> !sdfg.array<i32>
  }
}
