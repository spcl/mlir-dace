// RUN: sdfg-translate --mlir-to-sdfg %s | python3 %S/../import_translation_test.py

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  %A = sdfg.alloc() : !sdfg.array<2x12xi32>

  sdfg.state @state_0 {
    %b = sdfg.memlet_cast %A : !sdfg.array<2x12xi32> -> !sdfg.array<2x12xi32>
  }
} 
