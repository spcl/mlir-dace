// RUN: sdfg-translate --mlir-to-sdfg %s | python3 %S/../import_translation_test.py

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  sdfg.state @state_0 {
    %N = sdfg.alloc() : !sdfg.array<i32>
    %M = sdfg.alloc() : !sdfg.array<i32>

    sdfg.nested_sdfg{entry=@state_1} (%N: !sdfg.array<i32>) -> (%M: !sdfg.array<i32>) {
      sdfg.state @state_1 {}
    }
  }
} 
