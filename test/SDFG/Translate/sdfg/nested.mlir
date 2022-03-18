// RUN: sdfg-translate --mlir-to-sdfg %s | python3 %S/../import_translation_test.py

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  sdfg.state @state_0{
    sdfg.nested_sdfg () -> (%r: !sdfg.array<i32>) {
      sdfg.state @state_1{
      }
    }
  }
} 
