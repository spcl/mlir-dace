// RUN: sdfg-translate --mlir-to-sdfg %s | python3 %S/../import_translation_test.py

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  %a = sdfg.alloc() : !sdfg.stream<67x45xi32>
  sdfg.state @state_0 {}
}
