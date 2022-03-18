// RUN: sdfg-translate --mlir-to-sdfg %s | python3 %S/../import_translation_test.py

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  sdfg.state @state_0 {}
  sdfg.state @state_1 {}
  sdfg.edge{assign=["i: 1"]} @state_0 -> @state_1
}
