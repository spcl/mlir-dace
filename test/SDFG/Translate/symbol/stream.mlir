// RUN: sdfg-translate --mlir-to-sdfg %s | python3 %S/../import_translation_test.py

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  sdfg.state @state_0{
    sdfg.alloc_symbol("N")
    %a = sdfg.alloc() : !sdfg.stream<sym("N")xi32>
  }
}
