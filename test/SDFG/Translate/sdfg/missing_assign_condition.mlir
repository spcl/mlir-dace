// RUN: sdfg-translate --mlir-to-sdfg %s | python3 %S/../import_translation_test.py

sdfg.sdfg{entry=@state_0} {
    sdfg.state @state_0 {}
    sdfg.state @state_1 {}
    sdfg.edge @state_0 -> @state_1
}
