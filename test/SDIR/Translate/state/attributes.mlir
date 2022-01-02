// RUN: sdir-translate --mlir-to-sdfg %s | python %S/../import_translation_test.py

sdir.sdfg{entry=@state_0} @sdfg_0 {
    sdir.state {
        nosync=false,
        instrument="No_Instrumentation"
    } @state_0 {
    }
}
