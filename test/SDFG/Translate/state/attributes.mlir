// RUN: sdfg-translate --mlir-to-sdfg %s | python3 %S/../import_translation_test.py

sdfg.sdfg{entry=@state_0} @sdfg_0 {
    sdfg.state {
        nosync=false,
        instrument="No_Instrumentation"
    } @state_0 {
    }
}
