// RUN: sdir-translate --mlir-to-sdfg %s | python %S/../import_translation_test.py

sdir.sdfg{entry=@state_0} @sdfg_0 {
    sdir.state @state_0{
        sdir.sdfg{entry=@state_1} @sdfg_1 {
            sdir.state @state_1{
            }
        }
    }
} 
