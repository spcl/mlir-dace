// RUN: sdir-translate --mlir-to-sdfg %s | python %S/../import_translation_test.py

sdir.sdfg{entry=@state_0} @sdfg_0 {
    sdir.state @state_0{
    }
    sdir.state @state_1{
    }
    sdir.edge{assign=["i = 1"], condition=""} @state_0 -> @state_1
}
