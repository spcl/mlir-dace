// RUN: sdir-translate --mlir-to-sdfg %s | python %S/../import_translation_test.py

sdir.sdfg{entry=@state_0} @sdfg_0 {
    sdir.state @state_0{
        sdir.alloc_symbol("N")
        %res = sdir.sym("3*N+2") : i64
    }
}
