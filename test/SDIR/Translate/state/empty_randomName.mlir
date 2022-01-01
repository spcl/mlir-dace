// RUN: sdir-translate --mlir-to-sdfg %s | python %S/../import_translation_test.py

sdir.sdfg{entry=@randomName} @sdfg_0 {
    sdir.state @randomName{
    }
}
