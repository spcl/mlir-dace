// RUN: sdir-translate --mlir-to-sdfg %s | python %S/../import_translation_test.py

sdir.sdfg{entry=@state_0} @sdfg_0 {
    sdir.state @state_0{
        sdir.tasklet @add(%a: i32, %b: i32) -> i32{
            %c = arith.addi %a, %b : i32
            sdir.return %c : i32
        }
    }
}
