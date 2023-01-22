// RUN: sdfg-translate --mlir-to-sdfg %s | python3 %S/../import_translation_test.py
// TODO: Find a way if to fail the test if this doesn't get translated correctly (probably correctness test by executing)

sdfg.sdfg () -> () {
  sdfg.state @state_0{
    %n:2 = sdfg.tasklet() -> (i32, i32) {
      %1 = arith.constant 1 : i32
      %5 = arith.addi %1, %1 : i32
      sdfg.return %1, %5 : i32, i32
    }
  }
}
