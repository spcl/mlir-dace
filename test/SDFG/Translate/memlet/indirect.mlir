// RUN: sdir-translate --mlir-to-sdfg %s | python %S/../import_translation_test.py
module  {
  sdir.sdfg {entry = @state_1} @kernel_2mm(%arg0: index, %arg1: !sdir.memlet<sym("s_0")x900xi32>) {
    sdir.state @state_1 {
      sdir.tasklet @task_2(%arg2: i32) -> i32 {
        %c0 = arith.addi %arg2, %arg2 : i32
        sdir.return %c0 : i32
      }

      %n = sdir.load %arg1[%arg0, %arg0] : !sdir.memlet<sym("s_0")x900xi32> -> i32
      %0 = sdir.call @task_2(%n) : (i32) -> i32
      sdir.store %0, %arg1[%arg0, %arg0] : i32 -> !sdir.memlet<sym("s_0")x900xi32>
    }
  }
}
