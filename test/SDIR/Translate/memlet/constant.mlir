// RUN: sdir-translate --mlir-to-sdfg %s | python %S/../import_translation_test.py
module  {
  sdir.sdfg {entry = @state_1} @kernel_2mm(%arg1: !sdir.memlet<sym("s_0")x900xi32>) {
    sdir.state @state_1 {
      sdir.tasklet @task_2() -> i32 {
        %c0 = arith.constant 0 : i32
        sdir.return %c0 : i32
      }

      %0 = sdir.call @task_2() : () -> i32
      sdir.store %0, %arg1[0, 0] : i32 -> !sdir.memlet<sym("s_0")x900xi32>
    }
  }
}
