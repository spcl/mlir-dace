// RUN: sdir-translate --mlir-to-sdfg %s | python %S/../import_translation_test.py
module  {
  sdir.sdfg {entry = @state_0} @kernel_2mm(%arg0: !sdir.memlet<i32>) {
    sdir.state @state_0 {
      sdir.tasklet @task_1() -> i32 {
        %c42_i32 = arith.constant 42 : i32
        sdir.return %c42_i32 : i32
      }
      %0 = sdir.call @task_1() : () -> i32
      sdir.store %0, %arg0[] : i32 -> !sdir.memlet<i32>
    }
  }
}
