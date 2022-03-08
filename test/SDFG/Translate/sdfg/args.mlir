// RUN: sdfg-translate --mlir-to-sdfg %s | python3 %S/../import_translation_test.py
module  {
  sdfg.sdfg {entry = @state_0} @kernel_2mm(%arg0: !sdfg.memlet<i32>) {
    sdfg.state @state_0 {
      sdfg.tasklet @task_1() -> i32 {
        %c42_i32 = arith.constant 42 : i32
        sdfg.return %c42_i32 : i32
      }
      %0 = sdfg.call @task_1() : () -> i32
      sdfg.store %0, %arg0[] : i32 -> !sdfg.memlet<i32>
    }
  }
}
