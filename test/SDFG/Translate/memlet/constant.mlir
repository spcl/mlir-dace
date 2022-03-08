// RUN: sdfg-translate --mlir-to-sdfg %s | python3 %S/../import_translation_test.py
module  {
  sdfg.sdfg {entry = @state_1} @kernel_2mm(%arg1: !sdfg.memlet<sym("s_0")x900xi32>) {
    sdfg.state @state_1 {
      sdfg.tasklet @task_2() -> i32 {
        %c0 = arith.constant 0 : i32
        sdfg.return %c0 : i32
      }

      %0 = sdfg.call @task_2() : () -> i32
      sdfg.store %0, %arg1[0, 0] : i32 -> !sdfg.memlet<sym("s_0")x900xi32>
    }
  }
}
