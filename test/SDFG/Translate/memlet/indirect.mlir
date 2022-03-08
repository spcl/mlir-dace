// RUN: sdfg-translate --mlir-to-sdfg %s | python3 %S/../import_translation_test.py
module  {
  sdfg.sdfg {entry = @state_1} @kernel_2mm(%arg0: index, %arg1: !sdfg.memlet<sym("s_0")x900xi32>) {
    sdfg.state @state_1 {
      sdfg.tasklet @task_2(%arg2: i32) -> i32 {
        %c0 = arith.addi %arg2, %arg2 : i32
        sdfg.return %c0 : i32
      }

      %n = sdfg.load %arg1[%arg0, %arg0] : !sdfg.memlet<sym("s_0")x900xi32> -> i32
      %0 = sdfg.call @task_2(%n) : (i32) -> i32
      sdfg.store %0, %arg1[%arg0, %arg0] : i32 -> !sdfg.memlet<sym("s_0")x900xi32>
    }
  }
}
