// RUN: sdfg-translate --mlir-to-sdfg %s | python3 %S/../import_translation_test.py

sdfg.sdfg{entry = @state_1} (%arg0: index) -> (%arg1: !sdfg.array<sym("s_0")x900xi32>){
  sdfg.state @state_1 {
    %n = sdfg.load %arg1[%arg0, %arg0] : !sdfg.array<sym("s_0")x900xi32> -> i32

    %0 = sdfg.tasklet(%n: i32) -> (i32) {
      %c0 = arith.addi %n, %n : i32
      sdfg.return %c0 : i32
    }

    sdfg.store %0, %arg1[%arg0, %arg0] : i32 -> !sdfg.array<sym("s_0")x900xi32>
  }
}

