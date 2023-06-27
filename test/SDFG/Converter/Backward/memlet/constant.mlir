// RUN: sdfg-opt --lower-sdfg %s

sdfg.sdfg{entry = @state_1} () -> (%arg1: !sdfg.array<sym("s_0")x900xi32>) {
  sdfg.state @state_1 {
    %0 = sdfg.tasklet() -> (i32) {
      %c0 = arith.constant 0 : i32
      sdfg.return %c0 : i32
    }

    sdfg.store %0, %arg1[0, 0] : i32 -> !sdfg.array<sym("s_0")x900xi32>
  }
}

