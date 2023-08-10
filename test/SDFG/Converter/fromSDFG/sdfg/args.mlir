// RUN: sdfg-opt --lower-sdfg %s

module  {
  sdfg.sdfg () -> (%arg0: !sdfg.array<i32>) {
    sdfg.state @state_0 {
      %0 = sdfg.tasklet() -> (i32) {
        %c42_i32 = arith.constant 42 : i32
        sdfg.return %c42_i32 : i32
      }

      sdfg.store %0, %arg0[] : i32 -> !sdfg.array<i32>
    }
  }
}
