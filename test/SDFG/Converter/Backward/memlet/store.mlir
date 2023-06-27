// RUN: sdfg-opt --lower-sdfg %s

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  %A = sdfg.alloc() : !sdfg.array<i32>
  
  sdfg.state @state_0 {
    %1 = sdfg.tasklet() -> (i32) {
      %1 = arith.constant 1 : i32
      sdfg.return %1 : i32
    }

    sdfg.store %1, %A[] : i32 -> !sdfg.array<i32>
  }
}
