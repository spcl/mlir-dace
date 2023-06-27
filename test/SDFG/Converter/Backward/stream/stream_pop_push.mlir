// RUN: sdfg-opt --lower-sdfg %s

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  %A = sdfg.alloc() : !sdfg.stream<i32>

  sdfg.state @state_0 {        
    %1 = sdfg.tasklet() -> (i32) {
      %1 = arith.constant 1 : i32
      sdfg.return %1 : i32
    }

    sdfg.stream_push %1, %A : i32 -> !sdfg.stream<i32>
    %a_1 = sdfg.stream_pop %A : !sdfg.stream<i32> -> i32

    sdfg.store %a_1, %r[] : i32 -> !sdfg.array<i32>
  }
}
