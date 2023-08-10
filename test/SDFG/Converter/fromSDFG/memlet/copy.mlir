// RUN: sdfg-opt --lower-sdfg %s

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  %A = sdfg.alloc() : !sdfg.array<i32>
  %B = sdfg.alloc() : !sdfg.array<i32>
  sdfg.state @state_0 {
    sdfg.copy %A -> %B : !sdfg.array<i32>
  }
}
