// RUN: sdfg-opt --lower-sdfg %s

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  sdfg.state @state_0 {
    %a = sdfg.alloc() : !sdfg.array<i32>
  }
}
