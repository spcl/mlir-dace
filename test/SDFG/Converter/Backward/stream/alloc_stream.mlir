// XFAIL: *
// RUN: sdfg-opt --lower-sdfg %s

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  %a = sdfg.alloc() : !sdfg.stream<i32>
  sdfg.state @state_0 {}
}
