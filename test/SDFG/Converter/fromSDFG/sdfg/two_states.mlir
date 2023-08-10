// RUN: sdfg-opt --lower-sdfg %s

sdfg.sdfg{entry=@state_0} () -> (%r: !sdfg.array<i32>) {
  sdfg.state @state_1 {}
  sdfg.state @state_0 {}

  sdfg.edge{assign=["i: 1"]} @state_0 -> @state_1
}
