// RUN: sdfg-opt --lower-sdfg %s

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  sdfg.state @state_0{}
  sdfg.state @state_1{}
  sdfg.edge{assign=["i: 1", "j: 5"]} @state_0 -> @state_1
}
