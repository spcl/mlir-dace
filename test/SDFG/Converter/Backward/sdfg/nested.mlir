// RUN: sdfg-opt --lower-sdfg %s

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  sdfg.state @state_0{
    sdfg.nested_sdfg () -> (%r: !sdfg.array<i32>) {
      sdfg.state @state_1{
      }
    }
  }
} 
