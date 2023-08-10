// RUN: sdfg-opt --lower-sdfg %s

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  sdfg.state @state_0{
    sdfg.nested_sdfg () -> (%r: !sdfg.array<i32>) {
      sdfg.state @state_1{
        %0 = sdfg.load %r[] : !sdfg.array<i32> -> i32
        sdfg.store %0, %r[] : i32 -> !sdfg.array<i32>
      }
    }
  }
} 
