// RUN: sdfg-opt --lower-sdfg %s

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  sdfg.state @state_0 {
    %A = sdfg.alloc{transient}() : !sdfg.array<i32>
  }
}

