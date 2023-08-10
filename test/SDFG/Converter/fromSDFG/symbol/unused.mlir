// RUN: sdfg-opt --lower-sdfg %s

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  sdfg.state @state_0{
    sdfg.alloc_symbol("N")
    %res = sdfg.sym("3*N+2") : i64
  }
}
