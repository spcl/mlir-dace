// RUN: sdfg-opt --lower-sdfg %s

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  sdfg.state {nosync=false, instrument="No_Instrumentation"} @state_0 {}
}
