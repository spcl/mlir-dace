// RUN: not sdfg-opt --convert-to-sdfg %s 2>&1 | FileCheck %s
// CHECK: failed to legalize operation 'builtin.module'

module{
  sdfg.sdfg {entry=@state_0} () -> () {
    sdfg.state @state_0{

    }
  }

  sdfg.sdfg {entry=@state_1} () -> () {
    sdfg.state @state_1{

    }
  }
}
