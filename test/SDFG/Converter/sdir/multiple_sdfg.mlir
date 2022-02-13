// XFAIL: *
// RUN: sdfg-opt --convert-to-sdfg %s
module{
  sdfg.sdfg {entry=@state_0} @sdfg_0 {
    sdfg.state @state_0{

    }
  }

  sdfg.sdfg {entry=@state_1} @sdfg_1 {
    sdfg.state @state_1{

    }
  }
}
