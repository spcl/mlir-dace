// RUN: sdfg-opt --convert-to-sdfg %s | sdfg-opt
module{
  sdfg.sdfg {entry=@state_0} () -> () {
    sdfg.state @state_0{

    }
  }
}
