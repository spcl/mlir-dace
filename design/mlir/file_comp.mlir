sdir.state{nosync=false, instrument="No_Instrumentation"} @state_0{
    %a = sdir.get_access %A : !sdir.memlet<i32>
    %c = sdir.get_access %C : !sdir.memlet<i32>

    sdir.copy{dynamic=false, allow_oob=false} %a -> %c
}