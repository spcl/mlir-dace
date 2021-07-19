sdir.state @state_0{ 
    %a = sdir.get_access %A 
            : !sdir.memlet<i32>
    %c = sdir.get_access %C 
            : !sdir.memlet<i32>

    sdir.copy %a -> %c
}