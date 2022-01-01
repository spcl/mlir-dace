sdir.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdir.alloc() : !sdir.array<23x54xi32>
    sdir.state @state_0 {
        %a = sdir.get_access %A : !sdir.array<23x54xi32> -> !sdir.memlet<23x54xi32>
    }
} 
