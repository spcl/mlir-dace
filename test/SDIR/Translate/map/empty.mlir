sdir.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdir.alloc() : !sdir.array<2x6xi32>
    sdir.state @state_0 {
        %a = sdir.get_access %A : !sdir.array<2x6xi32> -> !sdir.memlet<2x6xi32>
        sdir.map (%i, %j) = (0, 0) to (2, 2) step (1, 1) {
        }
    }
}