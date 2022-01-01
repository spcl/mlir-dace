sdir.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdir.alloc() : !sdir.array<i32>
    sdir.state @state_0 {
        %a = sdir.get_access %A : !sdir.array<i32> -> !sdir.memlet<i32>
        %a_1 = sdir.load %a[] : !sdir.memlet<i32> -> i32
    }
}
