sdir.sdfg{entry=@state_0} @sdfg_0 {
    sdir.state @state_0 {
        sdir.sdfg{entry=@state_1} @sdfg_1(%a: !sdir.memlet<i32>, %b: !sdir.memlet<i32>) -> !sdir.memlet<i32> {
            sdir.state @state_1 {
            }
        }

        %N = sdir.alloc() : !sdir.array<i32>
        %n = sdir.get_access %N : !sdir.array<i32> -> !sdir.memlet<i32>
        %c = sdir.call @sdfg_1(%n, %n) : (!sdir.memlet<i32>, !sdir.memlet<i32>) -> !sdir.memlet<i32>
    }
} 
