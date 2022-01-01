sdir.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdir.alloc() : !sdir.array<12x12xi32>
    sdir.state @state_0 {
        sdir.alloc_symbol("N")
        sdir.tasklet @one() -> i32{
            %1 = arith.constant 1 : i32
            sdir.return %1 : i32
        }
        %1 = sdir.call @one() : () -> i32>
        %a = sdir.get_access %A : !sdir.array<12x12xi32> -> !sdir.memlet<12x12xi32>
        sdir.store %1, %a[0, sym("N")] : i32 -> !sdir.memlet<12x12xi32>
    }
}
