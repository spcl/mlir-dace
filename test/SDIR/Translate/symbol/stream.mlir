sdir.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdir.alloc_stream() : !sdir.stream_array<12xi32>
    sdir.state @state_0 {
        sdir.alloc_symbol("N")
        %a = sdir.get_access %A : !sdir.stream_array<12xi32> -> !sdir.stream<sym("N")xi32>
    }
}
