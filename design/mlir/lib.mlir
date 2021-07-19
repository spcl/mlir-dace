sdir.state @state_0{    
    %a = sdir.get_access %A
    %b = sdir.get_access %B
    %c = sdir.get_access %C

    %c_t = sdir.libcall "dace.libraries
            .blas.nodes.Gemm" (%a, %b)
    sdir.store %c_t, %c
}

