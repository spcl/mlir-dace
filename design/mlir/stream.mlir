%A = sdir.alloc_stream() : !sdir.stream<i32>
%42 = constant 42 : i32
sdir.stream_push %42, %A : i32
%42_p = sdir.stream_pop %A : i32