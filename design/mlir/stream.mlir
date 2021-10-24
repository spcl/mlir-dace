%A = sdir.alloc_stream() : !sdir.stream_array<i32>
%a = sdir.get_access %A : !sdir.stream_array<i32> -> !sdir.stream<i32>

%42 = arith.constant 42 : i32
sdir.stream_push %42, %a : i32 -> !sdir.stream<i32>
%42_p = sdir.stream_pop %a : !sdir.stream<i32> -> i32
