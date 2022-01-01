sdir.sdfg{entry=@state_0} @sdfg_0 {
    sdir.state @state_0{
        sdir.tasklet @get_zero() -> i32{
            %c = arith.constant 0 : i32
            sdir.return %c : i32
        }
    }
}
