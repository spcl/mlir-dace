// XFAIL: *
// RUN: sdfg-opt %s | sdfg-opt | FileCheck %s

sdfg.sdfg () -> (%D : !sdfg.array<sym("NI")xsym("NL")xf64>) {
  sdfg.alloc_symbol("NI")
  sdfg.alloc_symbol("NJ")
  sdfg.alloc_symbol("NK")
  sdfg.alloc_symbol("NL")
  sdfg.alloc_symbol("alpha")
  sdfg.alloc_symbol("beta")

  %tmp = sdfg.alloc{transient}() : !sdfg.array<sym("NI")xsym("NJ")xf64>
  %A = sdfg.alloc{transient}() : !sdfg.array<sym("NI")xsym("NK")xf64>
  %B = sdfg.alloc{transient}() : !sdfg.array<sym("NK")xsym("NJ")xf64>
  %C = sdfg.alloc{transient}() : !sdfg.array<sym("NJ")xsym("NL")xf64>

  sdfg.state @init_array {
    %ni = sdfg.sym("NI") : index
    %nj = sdfg.sym("NJ") : index
    %nk = sdfg.sym("NK") : index
    %nl = sdfg.sym("NL") : index

    sdfg.map (%i, %j) = (0, 0) to (sym("NI"), sym("NK")) step (1, 1) {
      %res = sdfg.tasklet(%i: index, %j: index, %ni: index) -> (f64){
        %i_i64 = arith.index_cast %i : index to i64
        %i_f64 = arith.uitofp %i_i64 : i64 to f64
        %j_i64 = arith.index_cast %j : index to i64
        %j_f64 = arith.uitofp %j_i64 : i64 to f64
        %ni_i64 = arith.index_cast %ni : index to i64
        %ni_f64 = arith.uitofp %ni_i64 : i64 to f64
        %mul = arith.mulf %i_f64, %j_f64 : f64
        %1 = arith.constant 1.0 : f64
        %add = arith.addf %mul, %1 : f64
        %mod = arith.remf %add, %ni_f64 : f64
        %div = arith.divf %mod, %ni_f64 : f64
        sdfg.return %div : f64
      }

      sdfg.store %res, %A[%i, %j] : f64 -> !sdfg.array<sym("NI")xsym("NK")xf64>
    }

    sdfg.map (%i, %j) = (0, 0) to (sym("NK"), sym("NJ")) step (1, 1) {
      %res = sdfg.tasklet(%i: index, %j: index, %nj: index) -> (f64){
        %i_i64 = arith.index_cast %i : index to i64
        %i_f64 = arith.uitofp %i_i64 : i64 to f64
        %j_i64 = arith.index_cast %j : index to i64
        %j_f64 = arith.uitofp %j_i64 : i64 to f64
        %nj_i64 = arith.index_cast %nj : index to i64
        %nj_f64 = arith.uitofp %nj_i64 : i64 to f64
        %1 = arith.constant 1.0 : f64
        %add = arith.addf %j_f64, %1 : f64
        %mul = arith.mulf %i_f64, %add : f64
        %mod = arith.remf %mul, %nj_f64 : f64
        %div = arith.divf %mod, %nj_f64 : f64
        sdfg.return %div : f64
      }

      sdfg.store %res, %B[%i, %j] : f64 -> !sdfg.array<sym("NK")xsym("NJ")xf64>
    }

    sdfg.map (%i, %j) = (0, 0) to (sym("NJ"), sym("NL")) step (1, 1) {
      %res = sdfg.tasklet(%i: index, %j: index, %nl: index) -> (f64){
        %i_i64 = arith.index_cast %i : index to i64
        %i_f64 = arith.uitofp %i_i64 : i64 to f64
        %j_i64 = arith.index_cast %j : index to i64
        %j_f64 = arith.uitofp %j_i64 : i64 to f64
        %nl_i64 = arith.index_cast %nl : index to i64
        %nl_f64 = arith.uitofp %nl_i64 : i64 to f64
        %1 = arith.constant 1.0 : f64
        %3 = arith.constant 3.0 : f64
        %add_3 = arith.addf %j_f64, %3 : f64
        %mul = arith.mulf %i_f64, %add_3 : f64
        %add_1 = arith.addf %mul, %1 : f64
        %mod = arith.remf %add_1, %nl_f64 : f64
        %div = arith.divf %mod, %nl_f64 : f64
        sdfg.return %div : f64
      }

      sdfg.store %res, %C[%i, %j] : f64 -> !sdfg.array<sym("NJ")xsym("NL")xf64>
    }

    sdfg.map (%i, %j) = (0, 0) to (sym("NI"), sym("NL")) step (1, 1) {
      %res = sdfg.tasklet(%i: index, %j: index, %nk: index) -> (f64){
        %i_i64 = arith.index_cast %i : index to i64
        %i_f64 = arith.uitofp %i_i64 : i64 to f64
        %j_i64 = arith.index_cast %j : index to i64
        %j_f64 = arith.uitofp %j_i64 : i64 to f64
        %nk_i64 = arith.index_cast %nk : index to i64
        %nk_f64 = arith.uitofp %nk_i64 : i64 to f64
        %2 = arith.constant 2.0 : f64
        %add = arith.addf %j_f64, %2 : f64
        %mul = arith.mulf %i_f64, %add : f64
        %mod = arith.remf %mul, %nk_f64 : f64
        %div = arith.divf %mod, %nk_f64 : f64
        sdfg.return %div : f64
      }

      sdfg.store %res, %D[%i, %j] : f64 -> !sdfg.array<sym("NI")xsym("NL")xf64>
    }
  }

  sdfg.state @kernel_2mm {
    %alpha = sdfg.sym("alpha") : index
    %beta = sdfg.sym("beta") : index

    sdfg.map (%i, %j) = (0, 0) to (sym("NI"), sym("NJ")) step (1, 1) {
      %0 = sdfg.tasklet() -> (f64){
        %0 = arith.constant 0.0 : f64
        sdfg.return %0 : f64
      }

      sdfg.store %0, %tmp[%i, %j] : f64 -> !sdfg.array<sym("NI")xsym("NJ")xf64>

      sdfg.map (%k) = (0) to (sym("NK")) step (1) {
        %A_v = sdfg.load %A[%i, %k] : !sdfg.array<sym("NI")xsym("NK")xf64> -> f64
        %B_v = sdfg.load %B[%k, %j] : !sdfg.array<sym("NK")xsym("NJ")xf64> -> f64
        %tmp_v = sdfg.load %tmp[%i, %j] : !sdfg.array<sym("NI")xsym("NJ")xf64> -> f64

        %res = sdfg.tasklet(%alpha: index, %A_v: f64, %B_v: f64, %tmp_v: f64) -> (f64){
          %alpha_i64 = arith.index_cast %alpha : index to i64
          %alpha_f64 = arith.uitofp %alpha_i64 : i64 to f64
          %mul_1 = arith.mulf %alpha_f64, %A_v : f64
          %mul_2 = arith.mulf %mul_1, %B_v : f64
          %add = arith.addf %mul_2, %tmp_v : f64
          sdfg.return %add : f64
        }

        sdfg.store %res, %tmp[%i, %j] : f64 -> !sdfg.array<sym("NI")xsym("NJ")xf64>
      }
    }

    sdfg.map (%i, %j) = (0, 0) to (sym("NI"), sym("NL")) step (1, 1) {
      %D_v = sdfg.load %D[%i, %j] : !sdfg.array<sym("NI")xsym("NL")xf64> -> f64

      %res_1 = sdfg.tasklet(%beta: index, %D_v: f64) -> (f64){
        %beta_i64 = arith.index_cast %beta : index to i64
        %beta_f64 = arith.uitofp %beta_i64 : i64 to f64
        %mul = arith.mulf %beta_f64, %D_v : f64
        sdfg.return %mul : f64
      }

      sdfg.store %res_1, %D[%i, %j] : f64 -> !sdfg.array<sym("NI")xsym("NL")xf64>

      sdfg.map (%k) = (0) to (sym("NJ")) step (1) {
        %tmp_v = sdfg.load %tmp[%i, %k] : !sdfg.array<sym("NI")xsym("NJ")xf64> -> f64
        %C_v = sdfg.load %C[%k, %j] : !sdfg.array<sym("NJ")xsym("NL")xf64> -> f64
        // %D_v = sdfg.load %D[%i, %j] : !sdfg.array<sym("NI")xsym("NL")xf64> -> f64

        %res_2 = sdfg.tasklet(%tmp_v: f64, %C_v: f64, %D_v: f64) -> (f64){
          %mul = arith.mulf %tmp_v, %C_v : f64
          %add = arith.addf %mul, %D_v : f64
          sdfg.return %add : f64
        }

        sdfg.store %res_2, %D[%i, %j] : f64 -> !sdfg.array<sym("NI")xsym("NL")xf64>
      }
    }
  }

  sdfg.edge @init_array -> @kernel_2mm
}
