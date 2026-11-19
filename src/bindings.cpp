// Python entry points, GPL-3.0-or-later; upstream core GPL-2.0-or-later.
#include "orthodr_bridge.h"
#include "cpp/orthoDr_reg.h"
#include "cpp/orthoDr_surv.h"
#include "cpp/orthoDr_pdose.h"

using py::arg;

PYBIND11_MODULE(_core, module) {
  module.doc() = "Standalone orthoDr numerical kernels; no R runtime dependency";
  module.attr("source_base_revision") = "a9588d8dd73c39188d1c46f371763addb10bee9a";
  module.attr("source_revision") = "a9588d8dd73c39188d1c46f371763addb10bee9a"
    "+native.b522e6dc7fa725890d824dd0b5c04e03372ca9d8c865e595692ba51eaeb8ebe1";
  module.attr("armadillo_version") = arma::arma_version::as_string();

  module.def("local_f", &local_f, arg("B"), arg("X"), arg("Y"), arg("bw"), arg("ncore"));
  module.def("sir_init", &sir_init, arg("B"), arg("X"), arg("Y"), arg("bw"), arg("ncore"), arg("pooled_ratio") = 0.0);
  module.def("save_init", &save_init, arg("B"), arg("X"), arg("Y"), arg("bw"), arg("ncore"),
             arg("identity_weight") = 1.0, arg("covariance_weight") = 1.0, arg("moment_scale") = 1.0);
  module.def("phd_init", &phd_init, arg("B"), arg("X"), arg("Y"), arg("bw"), arg("ncore"));
  module.def("seff_init", &seff_init, arg("B"), arg("X"), arg("Y"), arg("bw"), arg("ncore"));

#define REG_ARGS arg("B"), arg("X"), arg("Y"), arg("bw"), arg("rho"), arg("eta"), arg("gamma"), arg("tau"), arg("epsilon"), arg("btol"), arg("ftol"), arg("gtol"), arg("maxitr"), arg("verbose"), arg("ncore")
  module.def("local_solver", &local_solver, REG_ARGS);
  module.def("sir_solver", &sir_solver, REG_ARGS, arg("pooled_ratio") = 0.0);
  module.def("save_solver", &save_solver, REG_ARGS, arg("identity_weight") = 1.0,
             arg("covariance_weight") = 1.0, arg("moment_scale") = 1.0);
  module.def("phd_solver", &phd_solver, REG_ARGS);
  module.def("seff_solver", &seff_solver, REG_ARGS);
#undef REG_ARGS

#define OPT_ARGS arg("bw"), arg("rho"), arg("eta"), arg("gamma"), arg("tau"), arg("epsilon"), arg("btol"), arg("ftol"), arg("gtol"), arg("maxitr"), arg("verbose"), arg("ncore")
  module.def("surv_forward_solver", &surv_forward_solver, arg("B"), arg("X"), arg("Fail_Ind"), arg("Risk_Start"), OPT_ARGS);
  module.def("surv_dn_solver", &surv_dn_solver, arg("B"), arg("X"), arg("Phit"), arg("Fail_Ind"), arg("Risk_Start"), OPT_ARGS);
  module.def("surv_dm_solver", &surv_dm_solver, arg("B"), arg("X"), arg("Phit"), arg("Fail_Ind"), arg("Risk_Start"), OPT_ARGS);
  // Private audit helpers evaluate the committed objective at supplied states.
  module.def("surv_forward_f", &surv_forward_f, arg("B"), arg("X"), arg("Fail_Ind"), arg("Risk_Start"), arg("bw"), arg("ncore"));
  module.def("surv_dn_f", &surv_dn_f, arg("B"), arg("X"), arg("Phit"), arg("Fail_Ind"), arg("Risk_Start"), arg("bw"), arg("ncore"));
  module.def("surv_dm_f", &surv_dm_f, arg("B"), arg("X"), arg("Phit"), arg("Fail_Ind"), arg("Risk_Start"), arg("bw"), arg("ncore"));
  module.def("pdose_semi_solver", &pdose_semi_solver, arg("B"), arg("X"), arg("R"), arg("A"), arg("a_seq"), arg("lambda"), OPT_ARGS, arg("intercept") = false);
  module.def("pdose_semi_audit_solver", &pdose_semi_audit_solver,
             arg("B"), arg("X"), arg("R"), arg("A"), arg("a_seq"), arg("lambda"), OPT_ARGS,
             arg("intercept") = false, arg("trace") = true, arg("trial_budget") = 30,
             arg("audit_fail_gradient_iteration") = -1, arg("audit_fail_trial_iteration") = -1);
  module.def("pdose_semi_evaluate", &pdose_semi_evaluate,
             arg("B"), arg("X"), arg("R"), arg("A"), arg("bw"), arg("a_seq"), arg("lambda"),
             arg("intercept") = false, arg("cayley_step") = 1e-4,
             arg("dose_factor") = 0.7071067811865476);

#define DIRECT_ARGS arg("B"), arg("X"), arg("A"), arg("a_dist"), arg("a_seq"), arg("R"), arg("lambda"), OPT_ARGS
  module.def("pdose_direct_solver", &pdose_direct_solver, DIRECT_ARGS,
             arg("lambda_reward") = 0.1, arg("intercept") = false,
             arg("intercept_bounds") = py::none());
  module.def("pdose_direct_audit_solver", &pdose_direct_audit_solver, DIRECT_ARGS,
             arg("lambda_reward") = 0.1, arg("trace") = true, arg("B_trial_budget") = 30,
             arg("W_initial") = py::none(), arg("intercept") = false,
             arg("intercept_bounds") = py::none(), arg("alpha_initial") = py::none(),
             arg("audit_fail_final_sweep") = -1);
#undef DIRECT_ARGS
#undef OPT_ARGS
  module.def("pdose_direct_evaluate", &pdose_direct_evaluate,
             arg("B"), arg("X"), arg("A"), arg("R"), arg("bw"), arg("W"),
             arg("lambda_reward"), arg("cayley_step") = 0.0, arg("intercept") = false,
             arg("intercept_bounds") = py::none(), arg("alpha") = 0.0);
  module.def("pdose_direct_policy_direction", &pdose_direct_policy_direction,
             arg("gradient"), arg("hessian"), arg("alpha"), arg("intercept_bounds"));
  module.def("pdose_direct_spectral_scalar", &pdose_direct_spectral_scalar,
             arg("SS"), arg("SZ"), arg("ZZ"), arg("m"), arg("fallback"),
             arg("finite_vectors") = true, arg("evaluation_valid") = true);
  module.def("dosepred", &dosepred, arg("B"), arg("X"), arg("X_test"), arg("bw"), arg("W"));
  module.def("KernelDist_cross", &KernelDist_cross, arg("TestX"), arg("X"));
}
