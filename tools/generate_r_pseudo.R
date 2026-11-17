#!/usr/bin/env Rscript
# Independent fixed-state native R oracle for the revised pseudo-direct solver.
# Rscript tools/generate_r_pseudo.R SNAPSHOT OUTPUT
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("Supply isolated pinned source snapshot and output JSON paths")
snapshot <- normalizePath(args[[1L]], mustWork = TRUE)
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1L])
tool_dir <- dirname(normalizePath(script, mustWork = TRUE))
needed <- c("pkgload", "jsonlite", "Rcpp", "RcppArmadillo", "dr", "pracma")
missing <- needed[!vapply(needed, requireNamespace, logical(1L), quietly = TRUE)]
if (length(missing)) stop("Install in the shared user-wide R library: ", paste(missing, collapse = ", "))
pin <- jsonlite::read_json(file.path(tool_dir, "r_source_manifest.json"))
source(file.path(tool_dir, "check_r_source.R"))
before <- verify_pinned_source(snapshot, pin)
pkgload::load_all(snapshot, quiet = TRUE, export_all = TRUE, compile = TRUE,
                  helpers = FALSE, attach_testthat = FALSE)
clean <- function(z) {
  if (is.matrix(z)) return(z)
  if (is.list(z)) return(lapply(z, clean))
  if (!is.null(names(z))) return(lapply(as.list(z), clean))
  unname(z)
}
set.seed(781)
fixture <- function(p, d) {
  n <- 13L
  x <- matrix(rnorm(n * p), n, p)
  x <- sweep(x, 2, c(1, 4, .5, 2, 3, .8)[seq_len(p)], "*")
  x[, 2] <- x[, 2] + .6 * x[, 1]
  B <- pseudo_orthonormalize(matrix(sin(seq_len(p * d)), p, d))
  a <- seq(.1, 3.8, length.out = n)
  r <- 2 * exp(-(a - 2 - .5 * tanh(x[, 1]))^2) + .1 * x[, 3]
  list(B = B, X = x, R = r, A = a, bw = .73,
       a_seq = seq(min(a), max(a), length.out = 9L), lambda = c(.03, .2, 1),
       intercept = FALSE, cayley_step = 1e-4, dose_factor = 1 / sqrt(2))
}
cases <- list()
put <- function(name, input) {
  warnings <- character()
  expected <- withCallingHandlers(do.call(pdose_semi_evaluate, input),
    warning = function(w) {warnings <<- c(warnings, conditionMessage(w)); invokeRestart("muffleWarning")})
  cases[[name]] <<- list(input = input, expected = clean(expected), warnings = as.list(warnings))
}
for (p in c(3L, 6L)) for (d in 1:2) for (intercept in c(FALSE, TRUE)) {
  input <- fixture(p, d); input$intercept <- intercept
  put(paste0("p", p, "_d", d, if (intercept) "_intercept" else ""), input)
}
base <- cases$p6_d2$input
for (bw in c(.05, .01, .001)) {input <- base; input$bw <- bw; put(paste0("bw_", bw), input)}
input <- base; input$intercept <- TRUE; input$a_seq <- c(.1, .1); input$lambda <- c(.2, .03, .2)
put("constant_target", input)
input$bw <- 1e15; input$lambda <- c(1e-30, .1); put("mixed_ridge", input)
input <- base; input$dose_factor <- 1; put("aligned_dose_factor", input)
input <- base; input$lambda <- .2; put("scalar_ridge", input)
input <- base; input$intercept <- TRUE; input$lambda <- 1e200
put("large_ridge", input)
input <- base; input$intercept <- TRUE; input$R <- 1e8 * (input$R + 3)
put("reward_affine", input)
input <- base; input$intercept <- TRUE; input$A <- 3 * input$A + 1e4
input$a_seq <- 3 * input$a_seq + 1e4; put("dose_affine", input)
controls <- list(rho = 1e-4, eta = .2, gamma = .85, tau = 1,
  epsilon = 1e-6, btol = 1e-10, ftol = 1e-10, gtol = 1e-6)
audit_input <- base[c("B", "X", "R", "A", "a_seq", "lambda", "bw", "intercept")]
trajectories <- list()
for (steps in c(0L, 1L, 8L)) {
  call <- c(audit_input, controls, list(maxitr = steps, verbose = 0L, ncore = 1L))
  trajectories[[as.character(steps)]] <- clean(do.call(pdose_semi_audit_solver, call))
}
after <- verify_pinned_source(snapshot, pin)
if (!identical(unname(before), unname(after))) stop("Pinned source changed during generation")
provenance <- list(source_sha = pin$source_sha, source_files_md5 = as.list(before),
  source_hash_convention = pin$source_hash_convention,
  seed = 781L, R_version = R.version.string, platform = R.version$platform,
  packages = setNames(lapply(c("orthoDr", needed), function(p) as.character(packageVersion(p))), c("orthoDr", needed)),
  session_info = capture.output(sessionInfo()),
  purpose = "Fixed-state analytic pseudo-direct gradient, stabilized scores, ridge GCV, Cayley and accepted-state trace parity")
jsonlite::write_json(list(provenance = provenance, cases = cases,
  audit_input = audit_input, controls = controls, trajectories = trajectories), args[[2L]],
  auto_unbox = TRUE, digits = 17, matrix = "rowmajor", null = "null", na = "null", pretty = TRUE)
cat("Wrote", length(cases), "pseudo-direct fixed states and", length(trajectories), "audit trajectories\n")
