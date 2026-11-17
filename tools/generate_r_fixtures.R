#!/usr/bin/env Rscript
# Generate an independent oracle from the committed R implementation.
# Uses the existing user-wide R library, never installs or overwrites orthoDr.
# Usage: Rscript tools/generate_r_fixtures.R [snapshot-directory] [output.json]
args <- commandArgs(trailingOnly = TRUE)
snapshot <- if (length(args) >= 1L) args[[1L]] else
  "../audit/orthodr-pseudo-implementation-2026-11-17/r-source"
output <- if (length(args) >= 2L) args[[2L]] else "tests/fixtures/r_reference.json"
snapshot <- normalizePath(snapshot, mustWork = TRUE)
needed <- c("pkgload", "jsonlite", "Rcpp", "RcppArmadillo", "dr", "pracma")
missing <- needed[!vapply(needed, requireNamespace, logical(1L), quietly = TRUE)]
if (length(missing)) stop("Install in the user-wide R library: ", paste(missing, collapse = ", "))
# Validate the independent source pin before loading R code or replacing fixtures.
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1L])
package_dir <- dirname(dirname(normalizePath(script, mustWork = TRUE)))
reference <- jsonlite::read_json(file.path(package_dir, "tools", "r_source_manifest.json"))
source(file.path(package_dir, "tools", "check_r_source.R"))
invisible(verify_pinned_source(snapshot, reference))
# Rebuild from the verified source instead of trusting a cached DLL or objects.
pkgload::load_all(snapshot, quiet = TRUE, export_all = TRUE, compile = TRUE,
                  helpers = FALSE, attach_testthat = FALSE)
set.seed(20261116)
control <- list(rho = 1e-4, eta = 0.2, gamma = 0.85, tau = 1e-3,
                epsilon = 1e-6, btol = 1e-8, ftol = 1e-8, gtol = 1e-8)

# Keep matrix shapes, including p = 1 and ndr = 1, in the JSON representation.
clean <- function(z) {
  if (is.matrix(z)) return(z)
  if (is.list(z)) return(lapply(z, clean))
  if (!is.null(names(z))) return(lapply(as.list(z), clean))
  unname(z)
}
state <- function(fit) clean(fit[setdiff(names(fit), c("x", "y", "a", "r", "censor"))])
record <- function(input, fit, testx) list(
  input = input, expected = state(fit), testx = testx,
  prediction = clean(unclass(predict(fit, testx))))

n <- 36L; p <- 3L
x <- matrix(rnorm(n * p), n, p)
x[, 2L] <- 0.35 * x[, 1L] + 1.3 * x[, 2L] + 0.5
x[, 3L] <- 0.7 * x[, 3L] - 0.3
y <- 0.8 * x[, 1L] + 0.45 * x[, 2L]^2 + sin(x[, 3L]) + rnorm(n, sd = 0.25)
testx <- rbind(x[c(2L, 17L), , drop = FALSE], c(-0.7, 0.2, 1.1), c(1.4, -0.3, -0.8))
basis <- matrix(c(1, 0.35, -0.2, 0.1, 0.6, 0.8), p, 2L)
reg <- list(); initializers <- list(dr = list())
for (d in 1:2) for (method in c("sir", "save", "phd", "local", "seff")) {
  B <- basis[, seq_len(d), drop = FALSE]
  bw <- silverman(d, n)
  input <- list(x = x, y = y, ndr = d, method = method, B_initial = B,
                bw = bw, control = control, maxitr = 7L, ncore = 1L)
  fit <- orthoDr_reg(x, y, method = method, ndr = d, B.initial = B,
                    bw = bw, control = control, maxitr = 7L, ncore = 1L,
                    keep.data = TRUE)
  z <- record(input, fit, testx)
  Y <- as.matrix(scale(y) / n^(-1 / (d + 5)) / sqrt(2))
  z$initial_objective <- reg_init(method, gramSchmidt(B)$Q, x, Y, bw, 1L)
  z$scaled_y <- Y
  reg[[paste0(method, "_d", d)]] <- z
}
for (method in c("sir", "save", "phd")) {
  obj <- dr::dr(y ~ x, method = method)
  initializers$dr[[method]] <- list(x = x, y = y,
    expected = list(vectors = obj$evectors, values = obj$evalues, M = obj$M))
}
initializers$reg_selected <- list()
for (d in 1:2) for (method in c("sir", "save", "phd", "local", "seff")) {
  Y <- as.matrix(scale(y) / n^(-1 / (d + 5)) / sqrt(2))
  initializers$reg_selected[[paste0(method, "_d", d)]] <- list(
    x = x, y = Y, ndr = d, bw = silverman(d, n), method = method,
    expected = initB(x, Y, d, silverman(d, n), method, 1L))
}

# Complete ties include events and censored subjects at the same time.
sx <- x[1:24, , drop = FALSE]
sy <- c(3, 1, 2, 2, 5, 4, 1, 6, 4, 3, 5, 6, 2, 7, 3, 5, 1, 7, 4, 6, 2, 5, 3, 7)
sc <- c(1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1)
surv <- list()
surv_case <- function(name, xx, yy, cc, method, d = 1L) {
  B <- basis[seq_len(ncol(xx)), seq_len(d), drop = FALSE]
  input <- list(x = xx, y = yy, censor = cc, method = method, ndr = d,
                B_initial = B, bw = 0.65, control = control, maxitr = 7L, ncore = 1L)
  fit <- orthoDr_surv(xx, yy, cc, method = method, ndr = d, B.initial = B,
                     bw = 0.65, control = control, maxitr = 7L, ncore = 1L,
                     keep.data = TRUE)
  surv[[name]] <<- record(input, fit, testx[, seq_len(ncol(xx)), drop = FALSE])
  # Also retain the exact preprocessed inputs to distinguish floating-point
  # preprocessing differences from differences inside the native solver.
  ranks <- rank(yy, ties.method = "average") / (nrow(xx) + 1)
  YY <- if (sd(ranks) == 0) rep(0, length(ranks)) else
    drop(scale(ranks)) / sqrt(2) / silverman(1, length(ranks))
  ord <- order(YY); XX <- scale(xx)[ord, , drop = FALSE]
  YY <- YY[ord]; CC <- cc[ord]
  events <- which(CC == 1); starts <- match(YY[events], YY)
  ky <- exp(-as.matrix(dist(YY, method = "euclidean")))
  ky[outer(YY, YY, "<")] <- 0
  phi <- matrix(0, ncol(xx), length(events))
  for (j in seq_along(events)) phi[, j] <-
    apply(XX, 2L, weighted.mean, w = CC * ky[, events[j]]) -
    colMeans(XX[starts[j]:nrow(xx), , drop = FALSE])
  prepared <- list(B = gramSchmidt(B)$Q, X = XX, Y = YY, Phit = phi,
                   Fail_Ind = events, Risk_Start = starts, bw = 0.65)
  surv[[name]]$prepared <<- prepared
  if (name %in% c("dm_ties", "dn_one_event")) {
    call <- prepared[setdiff(names(prepared), "Y")]
    call <- c(call, control, list(verbose = 0L, ncore = 1L))
    solver <- get(paste0("surv_", method, "_solver"))
    surv[[name]]$native_trajectory <<- lapply(1:7, function(iter) {
      clean(do.call(solver, c(call, list(maxitr = iter))))
    })
  }
}
for (method in c("forward", "dn", "dm")) {
  surv_case(paste0(method, "_ties"), sx, sy, sc, method,
            if (method == "forward") 1L else 2L)
  surv_case(paste0(method, "_one_event"), sx, sy, as.integer(seq_along(sy) == 11L), method)
  surv_case(paste0(method, "_p1_equal_time"), sx[, 1L, drop = FALSE],
            rep(2, length(sy)), sc, method)
}
initializers$CP_SIR <- list()
for (case in c("ties", "one_event", "p1")) {
  xx <- if (case == "p1") sx[, 1L, drop = FALSE] else sx
  cc <- if (case == "one_event") as.integer(seq_along(sy) == 11L) else sc
  initializers$CP_SIR[[case]] <- list(x = xx, y = sy, censor = cc, bw = 0.65,
                                       expected = CP_SIR(xx, sy, cc, bw = 0.65))
}

# Dose units and nonzero dose origin exercise scaling and the intercept model.
px <- x[1:30, , drop = FALSE]
a <- runif(nrow(px), 0.4, 2.7)
r <- 3 + 0.5 * px[, 1L] - 1.2 * (a - (1.4 + 0.3 * px[, 1L] - 0.2 * px[, 2L]))^2
r <- r - min(r) + 0.3
pdose <- list()
for (case in c("direct", "direct_intercept", "pseudo_direct", "pseudo_direct_intercept", "direct_initialization")) {
  method <- if (startsWith(case, "pseudo_direct")) "pseudo_direct" else "direct"
  d <- if (method == "pseudo_direct") 2L else 1L
  B <- basis[, seq_len(d), drop = FALSE]
  input <- list(x = px, a = a, r = r, ndr = d, B_initial = B, bw = 0.7,
                lambda = c(0.05, 0.1, 0.3), K = 6L, method = method,
                control = control, maxitr = if (case == "direct_initialization") 0L else 7L,
                ncore = 1L)
  call <- input; names(call)[names(call) == "B_initial"] <- "B.initial"
  if (case == "pseudo_direct_intercept") {
    input$intercept <- TRUE
    call$intercept <- TRUE
  }
  if (method == "direct") {
    input$lambda_reward <- 0.12
    input$intercept <- identical(case, "direct_intercept")
    if (input$intercept) input$intercept_bounds <- c(0.8, 2.3)
    call$lambda.reward <- input$lambda_reward; call$intercept <- input$intercept
    if (input$intercept) call$intercept.bounds <- input$intercept_bounds
  }
  fit <- do.call(orthoDr_pdose, c(call, list(keep.data = TRUE)))
  pdose[[case]] <- record(input, fit, testx)
}
initializers$pSAVE <- list(
  ordinary = list(x = px, a = a, r = r, ndr = 2L, nslices0 = 2L,
                  expected = pSAVE(px, a, r, ndr = 2L)),
  p1 = list(x = px[, 1L, drop = FALSE], a = a, r = r, ndr = 1L, nslices0 = 2L,
            expected = pSAVE(px[, 1L, drop = FALSE], a, r, ndr = 1L)))
initializers$hMave <- list()
for (case in c("default", "explicit_B0")) {
  hx <- sx[1:20, 1:2, drop = FALSE]
  hy <- exp(0.2 * hx[, 1L] - 0.1 * hx[, 2L]) + seq_len(20) / 15
  hc <- sc[1:20]
  B0 <- if (case == "explicit_B0") matrix(c(1, 0.4), 2L, 1L) else NULL
  initializers$hMave[[case]] <- list(x = hx, y = hy, censor = hc, m0 = 1L,
                                    B0 = B0, expected = hMave(hx, hy, hc, 1L, B0))
}

evaluations <- list()
W <- sin(seq_len(nrow(px))) / 12
for (intercept in c(FALSE, TRUE)) {
  alpha <- if (intercept) 1.4 else 0
  bounds <- if (intercept) c(0.8, 2.3) else NULL
  B <- gramSchmidt(basis[, 1L, drop = FALSE])$Q
  input <- list(B = B, X = px, A = a, R = r, bw = 0.7, W = W,
                lambda_reward = 0.12, cayley_step = 0.03, intercept = intercept,
                intercept_bounds = bounds, alpha = alpha)
  evaluations[[if (intercept) "intercept" else "ordinary"]] <- list(
    input = input,
    expected = pdose_direct_evaluate(B, px, a, r, 0.7, W, 0.12,
                                    0.03, intercept, bounds, alpha))
}

defaults <- list()
for (method in c("local", "seff")) {
  d <- if (method == "local") 1L else 2L
  input <- list(x = x, y = y, method = method, ndr = d,
                bw = silverman(d, n), control = control, maxitr = 7L, ncore = 1L)
  fit <- do.call(orthoDr_reg, c(input, list(keep.data = TRUE)))
  defaults[[paste0("reg_", method)]] <- c(list(family = "regression"), record(input, fit, testx))
}
input <- list(x = sx, y = sy, censor = sc, method = "dm", ndr = 2L,
              bw = 0.65, control = control, maxitr = 7L, ncore = 1L)
fit <- do.call(orthoDr_surv, c(input, list(keep.data = TRUE)))
defaults$surv_dm <- c(list(family = "survival"), record(input, fit, testx))
input <- list(x = px, a = a, r = r, ndr = 1L, bw = 0.7,
              lambda = c(0.05, 0.1, 0.3), K = 6L, method = "direct",
              control = control, maxitr = 7L, ncore = 1L,
              lambda_reward = 0.12, intercept = TRUE, intercept_bounds = c(0.8, 2.3))
call <- input; names(call)[names(call) == "lambda_reward"] <- "lambda.reward"
names(call)[names(call) == "intercept_bounds"] <- "intercept.bounds"
fit <- do.call(orthoDr_pdose, c(call, list(keep.data = TRUE)))
defaults$pdose_direct <- c(list(family = "pdose"), record(input, fit, testx))

# Keep finite-difference sensitivity visible. This is a perturbation study of
# the saved starting matrix, not a claim that a particular BLAS is the cause.
sensitivity <- list()
prep <- surv$dm_ties$prepared
call <- c(prep[setdiff(names(prep), "Y")], control,
          list(maxitr = 7L, verbose = 0L, ncore = 1L))
for (delta in c(0, -1e-14, 1e-14, -1e-13, 1e-13)) {
  current <- call
  current$B[1L, 1L] <- current$B[1L, 1L] + delta
  current$B <- gramSchmidt(current$B)$Q
  native <- do.call(surv_dm_solver, current)
  raw <- native
  raw$B <- sweep(raw$B, 1L, apply(sx, 2L, sd), "/")
  raw$B <- sweep(raw$B, 2L, sqrt(colSums(raw$B^2)), "/")
  raw$bw <- 0.65; raw$x <- sx; raw$y <- sy; raw$censor <- sc; raw$keep.data <- TRUE
  class(raw) <- c("orthoDr", "fit", "surv")
  sensitivity[[as.character(delta)]] <- list(delta = delta, initial_B = current$B,
    native = clean(native), raw_B = raw$B,
    prediction = clean(unclass(predict(raw, testx))))
}

stationary <- list()
stationary_control <- control
stationary_control$btol <- stationary_control$ftol <- stationary_control$gtol <- 1e-6
input <- list(x = x, y = y, method = "sir", ndr = 1L, B_initial = basis[, 1L, drop = FALSE],
              bw = silverman(1L, n), control = stationary_control, maxitr = 250L, ncore = 1L)
call <- input; names(call)[names(call) == "B_initial"] <- "B.initial"
fit <- do.call(orthoDr_reg, c(call, list(keep.data = TRUE)))
stationary$reg_sir <- c(list(family = "regression"), record(input, fit, testx))
input <- list(x = sx, y = sy, censor = sc, method = "forward", ndr = 1L,
              B_initial = basis[, 1L, drop = FALSE], bw = 0.65,
              control = stationary_control, maxitr = 250L, ncore = 1L)
call <- input; names(call)[names(call) == "B_initial"] <- "B.initial"
fit <- do.call(orthoDr_surv, c(call, list(keep.data = TRUE)))
stationary$surv_forward <- c(list(family = "survival"), record(input, fit, testx))

optim <- list()
for (p in c(4L, 5L)) for (analytic in c(TRUE, FALSE)) for (maximize in c(TRUE, FALSE)) {
  d <- if (p == 4L) 2L else 1L
  A <- crossprod(matrix(seq_len(p * p) %% 7 - 3, p, p)) + diag(seq_len(p))
  B <- matrix(sin(seq_len(p * d)), p, d)
  fn <- function(B) if (maximize) 0.5 * sum(B * (A %*% B)) else -0.5 * sum(B * (A %*% B))
  grad <- if (analytic) function(B) if (maximize) A %*% B else -A %*% B else NULL
  fit <- ortho_optim(B, fn, grad, maximize = maximize, control = control, maxitr = 10L)
  key <- paste0("p", p, "_", if (analytic) "analytic" else "numeric", "_", if (maximize) "max" else "min")
  optim[[key]] <- list(input = list(B = B, A = A, analytic = analytic, maximize = maximize,
                                    control = control, maxitr = 10L), expected = state(fit))
}

invisible(verify_pinned_source(snapshot, reference))
source_files <- pinned_source_files(snapshot)
source_md5 <- as.list(unname(pinned_source_md5(snapshot, source_files)))
names(source_md5) <- source_files
fixture <- list(
  provenance = list(
    source_sha = reference$source_sha,
    source_hash_convention = reference$source_hash_convention,
    source_directory = snapshot, seed = 20261116L,
    source_files_md5 = source_md5,
    R_version = R.version.string, platform = R.version$platform,
    packages = setNames(lapply(c("orthoDr", needed), function(p) as.character(packageVersion(p))),
                        c("orthoDr", needed)),
    session_info = capture.output(sessionInfo()),
    purpose = "Short deterministic numerical parity checks, not statistical performance studies"),
  regression = reg, survival = surv, pdose = pdose,
  initializers = initializers, direct_evaluations = evaluations,
  default_fits = defaults, optim = optim, stationary_fits = stationary,
  dm_rounding_sensitivity = sensitivity)
dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
jsonlite::write_json(clean(fixture), output, auto_unbox = TRUE, matrix = "rowmajor",
                     digits = 17, pretty = TRUE, null = "null", na = "null")
cat("Wrote ", normalizePath(output), "\n", sep = "")
