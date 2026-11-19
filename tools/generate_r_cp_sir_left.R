#!/usr/bin/env Rscript
# Rscript tools/generate_r_cp_sir_left.R SNAPSHOT [OUTPUT]
# SNAPSHOT is the pinned archive with the revised R/CP_SIR.R overlaid.
args <- commandArgs(trailingOnly = TRUE)
if (!length(args)) stop("Supply an isolated R source snapshot with the CP-SIR switch")
snapshot <- normalizePath(args[[1L]], mustWork = TRUE)
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1L])
package_dir <- dirname(dirname(normalizePath(script, mustWork = TRUE)))
output <- if (length(args) >= 2L) args[[2L]] else
  file.path(package_dir, "tests", "fixtures", "r_cp_sir_left.json")
needed <- c("pkgload", "jsonlite", "Rcpp", "RcppArmadillo", "dr", "pracma")
missing <- needed[!vapply(needed, requireNamespace, logical(1L), quietly = TRUE)]
if (length(missing)) stop("Install in the shared R library: ", paste(missing, collapse = ", "))
pin <- jsonlite::read_json(file.path(package_dir, "tools", "r_source_manifest.json"))
source(file.path(package_dir, "tools", "check_r_source.R"))
overlay <- "R/CP_SIR.R"
pin$source_files_md5[[overlay]] <- "68b0b968faeb2cc7feb644eacca80d23"
before <- verify_pinned_source(snapshot, pin)
pkgload::load_all(snapshot, quiet = TRUE, export_all = TRUE, compile = TRUE,
                  helpers = FALSE, attach_testthat = FALSE)

base <- jsonlite::read_json(file.path(package_dir, "tests", "fixtures", "r_reference.json"),
                            simplifyVector = TRUE)
initializers <- lapply(base$initializers$CP_SIR, function(item) {
  do.call(CP_SIR, item[c("x", "y", "censor", "bw")])
})
item <- base$default_fits$surv_dm
fit_surv <- function(input) do.call(orthoDr_surv, c(input, list(keep.data = TRUE)))
record <- function(fit) list(
  expected = fit[setdiff(names(fit), c("x", "y", "censor"))],
  prediction = unclass(predict(fit, item$testx)))
fit <- fit_surv(item$input)
surv_dm <- record(fit)
first_step <- record(fit_surv(modifyList(item$input, list(maxitr = 1L))))

# Measure how small R-only changes of the same automatic start propagate
# through the inherited finite-difference solver.
n <- nrow(item$input$x)
y <- drop(scale(rank(item$input$y) / (n + 1))) / sqrt(2) / silverman(1, n)
basis <- CP_SIR(scale(item$input$x), y, item$input$censor)$vectors[, 1:2]
projector <- function(B) tcrossprod(qr.Q(qr(B)))
compare_start <- function(start) {
  other <- fit_surv(c(item$input, list(B.initial = start)))
  c(projector = max(abs(projector(other$B) - projector(fit$B))),
    prediction = max(abs(predict(other, item$testx)$surv - surv_dm$prediction$surv)))
}
perturbations <- lapply(seq_along(basis), function(i) {
  change <- matrix(0, nrow(basis), ncol(basis))
  change[i] <- 1e-14
  pmax(compare_start(basis + change), compare_start(basis - change))
})
sign_flips <- lapply(list(c(-1, 1), c(1, -1), c(-1, -1)), function(signs) {
  compare_start(sweep(basis, 2, signs, "*"))
})
sensitivity <- list(absolute_perturbation = 1e-14,
  perturbation_maxima = as.list(Reduce(pmax, perturbations)),
  sign_flip_maxima = as.list(Reduce(pmax, sign_flips)))
after <- verify_pinned_source(snapshot, pin)
if (!identical(before, after)) stop("Snapshot changed during generation")
provenance <- list(base_source_sha = pin$source_sha,
  source_overlay = overlay, source_files_md5 = as.list(before),
  source_hash_convention = pin$source_hash_convention,
  source_fixture = "r_reference.json",
  R_version = R.version.string, platform = R.version$platform,
  packages = setNames(lapply(c("orthoDr", needed), function(p) as.character(packageVersion(p))),
                      c("orthoDr", needed)),
  session_info = capture.output(sessionInfo()))
jsonlite::write_json(list(provenance = provenance, initializers = initializers,
                         default_fits = list(surv_dm = surv_dm),
                         martingale_first_step = first_step,
                         martingale_sensitivity = sensitivity), output, auto_unbox = TRUE,
                     matrix = "rowmajor", digits = 17, pretty = TRUE,
                     null = "null", na = "null")
cat("Wrote ", normalizePath(output), "\n", sep = "")
