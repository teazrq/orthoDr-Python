#!/usr/bin/env Rscript
# Rscript tools/generate_r_sir_scaled.R PINNED_SNAPSHOT [OUTPUT]
# The pinned R wrapper, fed pooled-SD coordinates, is the source of this fixture.
args <- commandArgs(trailingOnly = TRUE)
if (!length(args)) stop("Supply the pinned R source snapshot directory")
snapshot <- normalizePath(args[[1L]], mustWork = TRUE)
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1L])
package_dir <- dirname(dirname(normalizePath(script, mustWork = TRUE)))
fixtures <- file.path(package_dir, "tests", "fixtures")
output <- if (length(args) >= 2L) args[[2L]] else file.path(fixtures, "r_sir_scaled.json")
needed <- c("pkgload", "jsonlite", "Rcpp", "RcppArmadillo", "dr", "pracma")
missing <- needed[!vapply(needed, requireNamespace, logical(1L), quietly = TRUE)]
if (length(missing)) stop("Install in the shared R library: ", paste(missing, collapse = ", "))
pin <- jsonlite::read_json(file.path(package_dir, "tools", "r_source_manifest.json"))
source(file.path(package_dir, "tools", "check_r_source.R"))
before <- verify_pinned_source(snapshot, pin)
pkgload::load_all(snapshot, quiet = TRUE, export_all = FALSE, compile = TRUE,
                  helpers = FALSE, attach_testthat = FALSE)
base <- jsonlite::read_json(file.path(fixtures, "r_reference.json"), simplifyVector = TRUE)

record <- function(item) {
  call <- item$input
  centered <- scale(call$x, scale = FALSE)
  s <- sqrt(sum(centered^2) / ((nrow(centered) - 1) * ncol(centered)))
  call$x <- centered / s
  names(call)[names(call) == "B_initial"] <- "B.initial"
  fit <- do.call(orthoDr_reg, c(call, list(keep.data = TRUE)))
  fit$fn <- fit$fn * s^4
  fit$x <- item$input$x
  list(expected = unclass(fit[c("B", "fn", "itr", "converge", "method", "bw", "keep.data")]),
       prediction = unclass(predict(fit, item$testx)))
}

regression <- lapply(base$regression[c("sir_d1", "sir_d2")], record)
stationary <- lapply(base$stationary_fits["reg_sir"], record)
after <- verify_pinned_source(snapshot, pin)
stopifnot(identical(before, after))
provenance <- list(source_sha = pin$source_sha,
  source_files_md5 = as.list(before), source_hash_convention = pin$source_hash_convention,
  source_fixture = "r_reference.json",
  oracle = "Pinned R wrapper fitted to centered X divided by its pooled sample SD; fn and predictors restored to supplied units",
  R_version = R.version.string, platform = R.version$platform,
  packages = setNames(lapply(c("orthoDr", needed), function(p) as.character(packageVersion(p))),
                      c("orthoDr", needed)))
jsonlite::write_json(list(provenance = provenance, regression = regression,
                         stationary_fits = stationary), output, auto_unbox = TRUE,
                     matrix = "rowmajor", digits = 17, pretty = TRUE)
cat("Wrote ", normalizePath(output), "\n", sep = "")
