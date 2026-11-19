#!/usr/bin/env Rscript
# Rscript tools/generate_r_regression_scaled.R BUILT_R_TREE [OUTPUT]
args <- commandArgs(trailingOnly = TRUE)
if (!length(args)) stop("Supply the corresponding built R source tree")
snapshot <- normalizePath(args[[1L]], mustWork = TRUE)
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1L])
package_dir <- dirname(dirname(normalizePath(script, mustWork = TRUE)))
output <- if (length(args) >= 2L) args[[2L]] else
  file.path(package_dir, "tests", "fixtures", "r_regression_scaled.json")
needed <- c("pkgload", "jsonlite", "Rcpp", "RcppArmadillo", "dr", "pracma")
missing <- needed[!vapply(needed, requireNamespace, logical(1L), quietly = TRUE)]
if (length(missing)) stop("Install in the shared R library: ", paste(missing, collapse = ", "))
source(file.path(package_dir, "tools", "check_r_source.R"))
files <- c("R/orthoDr_reg.R", "R/param_check.R", "R/RcppExports.R", "src/RcppExports.cpp",
           "src/orthoDr_reg.h", "src/reg_projection.h", "src/solver_numerics.h",
           list.files(file.path(snapshot, "src"), pattern = "^reg_.*\\.cpp$",
                      full.names = FALSE))
files <- ifelse(startsWith(files, "reg_"), paste0("src/", files), files)
source_md5 <- pinned_source_md5(snapshot, files)
if (anyNA(source_md5)) stop("The regression source is incomplete")
pkgload::load_all(snapshot, quiet = TRUE, export_all = TRUE, compile = FALSE,
                  helpers = FALSE, attach_testthat = FALSE)
base <- jsonlite::read_json(file.path(package_dir, "tests", "fixtures", "r_reference.json"),
                            simplifyVector = TRUE)

record <- function(item) {
  call <- item$input
  names(call)[names(call) == "B_initial"] <- "B.initial"
  fit <- do.call(orthoDr_reg, c(call, list(keep.data = TRUE)))
  list(expected = fit[setdiff(names(fit), c("x", "y"))],
       prediction = list(pred = predict(fit, item$testx)$pred))
}

regression <- lapply(base$regression, record)
default_fits <- lapply(base$default_fits[c("reg_local", "reg_seff")], record)
stationary_fits <- lapply(base$stationary_fits["reg_sir"], record)
if (!identical(source_md5, pinned_source_md5(snapshot, files)))
  stop("R source changed during generation")
provenance <- list(source_files_md5 = as.list(source_md5),
  source_hash_convention = "CRLF normalized to LF; all other bytes unchanged",
  source_fixture = "r_reference.json",
  R_version = R.version.string, platform = R.version$platform,
  packages = setNames(lapply(c("orthoDr", needed), function(p) as.character(packageVersion(p))),
                      c("orthoDr", needed)))
jsonlite::write_json(list(provenance = provenance, regression = regression,
                         default_fits = default_fits, stationary_fits = stationary_fits),
                     output, auto_unbox = TRUE, matrix = "rowmajor", digits = 17,
                     pretty = TRUE, null = "null", na = "null")
cat("Wrote ", normalizePath(output), "\n", sep = "")
