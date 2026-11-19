#!/usr/bin/env Rscript
# Rscript tools/generate_r_seff_centered.R SNAPSHOT [OUTPUT]
# The verified pinned wrapper on centered X is an independent oracle for the
# corrected public preprocessing. Native routines and old fixtures stay pinned.
args <- commandArgs(trailingOnly = TRUE)
if (!length(args)) stop("Supply the pinned R source snapshot directory")
snapshot <- normalizePath(args[[1L]], mustWork = TRUE)
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1L])
package_dir <- dirname(dirname(normalizePath(script, mustWork = TRUE)))
output <- if (length(args) >= 2L) args[[2L]] else
  file.path(package_dir, "tests", "fixtures", "r_seff_centered.json")
needed <- c("pkgload", "jsonlite", "Rcpp", "RcppArmadillo", "dr", "pracma")
missing <- needed[!vapply(needed, requireNamespace, logical(1L), quietly = TRUE)]
if (length(missing)) stop("Install in the shared R library: ", paste(missing, collapse = ", "))
pin <- jsonlite::read_json(file.path(package_dir, "tools", "r_source_manifest.json"))
source(file.path(package_dir, "tools", "check_r_source.R"))
before <- verify_pinned_source(snapshot, pin)
pkgload::load_all(snapshot, quiet = TRUE, export_all = TRUE, compile = TRUE,
                  helpers = FALSE, attach_testthat = FALSE)
base <- jsonlite::read_json(file.path(package_dir, "tests", "fixtures", "r_reference.json"),
                            simplifyVector = TRUE)
clean <- function(z) {
  if (is.matrix(z)) return(z)
  if (is.list(z)) return(lapply(z, clean))
  if (!is.null(names(z))) return(lapply(as.list(z), clean))
  unname(z)
}
record <- function(item) {
  call <- item$input
  call$x <- sweep(call$x, 2L, colMeans(call$x), "-")
  names(call)[names(call) == "B_initial"] <- "B.initial"
  fit <- do.call(orthoDr_reg, c(call, list(keep.data = TRUE)))
  # Prediction uses the original predictor coordinates, just as the corrected
  # public wrapper does. Centering changes no directions' coefficient units.
  fit$x <- item$input$x
  list(expected = clean(fit[setdiff(names(fit), c("x", "y"))]),
       prediction = clean(unclass(predict(fit, item$testx))))
}
regression <- lapply(base$regression[c("seff_d1", "seff_d2")], record)
defaults <- lapply(base$default_fits["reg_seff"], record)
after <- verify_pinned_source(snapshot, pin)
if (!identical(unname(before), unname(after))) stop("Snapshot changed during generation")
provenance <- list(source_sha = pin$source_sha,
  source_files_md5 = as.list(before), source_hash_convention = pin$source_hash_convention,
  source_fixture = "r_reference.json",
  oracle = "Pinned R wrapper fitted to column-centered X; raw X restored for prediction",
  R_version = R.version.string, platform = R.version$platform,
  packages = setNames(lapply(c("orthoDr", needed), function(p) as.character(packageVersion(p))),
                      c("orthoDr", needed)),
  session_info = capture.output(sessionInfo()))
jsonlite::write_json(list(provenance = provenance, regression = regression,
                         default_fits = defaults), output, auto_unbox = TRUE,
                     matrix = "rowmajor", digits = 17, pretty = TRUE,
                     null = "null", na = "null")
cat("Wrote ", normalizePath(output), "\n", sep = "")
