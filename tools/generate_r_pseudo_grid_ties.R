#!/usr/bin/env Rscript
# Rscript tools/generate_r_pseudo_grid_ties.R BUILT_R_TREE [OUTPUT]
args <- commandArgs(trailingOnly = TRUE)
if (!length(args)) stop("Supply the corresponding built R source tree")
snapshot <- normalizePath(args[[1L]], mustWork = TRUE)
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1L])
package_dir <- dirname(dirname(normalizePath(script, mustWork = TRUE)))
fixtures <- file.path(package_dir, "tests", "fixtures")
output <- if (length(args) >= 2L) args[[2L]] else file.path(fixtures, "r_pseudo_grid_ties.json")
needed <- c("digest", "jsonlite", "pkgload", "Rcpp", "RcppArmadillo", "dr", "pracma")
missing <- needed[!vapply(needed, requireNamespace, logical(1L), quietly = TRUE)]
if (length(missing)) stop("Install in the shared R library: ", paste(missing, collapse = ", "))

source_fixture <- "r_pseudo.json"
case <- "bw_0.05"
base <- jsonlite::read_json(file.path(fixtures, source_fixture), simplifyVector = TRUE)
manifest <- jsonlite::read_json(file.path(package_dir, "src", "cpp", "provenance.json"))
if (!identical(base$provenance$source_sha, manifest$base_revision))
  stop("The historical fixture must use the same baseline as the native sources")
verify_native <- function() {
  files <- names(manifest$files)
  hashes <- vapply(files, function(name) {
    path <- file.path(snapshot, "src", name)
    if (!file.exists(path)) return(NA_character_)
    contents <- rawToChar(readBin(path, "raw", n = file.size(path)))
    normalized <- charToRaw(gsub("\r\n", "\n", contents, fixed = TRUE))
    digest::digest(normalized, algo = "sha256", serialize = FALSE)
  }, character(1L))
  expected <- vapply(manifest$files, function(file) file$source_sha256, character(1L))
  if (!identical(hashes, expected))
    stop("R native sources differ from the Python adapter's source fingerprints")
}
verify_native()
options(rgl.useNULL = TRUE)
pkgload::load_all(snapshot, quiet = TRUE, compile = FALSE, helpers = FALSE, attach_testthat = FALSE)

item <- base$cases[[case]]
p <- item$input
e <- do.call(orthoDr:::pdose_semi_evaluate, p)
targets <- drop(e$grid.targets)
historical <- drop(item$expected$grid.targets)
stopifnot(identical(which(historical != targets), 4L), targets[4] == .5625,
          e$lambda.init == item$expected$lambda.init)
predicted <- drop(orthoDr:::dosepred(p$B, p$X, p$X, p$bw, e$W)) + e$intercept
verify_native()

provenance <- list(source_fixture = source_fixture, case = case,
  base_source_sha = manifest$base_revision, native_revision = manifest$revision,
  pseudo_solver_sha256 = manifest$files$pdose_semi_solver.cpp$source_sha256,
  source_hash_convention = manifest$source_hash_convention, R_version = R.version.string)
updated <- list(grid.targets = targets, W = drop(e$W), GCV = drop(e$GCV),
                lambda.init = e$lambda.init, training.predictions = predicted)
jsonlite::write_json(list(provenance = provenance, expected = updated), output,
                     auto_unbox = TRUE, digits = NA, pretty = TRUE)
cat("Wrote ", normalizePath(output), "\n", sep = "")
