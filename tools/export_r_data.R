# Export the exact committed R dataset matrices into NumPy's binary format.
# Run with a shared R installation and the globally installed jsonlite package:
# Rscript tools/export_r_data.R path/to/pinned/r-source [orthodr/data]
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1L || length(args) > 2L)
  stop("Usage: Rscript tools/export_r_data.R R_SOURCE_DIRECTORY [OUTPUT_DIRECTORY]")
source_dir <- normalizePath(args[1L], mustWork = TRUE)
output_dir <- if (length(args) == 2L) args[2L] else file.path("orthodr", "data")
source_files <- file.path(source_dir, "data", c("skcm.clinical.rda", "skcm.melgene.rda"))
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1L])
package_dir <- dirname(dirname(normalizePath(script, mustWork = TRUE)))
expected <- jsonlite::read_json(file.path(package_dir, "orthodr", "data", "skcm.json"))$provenance
actual_md5 <- unname(tools::md5sum(source_files))
if (!identical(expected$source_commit, "d99bfa8") || anyNA(actual_md5) ||
    !identical(actual_md5, unlist(expected$source_md5, use.names = FALSE)))
  stop("Dataset files differ from the pinned R revision; use a git archive of d99bfa8")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
storage <- new.env(parent = emptyenv())
for (path in source_files) load(path, envir = storage)
clinical <- as.matrix(storage$skcm.clinical)
genes <- as.matrix(storage$skcm.melgene)

write_npy <- function(value, path) {
  stopifnot(is.matrix(value), is.numeric(value))
  # R NA uses a signalling-NaN bit pattern. Preserve missingness using a quiet
  # NaN so ordinary NumPy arithmetic does not emit invalid-operation warnings.
  value[is.na(value)] <- NaN
  shape <- paste(dim(value), collapse = ", ")
  header <- paste0("{'descr': '<f8', 'fortran_order': True, 'shape': (", shape, "), }")
  # Version 1.0 uses a ten-byte prefix and a little-endian uint16 header size.
  padding <- (64L - ((10L + nchar(header, type = "bytes") + 1L) %% 64L)) %% 64L
  header <- paste0(header, paste(rep(" ", padding), collapse = ""), "\n")
  connection <- file(path, "wb")
  on.exit(close(connection))
  writeBin(as.raw(c(0x93, utf8ToInt("NUMPY"), 1L, 0L)), connection)
  writeBin(as.integer(nchar(header, type = "bytes")), connection, size = 2L, endian = "little")
  writeChar(header, connection, eos = NULL, useBytes = TRUE)
  writeBin(as.double(value), connection, size = 8L, endian = "little")
}
write_npy(clinical, file.path(output_dir, "skcm_clinical.npy"))
write_npy(genes, file.path(output_dir, "skcm_genes.npy"))
metadata <- list(
  clinical_columns = colnames(clinical), gene_names = colnames(genes),
  subject_ids = rownames(clinical), gene_subject_ids = rownames(genes),
  provenance = list(
    source = "orthoDr committed R data", source_commit = "d99bfa8",
    source_files = basename(source_files), source_md5 = unname(tools::md5sum(source_files)),
    identifiers_aligned = identical(rownames(clinical), rownames(genes)),
    clinical_shape = dim(clinical), gene_shape = dim(genes),
    clinical_missing = colSums(is.na(clinical)), gene_missing = colSums(is.na(genes)),
    clinical_sums = colSums(clinical, na.rm = TRUE), gene_sums = colSums(genes),
    clinical_sum_squares = colSums(clinical ^ 2, na.rm = TRUE),
    gene_sum_squares = colSums(genes ^ 2),
    first_clinical = clinical[1L, ], first_genes = genes[1L, ],
    last_clinical = clinical[nrow(clinical), ], last_genes = genes[nrow(genes), ],
    missing_age_subject_ids = rownames(clinical)[is.na(clinical[, "Age"])],
    event_count = sum(clinical[, "Censor"] == 1)
  )
)
jsonlite::write_json(metadata, file.path(output_dir, "skcm.json"), auto_unbox = TRUE,
                     digits = NA, pretty = TRUE, na = "null")
cat("Exported", nrow(clinical), "aligned clinical and gene-expression rows\n")
