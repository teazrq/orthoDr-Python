# Shared by the fixture generators. Native objects are rebuilt after this check.
pinned_source_files <- function(snapshot) {
  r_files <- list.files(file.path(snapshot, "R"), all.files = TRUE,
                        recursive = TRUE, no.. = TRUE)
  native <- list.files(file.path(snapshot, "src"), all.files = TRUE,
                       recursive = TRUE, no.. = TRUE)
  native <- native[!grepl("\\.(o|so|dll)$|(^|/)symbols\\.rds$", native)]
  c("DESCRIPTION", "NAMESPACE", paste0("R/", r_files), paste0("src/", native))
}

pinned_source_md5 <- function(snapshot, files) {
  paths <- file.path(snapshot, files)
  result <- tools::md5sum(paths)
  names(result) <- files
  # Git checkouts and pkgbuild can change text line endings. Normalize CRLF
  # to LF for every pinned text file, without changing any other source bytes.
  temporary <- tempfile("orthodr-source-fingerprint-")
  on.exit(unlink(temporary))
  for (i in seq_along(files)) {
    if (!file.exists(paths[i])) next
    content <- readBin(paths[i], "raw", n = file.info(paths[i])$size)
    content <- gsub("\r\n", "\n", rawToChar(content), fixed = TRUE)
    writeBin(charToRaw(content), temporary)
    result[i] <- unname(tools::md5sum(temporary))
  }
  result
}

verify_pinned_source <- function(snapshot, reference) {
  expected <- unlist(reference$source_files_md5)
  files <- pinned_source_files(snapshot)
  actual <- pinned_source_md5(snapshot, names(expected))
  if (!identical(reference$source_sha, "a9588d8dd73c39188d1c46f371763addb10bee9a") ||
      !setequal(files, names(expected)) || anyNA(actual) ||
      !identical(unname(actual), unname(expected)))
    stop("R source differs from the pinned revision; use a git archive of ",
         reference$source_sha, " before regenerating fixtures")
  actual
}
