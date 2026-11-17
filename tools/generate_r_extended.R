#!/usr/bin/env Rscript
# Rscript tools/generate_r_extended.R SNAPSHOT OUTPUT [SOURCE_MANIFEST]
# Uses an isolated pinned source checkout and existing shared R libraries.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2L) stop("Supply isolated source snapshot and output JSON paths")
snapshot <- normalizePath(args[[1L]], mustWork = TRUE)
output <- args[[2L]]
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1L])
reference <- if (length(args) >= 3L) args[[3L]] else
  file.path(dirname(normalizePath(script)), "r_source_manifest.json")
needed <- c("pkgload", "jsonlite", "Rcpp", "RcppArmadillo", "dr", "pracma")
missing <- needed[!vapply(needed, requireNamespace, logical(1L), quietly = TRUE)]
if (length(missing)) stop("Install globally or in the shared user library: ", paste(missing, collapse = ", "))
base <- jsonlite::read_json(reference, simplifyVector = TRUE)
source(file.path(dirname(normalizePath(script)), "check_r_source.R"))
hashes <- unlist(base$source_files_md5)
before <- verify_pinned_source(snapshot, base)
Sys.setenv(RGL_USE_NULL = "TRUE")
pkgload::load_all(snapshot, quiet = TRUE, export_all = TRUE, compile = TRUE,
                  helpers = FALSE, attach_testthat = FALSE)
clean <- function(z) {
  if (is.matrix(z)) return(z)
  if (is.list(z)) return(lapply(z, clean))
  if (!is.null(names(z))) return(lapply(as.list(z), clean))
  unname(z)
}
state <- function(fit) clean(fit[setdiff(names(fit), c("x", "y", "a", "r", "censor"))])
set.seed(271828)
n <- 48L; p <- 4L
x <- matrix(rnorm(n*p), n, p)
x <- sweep(x, 2, c(.6, 1.3, 2, .9), "*")
x[,2] <- x[,2] + .3*x[,1]
y <- x[,1] + .6*x[,2]^2 + rnorm(n, sd=.4)
time <- round(exp(.5*x[,1] + .3*x[,2] + rnorm(n)), 1)
censor <- as.numeric(runif(n) > .25)
a <- runif(n, -.4, 2.1)
reward <- 5 - (a - (.8 + .4*x[,1] - .3*x[,2]))^2 + rnorm(n, sd=.2)
B0 <- qr.Q(qr(matrix(c(1,.2,-.4,.7, .3,1,.5,-.2), p, 2)))
testx <- rbind(x[1:8,,drop=FALSE], matrix(rnorm(20), 5, p))
records <- list()
put <- function(name, family, input) {
  fun <- switch(family, regression=orthoDr_reg, survival=orthoDr_surv, pdose=orthoDr_pdose)
  fit <- do.call(fun, c(input, list(keep.data=TRUE, ncore=1L)))
  init <- input$B.initial
  if (is.null(init) && family == "regression") {
    yy <- as.matrix(scale(input$y)/n^(-1/(input$ndr+5))/sqrt(2))
    init <- initB(x, yy, input$ndr, fit$bw, input$method, 1L)
  }
  if (is.null(init) && family == "survival") {
    yy <- drop(scale(rank(input$y)/(n+1)))/sqrt(2)/silverman(1,n)
    init <- CP_SIR(scale(x),yy,input$censor)$vectors[,seq_len(input$ndr),drop=FALSE]
  }
  if (is.null(init) && family == "pdose") init <- pSAVE(x,a,reward,ndr=input$ndr)
  prepared_initial <- if (family == "pdose" && input$method == "pseudo_direct")
    pseudo_orthonormalize(init) else gramSchmidt(init)$Q
  records[[name]] <<- list(family=family, input=input, expected=state(fit),
    initial=prepared_initial, prediction=clean(unclass(predict(fit,testx))))
}
for (method in c("sir","save","phd","local","seff")) for (d in 1:2) for (init in c("explicit","default")) {
  inp <- list(x=x,y=y,method=method,ndr=d,bw=.7,maxitr=80L)
  if (init == "explicit") inp$B.initial <- B0[,seq_len(d),drop=FALSE]
  put(paste("reg",method,d,init,sep="_"), "regression", inp)
}
for (method in c("forward","dn","dm")) for (d in if (method=="forward") 1L else 1:2) for (init in c("explicit","default")) {
  inp <- list(x=x,y=time,censor=censor,method=method,ndr=d,bw=.7,maxitr=100L)
  if (init == "explicit") inp$B.initial <- B0[,seq_len(d),drop=FALSE]
  put(paste("surv",method,d,init,sep="_"), "survival", inp)
}
for (method in c("direct","pseudo_direct")) for (d in 1:2) for (init in c("explicit","default")) {
  inp <- list(x=x,a=a,r=reward,method=method,ndr=d,bw=.7,K=9L,
              lambda=c(.05,.1,.2),maxitr=100L)
  if (init == "explicit") inp$B.initial <- B0[,seq_len(d),drop=FALSE]
  put(paste("dose",method,d,init,sep="_"), "pdose", inp)
}
put("dose_direct_2_intercept", "pdose", list(x=x,a=a,r=reward,method="direct",ndr=2L,
  bw=.7,K=9L,lambda=c(.05,.1,.2),B.initial=B0,maxitr=100L,
  intercept=TRUE,intercept.bounds=c(.2,1.2)))
selected <- c("reg_sir_2_default","reg_save_1_default","reg_save_2_default",
  "reg_phd_2_default","reg_local_2_default","reg_seff_2_default",
  "surv_forward_1_default","surv_dn_2_default","surv_dm_2_default",
  "dose_pseudo_direct_2_default","dose_direct_2_explicit","dose_direct_2_default",
  "dose_direct_2_intercept")
controls <- control.check(list())
native <- list()
for (name in c("reg_local_2_default","reg_save_2_default","reg_sir_2_default",
               "dose_pseudo_direct_2_default","surv_dm_2_explicit")) {
  item <- records[[name]]; inp <- item$input; d <- inp$ndr; bw <- inp$bw
  if (item$family == "regression") {
    Y <- as.matrix(scale(inp$y)/n^(-1/(d+5))/sqrt(2))
    B <- if (is.null(inp$B.initial)) initB(x,Y,d,bw,inp$method,1L) else gramSchmidt(inp$B.initial)$Q
    prepared <- list(B=B,X=x,Y=Y,bw=bw)
    fun <- get(paste0(inp$method,"_solver")); steps <- c(5L,20L,80L)
  } else if (item$family == "pdose") {
    B <- pseudo_orthonormalize(pSAVE(x,a,reward,ndr=d))
    grid <- seq(min(a),max(a),length.out=inp$K)
    prepared <- list(B=B,X=x,R=reward,A=a,a_seq=grid,lambda=inp$lambda,bw=bw)
    fun <- pdose_semi_solver; steps <- c(1L,5L,20L,100L)
  } else {
    Y <- drop(scale(rank(time)/(n+1)))/sqrt(2)/silverman(1,n)
    ord <- order(Y); X <- scale(x)[ord,,drop=FALSE]; Y <- Y[ord]; C <- censor[ord]
    events <- which(C==1); starts <- match(Y[events],Y)
    ky <- exp(-as.matrix(dist(Y))); ky[outer(Y,Y,"<")] <- 0
    phi <- matrix(0,p,length(events))
    for (j in seq_along(events)) phi[,j] <- apply(X,2,weighted.mean,w=C*ky[,events[j]])-
      colMeans(X[starts[j]:n,,drop=FALSE])
    prepared <- list(B=gramSchmidt(inp$B.initial)$Q,X=unname(X),Phit=phi,
                     Fail_Ind=events,Risk_Start=starts,bw=bw)
    fun <- surv_dm_solver; steps <- c(1L,5L,20L,100L)
  }
  # Keep each independent trajectory isolated from any native input aliasing.
  saved <- serialize(prepared,NULL)
  trajectory <- list()
  for (step in steps) trajectory[[as.character(step)]] <- clean(do.call(fun,
    c(unserialize(saved),controls,list(maxitr=step,verbose=0L,ncore=1L))))
  native[[name]] <- list(family=item$family,method=inp$method,
    prepared=unserialize(saved),controls=controls,trajectory=trajectory)
}
projector <- function(B) {Q <- qr.Q(qr(B)); tcrossprod(Q)}
sensitivity <- list()
for (name in c("reg_sir_2_default","reg_save_2_default","reg_local_2_explicit",
  "reg_local_2_default","reg_seff_1_default","dose_pseudo_direct_2_explicit",
  "dose_pseudo_direct_2_default","reg_phd_2_explicit","surv_dm_2_explicit")) {
  item <- records[[name]]; inp <- item$input; baseline <- item$expected
  fun <- switch(item$family,regression=orthoDr_reg,survival=orthoDr_surv,pdose=orthoDr_pdose)
  sensitivity[[name]] <- list()
  for (delta in c(0,1e-14,-1e-14,1e-13,-1e-13,1e-12,-1e-12,1e-11,-1e-11)) {
    b <- item$initial; b[1,1] <- b[1,1]+delta; inp$B.initial <- b
    fit <- do.call(fun,c(inp,list(ncore=1L,keep.data=TRUE)))
    prediction <- unclass(predict(fit,testx))
    diffs <- mapply(function(a,b) max(abs(a-b)),prediction,item$prediction)
    sensitivity[[name]][[length(sensitivity[[name]])+1L]] <- list(delta=delta,
      fn=fit$fn,itr=fit$itr,converge=fit$converge,fn_difference=abs(fit$fn-baseline$fn),
      projector_difference=norm(projector(fit$B)-projector(baseline$B),"F"),
      prediction_difference=max(diffs))
  }
}
evaluations <- list()
for (name in selected[grepl("^dose_direct",selected)]) {
  item <- records[[name]]; inp <- item$input; fit <- item$expected
  intercept <- isTRUE(inp$intercept)
  ev <- pdose_direct_evaluate(fit$B,x,a,reward,fit$bw,drop(fit$W),.1,0,
    intercept,inp$intercept.bounds,if(intercept) fit$intercept else 0)
  fields <- c("fn","value","penalty","gB","gW","gradient.norm","hessian.W")
  if (intercept) fields <- c(fields,"g.intercept","hessian.policy")
  evaluations[[name]] <- clean(ev[fields])
}
cases <- records[selected]
for (name in names(cases)) {
  keys <- names(cases[[name]]$input)
  keys[keys=="B.initial"] <- "B_initial"
  keys[keys=="lambda"] <- "lambda_init"
  keys[keys=="intercept.bounds"] <- "intercept_bounds"
  names(cases[[name]]$input) <- keys
}
after <- verify_pinned_source(snapshot, base)
if (!identical(unname(after),unname(before))) stop("Snapshot source changed during loading")
provenance <- list(source_sha=base$source_sha,source_files_md5=as.list(before),
  source_hash_convention=base$source_hash_convention,seed=271828L,
  R_version=R.version.string,platform=R.version$platform,
  packages=lapply(c("orthoDr",needed),function(p) as.character(packageVersion(p))),
  session_info=capture.output(sessionInfo()),audited_fits=39L,
  purpose="Extended d=2 fixed-state parity; analytic pseudo-direct and direct fits; inherited finite-difference sensitivity evidence")
names(provenance$packages) <- c("orthoDr",needed)
jsonlite::write_json(list(provenance=provenance,testx=testx,cases=cases,native=native,
  direct_evaluations=evaluations,sensitivity=sensitivity),output,
  auto_unbox=TRUE,digits=17,null="null",pretty=FALSE)
cat("Wrote",length(cases),"selected cases and",length(native),"native trajectories\n")
