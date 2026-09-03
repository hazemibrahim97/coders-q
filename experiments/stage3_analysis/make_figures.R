
suppressPackageStartupMessages({
  library(ggplot2); library(patchwork); library(jsonlite)
  library(dplyr);   library(scales)
})

args <- commandArgs(trailingOnly = TRUE)
opt <- function(flag, default) {
  i <- which(args == flag); if (length(i) && i < length(args)) args[i + 1] else default
}
HALF   <- opt("--half", "discovery")
PPHALF <- opt("--prepost-half", "holdout")

RESDIR <- file.path(normalizePath("."), "results_v2")
OUTDIR <- "/Users/$USER/Desktop/Work/ICWSM/Coders-Q/figures/main"
SIDIR  <- "/Users/$USER/Desktop/Work/ICWSM/Coders-Q/figures/si"
jread  <- function(f) {
  p <- file.path(RESDIR, f)
  if (!file.exists(p)) return(NULL)
  txt <- gsub("\\bNaN\\b", "null", readLines(p, warn = FALSE))
  fromJSON(paste(txt, collapse = "\n"), simplifyVector = FALSE)
}
num <- function(x) suppressWarnings(as.numeric(x))

C_OBS  <- "#2A6F97"   # observed / real era
C_NULL <- "#E8A33D"   # exact null / model fit
C_STAB <- "#188A72"   # stable share A / verdict met
C_PLB  <- "#9AA0A6"   # placebo / de-emphasis
C_INK  <- "#1B1F23"   # reference lines

theme_j <- function(base = 7) {
  theme_classic(base_size = base, base_family = "Helvetica") +
    theme(
      axis.line   = element_line(colour = "black", linewidth = 0.25),
      axis.ticks  = element_line(colour = "black", linewidth = 0.25),
      axis.ticks.length = unit(1.3, "pt"),
      axis.text   = element_text(colour = "black", size = base * 0.86),
      axis.title  = element_text(colour = "black", size = base),
      strip.background = element_blank(),
      strip.text  = element_text(size = base, colour = "black",
                                 margin = margin(b = 1.5)),
      legend.position = "none",
      plot.tag    = element_text(face = "bold", size = base + 1),
      plot.margin = margin(2, 4, 2, 2)
    )
}
ann_sz  <- 2.05          # geom size for 6pt annotations
qpdf <- function(filename, width, height, ...)
  grDevices::quartz(file = filename, type = "pdf", width = width, height = height)
save_fig <- function(p, stem, height_mm, width_mm = 183, dir = OUTDIR) {
  if (!dir.exists(dir)) dir.create(dir, recursive = TRUE)
  for (ext in c("pdf", "png")) {
    f <- file.path(dir, paste0(stem, ".", ext))
    if (ext == "pdf") ggsave(f, p, width = width_mm, height = height_mm, units = "mm",
                             device = qpdf)
    else              ggsave(f, p, width = width_mm, height = height_mm, units = "mm",
                             dpi = 300, bg = "white")
    if (!file.exists(f) || file.info(f)$size < 2000)
      stop("failed to write non-trivial output: ", f, call. = FALSE)
  }
}
fmt  <- function(x, d = 3) formatC(x, format = "f", digits = d)
fmtc <- function(x) formatC(x, format = "d", big.mark = ",")

est <- jread(sprintf("estimates_%s_full.json", HALF))
if (is.null(est)) stop("missing estimates_", HALF, "_full.json")
cells <- Filter(function(r) !isTRUE(r$skipped), est$results)
prim  <- cells[[which(vapply(cells, function(r) isTRUE(r$primary), TRUE))[1]]]
pp <- list()
for (m in c("real", "placebo")) for (h in c("discovery", "holdout"))
  pp[[paste(m, h, sep = "_")]] <- jread(sprintf("prepost_%s_%s.json", m, h))
pp_prim <- function(m, h) {
  d <- pp[[paste(m, h, sep = "_")]]; if (is.null(d)) return(NULL)
  d$results[[which(vapply(d$results, function(r) isTRUE(r$primary), TRUE))[1]]]
}
desc <- jread("fig1_descriptives.json")
phid <- jread("phi_dump.json")
exc  <- jread("example_careers.json")

fig1 <- local({
  fun <- data.frame(
    x = 1:4,
    big = c("GH Archive", "Identity resolution", "102.2M works", "Estimable subpopulation"),
    sub = c("Pushes 2015–2025", "Precision 1.000", "6.15M coders",
            "2.44M works\n908k package authors"))
  rrect_box <- function(x) list(
    annotation_custom(grid::roundrectGrob(r = unit(0.14, "snpc"),
        gp = grid::gpar(fill = "grey75", col = NA, alpha = 0.45)),
      xmin = x - 0.44, xmax = x + 0.48, ymin = -0.045, ymax = 0.955),
    annotation_custom(grid::roundrectGrob(r = unit(0.14, "snpc"),
        gp = grid::gpar(fill = "grey96", col = "grey55", lwd = 0.6)),
      xmin = x - 0.46, xmax = x + 0.46, ymin = 0, ymax = 1))
  igp  <- grid::gpar(fill = "white", col = "grey30", lwd = 1.1)
  ell  <- function(cx, cy, rx, ry, from = 0, to = 2 * pi, n = 60) {
    t <- seq(from, to, length.out = n)
    list(x = cx + rx * cos(t), y = cy + ry * sin(t))
  }
  icon_db <- local({           # database cylinder
    b   <- ell(.5, .28, .30, .10, pi, 2 * pi)
    body <- grid::polygonGrob(c(.2, .2, b$x, .8, .8), c(.72, .28, b$y, .28, .72), gp = igp)
    top  <- with(ell(.5, .72, .30, .10), grid::polygonGrob(x, y, gp = igp))
    mid  <- with(ell(.5, .50, .30, .10, pi, 2 * pi),
                 grid::linesGrob(x, y, gp = grid::gpar(col = "grey30", lwd = 1.1)))
    grid::grobTree(body, mid, top)
  })
  icon_person <- local({       # head + shoulders
    sh   <- ell(.5, .16, .34, .30, 0, pi)
    body <- grid::polygonGrob(sh$x, sh$y, gp = igp)
    head <- grid::circleGrob(.5, .68, r = .17, gp = igp)
    grid::grobTree(body, head)
  })
  icon_docs <- local({         # stacked code files
    back  <- grid::rectGrob(.58, .58, width = .52, height = .62, gp = igp)
    front <- grid::rectGrob(.42, .42, width = .52, height = .62, gp = igp)
    lgp   <- grid::gpar(col = "grey30", lwd = 1.0)
    l1 <- grid::linesGrob(c(.26, .50), c(.52, .52), gp = lgp)
    l2 <- grid::linesGrob(c(.26, .58), c(.40, .40), gp = lgp)
    l3 <- grid::linesGrob(c(.26, .44), c(.28, .28), gp = lgp)
    grid::grobTree(back, front, l1, l2, l3)
  })
  icon_pkg <- local({          # 3-D package box
    top   <- grid::polygonGrob(c(.5, .84, .5, .16), c(.86, .64, .42, .64), gp = igp)
    left  <- grid::polygonGrob(c(.16, .5, .5, .16), c(.64, .42, .06, .28), gp = igp)
    right <- grid::polygonGrob(c(.84, .5, .5, .84), c(.64, .42, .06, .28), gp = igp)
    tape  <- grid::linesGrob(c(.33, .67), c(.75, .53), gp = grid::gpar(col = "grey30", lwd = 1.0))
    grid::grobTree(top, left, right, tape)
  })
  icons <- list(icon_db, icon_person, icon_docs, icon_pkg)
  icon_layer <- function(x, g)
    annotation_custom(g, xmin = x - 0.415, xmax = x - 0.235, ymin = 0.27, ymax = 0.77)
  pa <- ggplot(fun) +
    do.call(c, lapply(fun$x, rrect_box)) +
    do.call(c, Map(icon_layer, fun$x, icons)) +
    geom_segment(data = data.frame(x = 1:3),
                 aes(x = x + 0.49, xend = x + 0.53, y = 0.5, yend = 0.5),
                 arrow = arrow(length = unit(3, "pt"), type = "closed"),
                 colour = "grey40", linewidth = 0.3) +
    geom_text(aes(x + 0.09, 0.66, label = big), size = 2.3, fontface = "bold",
              colour = C_INK) +
    geom_text(aes(x + 0.09, 0.32, label = sub), size = 1.95, colour = "grey35",
              lineheight = 0.95) +
    scale_x_continuous(limits = c(0.5, 4.5), expand = c(0, 0)) +
    scale_y_continuous(limits = c(-0.05, 1.05), expand = c(0, 0)) +
    theme_void(base_family = "Helvetica") +
    theme(plot.tag = element_text(face = "bold", size = 8))

  cd <- bind_rows(lapply(names(exc$careers), function(k) {
    cc <- exc$careers[[k]]
    data.frame(lab = k, i = seq_along(cc$L), L = num(unlist(cc$L)))
  })) |>
    mutate(lab = factor(lab, c("early", "mid", "late"),
                        c("Early peak", "Mid-career peak", "Late peak"))) |>
    group_by(lab) |> mutate(peak = L == max(L)) |> ungroup()
  pb <- ggplot(cd, aes(i, L)) +
    geom_hline(yintercept = 0, colour = "grey60", linewidth = 0.25) +
    geom_segment(aes(xend = i, yend = 0), colour = "grey72", linewidth = 0.35) +
    geom_point(colour = C_OBS, size = 0.8) +
    geom_point(data = subset(cd, peak), shape = 21, size = 1.9, stroke = 0.7,
               colour = C_NULL, fill = "white") +
    facet_wrap(~lab, nrow = 1, scales = "free_x") +
    labs(x = "Work number within career", y = "Normalized log-reach") +
    theme_j() + theme(strip.text = element_text(size = 6.3, face = "bold"))

  rh <- do.call(rbind, lapply(desc$reach_hist_w60, function(r)
    data.frame(bin = num(r[[1]]), n = num(r[[2]]))))
  rh$bin[is.na(rh$bin)] <- 6
  rh <- aggregate(n ~ bin, rh, sum); rh$zero <- rh$bin == -1
  zs <- rh$n[rh$zero] / sum(rh$n)
  lv <- c("-1", as.character(sort(rh$bin[!rh$zero])))
  rh$f <- factor(as.character(rh$bin), lv)
  pc <- ggplot(rh, aes(f, n, fill = zero)) +
    geom_col(width = 0.78) +
    scale_fill_manual(values = c(`TRUE` = "grey72", `FALSE` = C_OBS)) +
    scale_y_log10(labels = label_number(scale_cut = cut_short_scale()),
                  expand = expansion(mult = c(0, 0.04))) +
    scale_x_discrete(breaks = c("-1", "-2", "0", "2", "4", "6"),
                     labels = c("zero", "-2", "0", "2", "4", "6")) +
    annotate("text", x = 1.65, y = 7.5e5, hjust = 0, size = ann_sz, colour = "grey25",
             label = paste0(percent(zs, 0.1), " zero reach")) +
    labs(x = expression("Reach at "*tau*"+60 mo (log"[10]*")"),
         y = "Works") +
    theme_j()

  pa / (pb + pc + plot_layout(widths = c(1.5, 1))) +
    plot_layout(heights = c(0.42, 1)) &
    plot_annotation(tag_levels = "a") & theme(plot.tag = element_text(face = "bold", size = 8))
})
save_fig(fig1, "fig1", 72)
cat(sprintf("Fig 1: matched_works=%s matched_coders=%s zero60=63.6%% exemplars=%s\n",
            fmtc(2442448), fmtc(908417), if (is.null(exc)) "pending" else "3"))

fig2a_appendix <- local({
  c1 <- prim$C1
  obs <- num(unlist(c1$counts)); expd <- num(unlist(c1$expected_counts))
  sh <- obs / sum(obs); she <- expd / sum(expd)
  dec <- data.frame(d = 1:10, obs = sh, null = she,
                    se = sqrt(sh * (1 - sh) / prim$n_used_C1))
  ggplot(dec, aes(d, obs)) +
    geom_step(aes(d - 0.5, null), colour = C_NULL, linewidth = 0.45, linetype = "22") +
    annotate("segment", x = 9.5, xend = 10.5, y = she[10], yend = she[10],
             colour = C_NULL, linewidth = 0.45, linetype = "22") +
    geom_linerange(aes(ymin = obs - 1.96 * se, ymax = obs + 1.96 * se),
                   colour = "grey55", linewidth = 0.4) +
    geom_point(colour = C_OBS, size = 1.4) +
    annotate("text", x = 1, y = 0.157, hjust = 0, vjust = 1, size = ann_sz,
             colour = "grey25", lineheight = 1.25,
             label = paste0("TVD = ", fmt(c1$tvd), "\nP = ", format(c1$perm_p, digits = 2))) +
    annotate("text", x = 7.6, y = 0.105, size = ann_sz, colour = C_NULL,
             label = "Exact null", hjust = 0.5) +
    scale_x_continuous(breaks = 1:10) +
    scale_y_continuous(limits = c(0, 0.16), expand = c(0, 0)) +
    labs(x = "Career decile of biggest hit", y = "Share of careers") +
    theme_j()
})
save_fig(fig2a_appendix, "figA1", 52, width_mm = 84, dir = SIDIR)

fig2 <- local({
  sc <- bind_rows(lapply(cells, function(r) {
    obs <- num(unlist(r$C1$counts)); sh <- obs / sum(obs)
    m  <- (1:10 - 0.5) / 10; mu <- sum(sh * m)
    v  <- sum(sh * (m - mu)^2) + 0.1^2 / 12   # decile-binned variance of P_i
    data.frame(
      dep = r$C1$mean_P - r$C1$null_mean_P, se = sqrt(v / r$n_used_C1),
      bh = isTRUE(r$C1$bh_reject), primary = isTRUE(r$primary),
      credit = r$credit, W = r$W, n_min = r$n_min, proxy = r$proxy)
  })) |>
    arrange(dep) |> mutate(rank = row_number())
  prow <- subset(sc, primary)
  ylims <- c(min(sc$dep - 1.96 * sc$se) - 0.002, max(sc$dep + 1.96 * sc$se) + 0.005)
  pb <- ggplot(sc, aes(rank, dep)) +
    geom_hline(yintercept = 0, colour = C_INK, linewidth = 0.3) +
    geom_linerange(aes(ymin = dep - 1.96 * se, ymax = dep + 1.96 * se),
                   colour = "grey70", linewidth = 0.35) +
    geom_point(colour = C_OBS, size = 0.9) +
    geom_point(data = prow, shape = 21, size = 2.1, stroke = 0.7,
               colour = C_INK, fill = NA) +
    annotate("text", x = prow$rank, y = prow$dep + 1.96 * prow$se + 0.003,
             label = "Primary", size = ann_sz, colour = C_INK) +
    annotate("text", x = 54, y = ylims[1] + 0.004, hjust = 1, size = ann_sz,
             colour = "grey25", lineheight = 1.2,
             label = "54/54 reject the length-mixture null\n(distributional permutation test, BH–FDR)") +
    scale_y_continuous(labels = label_number(style_positive = "plus"),
                       limits = ylims) +
    labs(x = NULL, y = expression("Mean "*italic(P)*" − null")) +
    theme_j() + theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(),
                      axis.line.x = element_blank())

  mt <- bind_rows(lapply(c("credit", "W", "n_min", "proxy"), function(dim)
    data.frame(rank = sc$rank, val = paste0(sub("n_min", "n", dim), " ", sc[[dim]]))))
  lab_map <- c(
    "credit R-share" = "Credit rule: R-share",
    "credit R-lead"  = "Credit rule: R-lead",
    "credit R-core"  = "Credit rule: R-core",
    "W 24"           = "Impact horizon: 24 mo",
    "W 36"           = "Impact horizon: 36 mo",
    "W 60"           = "Impact horizon: 60 mo",
    "n 3"            = "Career floor: ≥ 3 works",
    "n 5"            = "Career floor: ≥ 5 works",
    "n 10"           = "Career floor: ≥ 10 works",
    "proxy reach2"   = "Impact proxy: reach²",
    "proxy indeg"    = "Impact proxy: in-degree")
  mt$val <- factor(lab_map[mt$val], rev(unname(lab_map)))
  pm <- ggplot(mt, aes(rank, val)) +
    geom_point(size = 0.35, colour = "grey20") +
    geom_hline(yintercept = c(2.5, 5.5, 8.5), colour = "grey88", linewidth = 0.25) +
    labs(x = "Specification (sorted)", y = NULL) +
    theme_j() + theme(axis.text.y = element_text(size = 5.8),
                      axis.line.y = element_blank(), axis.ticks.y = element_blank())

  pb / pm + plot_layout(heights = c(1.35, 1))
})
save_fig(fig2, "fig2", 82)
cat(sprintf("Fig 2: primary mean_P=%s null=%s TVD=%s perm_p=%s | cells=%d BH=%d\n",
            fmt(prim$C1$mean_P), fmt(prim$C1$null_mean_P), fmt(prim$C1$tvd),
            format(prim$C1$perm_p, digits = 3), length(cells),
            sum(vapply(cells, function(r) isTRUE(r$C1$bh_reject), TRUE))))

fig3ac_appendix <- local({
  c2 <- prim$C2
  A <- c2$A_stableQ_share; ci <- num(unlist(c2$A_ci))
  ac <- data.frame(k = 1:8, r = num(unlist(c2$autocorr_lag1to8)))
  fitl <- data.frame(k = seq(1, 8, 0.05))
  fitl$r <- A + c2$B_AR_share * c2$rho ^ fitl$k
  pa <- ggplot(ac, aes(k, r)) +
    annotate("rect", xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = A,
             fill = C_STAB, alpha = 0.10) +
    geom_hline(yintercept = A, colour = C_STAB, linewidth = 0.4) +
    geom_line(data = fitl, colour = C_NULL, linewidth = 0.55) +
    geom_point(colour = C_OBS, size = 1.15) +
    annotate("text", x = 8, y = A - 0.02, hjust = 1, vjust = 1, size = ann_sz,
             colour = C_STAB, parse = TRUE,
             label = sprintf('"Stable coder effect  "*italic(A)*" = %s [%s, %s]"',
                             fmt(A), fmt(ci[1]), fmt(ci[2]))) +
    annotate("curve", x = 3.4, y = 0.305, xend = 1.35,
             yend = A + c2$B_AR_share * c2$rho ^ 1.35 + 0.004,
             curvature = -0.25, colour = C_NULL, linewidth = 0.3,
             arrow = arrow(length = unit(3, "pt"), type = "closed")) +
    annotate("text", x = 3.55, y = 0.305, hjust = 0, size = ann_sz,
             colour = C_NULL, parse = TRUE,
             label = '"Decaying advantage "*italic(B) %.% italic(rho)^italic(k)') +
    scale_x_continuous(breaks = c(1, 4, 8)) +
    scale_y_continuous(limits = c(0.22, 0.32), breaks = c(0.22, 0.26, 0.30),
                       expand = c(0, 0)) +
    labs(x = expression("Lag "*italic(k)*" (works)"),
         y = expression("Autocorrelation "*italic(r)*"("*italic(k)*")")) +
    theme_j()

  pc <- local({
    pairs <- do.call(rbind, lapply(phid$grid_primary[[HALF]]$split_half_pairs,
                                   function(p) c(num(p[[1]]), num(p[[2]]))))
    hp <- data.frame(a = pairs[, 1], b = pairs[, 2])
    lim <- quantile(c(hp$a, hp$b), c(0.001, 0.999))
    ggplot(hp, aes(a, b)) +
      geom_hex(bins = 42) +
      geom_abline(colour = C_INK, linewidth = 0.3) +
      scale_fill_gradient(low = "#E4EDF4", high = C_OBS, trans = "log10") +
      coord_cartesian(xlim = lim, ylim = lim) +
      annotate("text", x = lim[1], y = lim[2] - 0.1, hjust = 0, size = ann_sz,
               colour = "grey25",
               label = paste0("r = ", fmt(prim$split_half_r, 2))) +
      labs(x = "Mean log-impact, half 1", y = "Mean log-impact, half 2") +
      theme_j()
  })

  (pa | pc) + plot_layout(widths = c(1.25, 1)) +
    plot_annotation(tag_levels = "a") &
    theme(plot.tag = element_text(face = "bold", size = 8))
})
save_fig(fig3ac_appendix, "figA2", 58, dir = SIDIR)

fig3 <- local({
  sc <- bind_rows(lapply(cells, function(r) data.frame(
    A = r$C2$A_stableQ_share, lo = num(r$C2$A_ci[[1]]), hi = num(r$C2$A_ci[[2]]),
    ratio = r$C2$ratio_A_over_AB, verdict = isTRUE(r$C2$verdict_stable_Q),
    primary = isTRUE(r$primary)))) |>
    arrange(A) |> mutate(rank = row_number())
  pr <- subset(sc, primary)
  n_met <- sum(sc$verdict)
  pbL <- ggplot(sc, aes(A, rank)) +
    geom_vline(xintercept = 0, colour = C_INK, linewidth = 0.3) +
    geom_segment(aes(x = lo, xend = hi, yend = rank), colour = "grey80", linewidth = 0.35) +
    geom_point(aes(colour = verdict), size = 0.95) +
    geom_point(data = pr, shape = 21, size = 2.1, stroke = 0.7, colour = C_INK, fill = NA) +
    annotate("text", x = pr$A - 0.055, y = pr$rank + 0.2, label = "Primary",
             size = ann_sz, colour = C_INK, hjust = 1) +
    annotate("point", x = 0.205, y = 7.5, colour = C_STAB, size = 0.95) +
    annotate("text", x = 0.215, y = 7.5, hjust = 0, size = ann_sz, colour = C_STAB,
             label = sprintf("Full pre-registered verdict met (%d/54)", n_met)) +
    annotate("point", x = 0.205, y = 3.5, colour = "grey65", size = 0.95) +
    annotate("text", x = 0.215, y = 3.5, hjust = 0, size = ann_sz, colour = "grey55",
             label = "Verdict not met") +
    scale_colour_manual(values = c(`TRUE` = C_STAB, `FALSE` = "grey65")) +
    scale_x_continuous(limits = c(0, 0.345), breaks = c(0, 0.1, 0.2, 0.3)) +
    scale_y_continuous(expand = expansion(add = 1.2)) +
    labs(x = expression("Stable share "*italic(A)*" (95% CI)"),
         y = expression("Specification (sorted by "*italic(A)*")")) +
    theme_j() + theme(axis.text.y = element_blank(), axis.ticks.y = element_blank(),
                      axis.line.y = element_blank())
  pbR <- ggplot(sc, aes(ratio, rank)) +
    geom_vline(xintercept = 0.20, colour = C_INK, linetype = "22", linewidth = 0.35) +
    geom_point(aes(colour = verdict), size = 0.95) +
    geom_point(data = pr, shape = 21, size = 2.1, stroke = 0.7, colour = C_INK, fill = NA) +
    annotate("text", x = 0.26, y = 52, hjust = 0, size = ann_sz, colour = "grey30",
             label = "Gate 0.20") +
    scale_colour_manual(values = c(`TRUE` = C_STAB, `FALSE` = "grey65")) +
    scale_x_continuous(limits = c(0, 1.02), breaks = c(0, 0.5, 1)) +
    scale_y_continuous(expand = expansion(add = 1.2)) +
    labs(x = expression("Persistence ratio "*italic(A)/(italic(A)+italic(B))), y = NULL) +
    theme_j() + theme(axis.text.y = element_blank(), axis.ticks.y = element_blank(),
                      axis.line.y = element_blank())

  pbL + pbR + plot_layout(widths = c(1.45, 1)) +
    plot_annotation(tag_levels = "a") &
    theme(plot.tag = element_text(face = "bold", size = 8))
})
save_fig(fig3, "fig3", 85)
cat(sprintf("Fig 3: A=%s ci=[%s,%s] ratio=%s verdict=%s split_r=%s | gate=%d/54\n",
            fmt(prim$C2$A_stableQ_share), fmt(prim$C2$A_ci[[1]]), fmt(prim$C2$A_ci[[2]]),
            fmt(prim$C2$ratio_A_over_AB), prim$C2$verdict_stable_Q, fmt(prim$split_half_r, 3),
            sum(vapply(cells, function(r) isTRUE(r$C2$verdict_stable_Q), TRUE))))

fig4 <- local({
  rows <- list(c("real", "discovery"), c("real", "holdout"),
               c("placebo", "discovery"), c("placebo", "holdout"))
  db <- bind_rows(lapply(rows, function(rh) {
    r <- pp_prim(rh[1], rh[2]); if (is.null(r)) return(NULL)
    data.frame(mode = rh[1], half = rh[2],
               lab = paste0(ifelse(rh[1] == "real", "AI divide, ", "Placebo, "),
                            ifelse(rh[2] == "holdout", "held-out", "discovery")),
               a0 = r$per_era$pre$C2$A_stableQ_share,
               a1 = r$per_era$post$C2$A_stableQ_share)
  }))
  db$y <- c(4, 3, 2, 1)[seq_len(nrow(db))]
  db$col <- ifelse(db$mode == "real", C_OBS, C_PLB)
  db$conf <- db$half == "holdout"
  xmax <- max(db$a0, db$a1)
  db$aseg <- db$a1 - 0.004 * sign(db$a1 - db$a0)
  pa <- ggplot(db) +
    geom_segment(aes(x = a0, xend = aseg, y = y, yend = y, colour = I(col)),
                 linewidth = 0.7, arrow = arrow(length = unit(4, "pt"), type = "closed")) +
    geom_point(aes(a0, y, colour = I(col)), size = 1.6) +
    geom_point(aes(a1, y, colour = I(col)), size = 1.6) +
    geom_text(aes(xmax + 0.02, y, colour = I(col),
                  fontface = ifelse(conf, "bold", "plain"),
                  label = sprintf("%+.3f", a1 - a0)),
              hjust = 0, size = ann_sz) +
    annotate("text", x = xmax + 0.02, y = 4.5, hjust = 0, size = ann_sz,
             colour = "grey25", parse = TRUE, label = 'Delta*italic(A)') +
    annotate("text", x = min(db$a0) - 0.015, y = 4.5, hjust = 0, size = ann_sz,
             colour = "grey40", label = "Bold = held-out (confirmatory half)") +
    scale_y_continuous(breaks = db$y, labels = db$lab,
                       limits = c(0.6, 4.65), expand = c(0, 0)) +
    scale_x_continuous(limits = c(min(db$a0) - 0.02, xmax + 0.06),
                       breaks = seq(0.25, 0.40, 0.05)) +
    labs(x = expression("Stable share "*italic(A)*"  (pre" %->% "post)"), y = NULL) +
    theme_j() +
    theme(axis.line.y = element_blank(), axis.ticks.y = element_blank(),
          axis.text.y = element_text(hjust = 1, size = 6,
                                     face = c("plain", "bold", "plain", "bold")))

  pb <- local({
    r <- pp_prim("real", PPHALF)
    prm <- do.call(rbind, lapply(phid$prepost_primary$real[[PPHALF]]$phi_pairs,
                                 function(p) c(num(p[[1]]), num(p[[2]]))))
    sp <- data.frame(x = prm[, 1], y = prm[, 2])
    lim <- c(-0.5, 2)
    ggplot(sp, aes(x, y)) +
      geom_hex(bins = 26) +
      geom_abline(colour = C_INK, linewidth = 0.3) +
      scale_fill_gradient(low = "#E4EDF4", high = C_OBS, trans = "log10") +
      annotate("text", x = lim[1], y = lim[2] - 0.1, hjust = 0, vjust = 1, size = ann_sz,
               colour = "grey25", lineheight = 1.3,
               label = paste0("Rank ρ = ", fmt(r$phi_rank_corr$rho, 2),
                              "\nPlacebo ρ = ",
                              fmt(pp_prim("placebo", PPHALF)$phi_rank_corr$rho, 2))) +
      coord_equal(xlim = lim, ylim = lim) +
      labs(x = expression(hat(Phi)*", pre-era"), y = expression(hat(Phi)*", AI era")) +
      theme_j()
  })

  (pa | pb) + plot_layout(widths = c(1.35, 1)) +
    plot_annotation(tag_levels = "a") &
    theme(plot.tag = element_text(face = "bold", size = 8))
})
save_fig(fig4, "fig4", 58)

fig4b_appendix <- local({
  r <- pp_prim("real", PPHALF)
  dd <- bind_rows(lapply(c("pre", "post"), function(e) {
    c1 <- r$per_era[[e]]$C1
    o <- num(unlist(c1$counts)); x <- num(unlist(c1$expected_counts))
    data.frame(era = e, d = 1:10, dep = o / sum(o) - x / sum(x),
               tvd = c1$tvd, p = c1$perm_p)
  })) |> mutate(era = factor(era, c("pre", "post"),
                             c("Pre-era", "AI era")))
  ann <- dd |> group_by(era) |> summarise(tvd = first(tvd), p = first(p), .groups = "drop")
  ggplot(dd, aes(d, dep)) +
    geom_hline(yintercept = 0, colour = C_INK, linewidth = 0.3) +
    geom_col(fill = C_OBS, width = 0.72) +
    geom_text(data = ann, aes(x = 5.5, y = 0.066,
                              label = paste0("TVD ", fmt(tvd, 2), ", P = ", fmt(p, 2))),
              size = 1.95, colour = "grey30") +
    facet_wrap(~era, nrow = 1) +
    scale_x_continuous(breaks = c(1, 5, 10)) +
    scale_y_continuous(limits = c(-0.08, 0.08),
                       labels = label_number(style_positive = "plus")) +
    labs(x = "Decile of era's biggest hit", y = "Observed − null") +
    theme_j() + theme(strip.text = element_text(size = 6.3, face = "bold"))
})
save_fig(fig4b_appendix, "figA3", 46, width_mm = 84, dir = SIDIR)
r_h <- pp_prim("real", "holdout"); p_h <- pp_prim("placebo", "holdout")
cat(sprintf("Fig 4: real hold dA=%+.3f placebo hold dA=%+.3f | rho=%s\n",
            r_h$per_era$post$C2$A_stableQ_share - r_h$per_era$pre$C2$A_stableQ_share,
            p_h$per_era$post$C2$A_stableQ_share - p_h$per_era$pre$C2$A_stableQ_share,
            fmt(r_h$phi_rank_corr$rho, 3)))

ckey <- function(r) paste(r$credit, r$W, r$n_min, r$proxy)
cell_df <- function(cs) bind_rows(lapply(cs, function(r) data.frame(
  key = ckey(r), primary = isTRUE(r$primary),
  dep = r$C1$mean_P - r$C1$null_mean_P, tvd = r$C1$tvd,
  A = r$C2$A_stableQ_share, ratio = r$C2$ratio_A_over_AB)))

figA4 <- local({
  hd <- cell_df(Filter(function(r) !isTRUE(r$skipped),
                       jread("estimates_holdout_full.json")$results))
  dd <- cell_df(cells)
  m <- merge(dd, hd, by = "key", suffixes = c("_d", "_h"))
  sq <- function(x, y, lab_x, lab_y, prim_col) {
    lim <- range(c(x, y)); pad <- diff(lim) * 0.06; lim <- lim + c(-pad, pad)
    df <- data.frame(x = x, y = y, primary = m$primary_d)
    ggplot(df, aes(x, y)) +
      geom_abline(colour = "grey65", linewidth = 0.3) +
      geom_point(colour = C_OBS, size = 0.95) +
      geom_point(data = subset(df, primary), shape = 21, size = 2.1, stroke = 0.7,
                 colour = C_INK, fill = NA) +
      annotate("text", x = lim[1], y = lim[2], hjust = 0, vjust = 1, size = ann_sz,
               colour = "grey25", label = sprintf("r = %s", fmt(cor(x, y), 2))) +
      coord_equal(xlim = lim, ylim = lim) +
      labs(x = lab_x, y = lab_y) + theme_j()
  }
  pa <- sq(m$dep_d, m$dep_h, "Timing departure, discovery", "Timing departure, held-out")
  pb <- sq(m$A_d, m$A_h, expression("Stable share "*italic(A)*", discovery"),
           expression("Stable share "*italic(A)*", held-out"))
  (pa | pb) + plot_annotation(tag_levels = "a") &
    theme(plot.tag = element_text(face = "bold", size = 8))
})
save_fig(figA4, "figA4", 100, dir = SIDIR)

figA7_cal <- local({
  dd <- cell_df(cells)
  sens <- bind_rows(
    cell_df(Filter(function(r) !isTRUE(r$skipped),
                   jread("estimates_sens_nohole_discovery.json")$results)) |>
      mutate(variant = "No-gap calendar (54 cells)"),
    cell_df(Filter(function(r) !isTRUE(r$skipped),
                   jread("estimates_sens_clean_discovery.json")$results)) |>
      mutate(variant = "Clean cadence, W = 60 (18 cells)"))
  m <- merge(dd, sens, by = "key", suffixes = c("_main", "_sens"))
  sq <- function(xv, yv, lab, gate = NULL) {
    df <- data.frame(x = m[[xv]], y = m[[yv]], v = m$variant,
                     primary = m$primary_main)
    lim <- range(c(df$x, df$y)); pad <- diff(lim) * 0.06; lim <- lim + c(-pad, pad)
    p <- ggplot(df, aes(x, y, colour = v)) +
      geom_abline(colour = "grey65", linewidth = 0.3)
    if (!is.null(gate))
      p <- p + geom_vline(xintercept = gate, colour = C_INK, linetype = "22",
                          linewidth = 0.3) +
        geom_hline(yintercept = gate, colour = C_INK, linetype = "22",
                   linewidth = 0.3)
    p + geom_point(size = 0.95) +
      geom_point(data = subset(df, primary), shape = 21, size = 2.1, stroke = 0.7,
                 colour = C_INK, fill = NA) +
      scale_colour_manual(values = c("No-gap calendar (54 cells)" = C_OBS,
                                     "Clean cadence, W = 60 (18 cells)" = C_NULL)) +
      coord_equal(xlim = lim, ylim = lim) +
      labs(x = paste(lab, "(main calendar)"), y = paste(lab, "(sensitivity)")) +
      theme_j() + theme(legend.position = c(0.98, 0.02),
                        legend.justification = c(1, 0),
                        legend.title = element_blank(),
                        legend.text = element_text(size = 5.6),
                        legend.key.size = unit(6, "pt"),
                        legend.background = element_blank())
  }
  pa <- sq("A_main", "A_sens", "Stable share A")
  pb <- sq("ratio_main", "ratio_sens", "Persistence ratio", gate = 0.20)
  (pa | pb) + plot_annotation(tag_levels = "a") &
    theme(plot.tag = element_text(face = "bold", size = 8))
})
save_fig(figA7_cal, "figA7", 98, dir = SIDIR)

figA5_pipe <- local({
  v <- read.csv(file.path(RESDIR, "si_v1_v2_comparison.csv"))
  sq <- function(xv, yv, lab) {
    df <- data.frame(x = v[[xv]], y = v[[yv]], primary = v$primary == "True")
    lim <- range(c(df$x, df$y)); pad <- diff(lim) * 0.06; lim <- lim + c(-pad, pad)
    ggplot(df, aes(x, y)) +
      geom_abline(colour = "grey65", linewidth = 0.3) +
      geom_point(colour = C_OBS, size = 0.95) +
      geom_point(data = subset(df, primary), shape = 21, size = 2.1, stroke = 0.7,
                 colour = C_INK, fill = NA) +
      annotate("text", x = lim[1], y = lim[2], hjust = 0, vjust = 1, size = ann_sz,
               colour = "grey25",
               label = sprintf("Median shift %+.3f", median(df$y - df$x))) +
      coord_equal(xlim = lim, ylim = lim) +
      labs(x = paste(lab, "(pipeline v1)"), y = paste(lab, "(pipeline v2)")) +
      theme_j()
  }
  pa <- sq("dep_v1", "dep_v2", "Timing departure")
  pb <- sq("A_v1", "A_v2", "Stable share A")
  (pa | pb) + plot_annotation(tag_levels = "a") &
    theme(plot.tag = element_text(face = "bold", size = 8))
})
save_fig(figA5_pipe, "figA5", 100, dir = SIDIR)

figA6_grid <- local({
  gd <- bind_rows(lapply(c("real", "placebo"), function(mo)
    bind_rows(lapply(c("discovery", "holdout"), function(h) {
      d <- pp[[paste(mo, h, sep = "_")]]; if (is.null(d)) return(NULL)
      bind_rows(lapply(Filter(function(r) !isTRUE(r$skipped), d$results), function(r)
        data.frame(mode = mo, half = h, W = r$W,
                   dA = r$per_era$post$C2$A_stableQ_share -
                        r$per_era$pre$C2$A_stableQ_share)))
    }))))
  gd$grp <- factor(paste0(ifelse(gd$mode == "real", "AI divide, ", "Placebo, "),
                          ifelse(gd$half == "holdout", "held-out", "discovery")),
                   c("AI divide, discovery", "AI divide, held-out",
                     "Placebo, discovery", "Placebo, held-out"))
  gd$Wlab <- factor(paste0("Horizon W = ", gd$W, " mo"))
  med <- gd |> group_by(Wlab, grp) |> summarise(dA = median(dA), .groups = "drop")
  ggplot(gd, aes(grp, dA, colour = ifelse(mode == "real", C_OBS, C_PLB))) +
    geom_hline(yintercept = 0, colour = C_INK, linewidth = 0.3) +
    geom_jitter(width = 0.12, height = 0, size = 0.8, alpha = 0.65,
                aes(colour = I(ifelse(mode == "real", C_OBS, C_PLB)))) +
    geom_crossbar(data = med, aes(ymin = dA, ymax = dA,
                                  colour = I(ifelse(grepl("AI", grp), C_OBS, C_PLB))),
                  width = 0.42, linewidth = 0.45, fatten = 0) +
    facet_wrap(~Wlab) +
    scale_y_continuous(labels = label_number(style_positive = "plus")) +
    labs(x = NULL, y = expression(Delta*italic(A)*"  (post − pre)")) +
    theme_j() +
    theme(strip.text = element_text(size = 6.3, face = "bold"),
          axis.text.x = element_text(angle = 30, hjust = 1))
})
save_fig(figA6_grid, "figA6", 60, dir = SIDIR)
cat("figures ->", OUTDIR, "\n")
