#!/usr/bin/env Rscript
# Chord diagram for neuron overlap between tasks.
# Reads CSV from run_overlap.py output.
#
# Usage:
#   Rscript scripts/plot_chord.R results/Qwen_Qwen2-5-32B-Instruct/overlap/positive_1.0pct_overlap.csv 10
#   Rscript scripts/plot_chord.R <csv_path> <cutoff_percent> [output_path]

# R_LIBS_USER=~/R/libs Rscript scripts/run_overlap_chord.R results/Qwen_Qwen2-5-32B-Instruct/overlap/positive_1.0pct_heads_overlap.csv 10



# R_LIBS_USER=~/R/libs Rscript scripts/run_overlap_chord.R results/mistralai_Mistral-Small-24B-Instruct-2501/overlap/positive_1.0pct_overlap_doubly_stochastic.csv 10


suppressPackageStartupMessages({
  library(circlize)
  library(dplyr)
})

# Try to load svglite for SVG output (works without Cairo)
have_svglite <- requireNamespace("svglite", quietly = TRUE)

# ============================================================
# Parse command line args
# ============================================================
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  cat("Usage: Rscript scripts/plot_chord.R <csv_path> <cutoff_percent> [output_path]\n")
  cat("Example: Rscript scripts/plot_chord.R results/.../positive_1.0pct_overlap.csv 8\n")
  quit(status = 1)
}

csv_file <- args[1]
OVERLAP_CUTOFF <- as.numeric(args[2])
WIDTH_TRANSFORM <- 'linear'

# Output path: user-specified or auto from CSV
if (length(args) >= 3) {
  output_file <- args[3]
} else {
  csv_dir <- dirname(csv_file)
  csv_base <- sub('\\.csv$', '', basename(csv_file))
  # Put in figures dir (sibling of overlap dir)
  figures_dir <- file.path(dirname(csv_dir), 'figures')
  dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)
  output_file <- file.path(figures_dir, sprintf('%s_chord_%gpct.svg', csv_base, OVERLAP_CUTOFF))
}

cat(sprintf("CSV: %s\n", csv_file))
cat(sprintf("Cutoff: %g%%\n", OVERLAP_CUTOFF))
cat(sprintf("Output: %s\n", output_file))

# ============================================================
# Extract model info from CSV filename
# ============================================================
extract_model_info <- function(csv_path) {
  filename <- basename(csv_path)
  filename <- sub('\\.csv$', '', filename)

  # Our pipeline format: {sign}_{pct}pct_overlap.csv
  # e.g. positive_1.0pct_overlap
  pct_match <- regmatches(filename, regexpr('[0-9.]+pct', filename))
  pct <- if (length(pct_match) > 0) pct_match else 'unknown'

  # Model name from parent dirs: results/{model_short}/overlap/
  parts <- strsplit(csv_path, .Platform$file.sep)[[1]]
  overlap_idx <- which(parts == 'overlap')
  model_name <- if (length(overlap_idx) > 0 && overlap_idx > 1) parts[overlap_idx - 1] else 'unknown'

  return(list(model_name = model_name, pct = pct))
}

model_info <- extract_model_info(csv_file)
model_name <- model_info$model_name
pct <- model_info$pct
cat(sprintf("Model: %s\n", model_name))
cat(sprintf("Pct: %s\n", pct))

# ============================================================
# Domain mapping — parse "domain/task" labels from CSV
# ============================================================
parse_domain <- function(label) {
  parts <- strsplit(label, '/', fixed = TRUE)[[1]]
  if (length(parts) >= 2) return(parts[1])
  return('Other')
}

parse_task <- function(label) {
  parts <- strsplit(label, '/', fixed = TRUE)[[1]]
  if (length(parts) >= 2) return(parts[2])
  return(label)
}

# Map pipeline domain codes to display names
domain_display <- c(
  'Lan' = 'Language',
  'MD' = 'Formal',
  'phys' = 'Physical',
  'ToM' = 'Social'
)

# ============================================================
# OLD subdomain mapping (commented out, kept for reference)
# ============================================================
# task_to_subdomain <- function(domain, task) {
#   if (domain == 'MD') {
#     if (task %in% c('add_sub_2op_symbolic', 'add_sub_2op_verbal',
#                      'add_sub_3op_symbolic', 'add_sub_3op_verbal',
#                      'mul_div_2op_symbolic', 'mul_div_2op_verbal',
#                      'mul_div_3op_symbolic', 'mul_div_3op_verbal',
#                      'simple_equation')) return('MD_Arithmetic')
#     if (grepl('logic', task)) return('MD_Logic')
#     return('MD_Code')
#   }
#   if (domain == 'Lan') return('Language')
#   if (domain == 'phys') return('Physical')
#   if (domain == 'ToM') return('ToM')
#   return('Other')
# }

# ============================================================
# NEW subdomain mapping
# MD splits into 3 sub-shades: Arithmetic / Logic / Code+Algorithmic
# - simple_equation belongs to Arithmetic (it's solving linear equations)
# - number_sequence and number_sorting are merged with Code (algorithmic)
# ============================================================
task_to_subdomain <- function(domain, task) {
  if (domain == 'MD') {
    arithmetic_tasks <- c('add_sub_2op_symbolic', 'add_sub_2op_verbal',
                          'add_sub_3op_symbolic', 'add_sub_3op_verbal',
                          'mul_div_2op_symbolic', 'mul_div_2op_verbal',
                          'mul_div_3op_symbolic', 'mul_div_3op_verbal',
                          'simple_equation')
    if (task %in% arithmetic_tasks) return('MD_Arithmetic')
    if (grepl('logic', task)) return('MD_Logic')
    # Code + Algorithmic (number_sequence, number_sorting) merged
    return('MD_CodeAlgo')
  }
  if (domain == 'Lan') return('Language')
  if (domain == 'phys') return('Physical')
  if (domain == 'ToM') return('ToM')
  return('Other')
}

# ============================================================
# OLD short task names (commented out, kept for reference)
# ============================================================
# task_short_names <- c(
#   'sentence_vs_non-words_1' = 'Lang1', 'sentence_vs_non-words_2' = 'Lang2',
#   'anaphor_gender_agreement' = 'Anaphor', 'passive_2' = 'Passive',
#   'principle_A_case_1' = 'PrincA', 'existential_there_subject_raising' = 'Exist',
#   'determiner_noun_agreement1' = 'Det-N', 'ellipsis_n_bar_2' = 'Ellipsis',
#   'wh_vs_that_no_gap' = 'WH-That', 'irregular_past_participle_adjectives' = 'Irreg',
#   'coordinate_structure_constraint_object_extraction' = 'Coord',
#   'sentential_negation_npi_licensor_present' = 'Neg',
#   'existential_there_quantifiers_1' = 'Quant',
#   'regular_plural_subject_verb_agreement_2' = 'S-V',
#   'blimp_group1_agreement' = 'G1', 'blimp_group2_depen' = 'G2', 'blimp_group3_interp' = 'G3',
#   'det_noun_agreement_irregular' = 'D-N_Irr',
#   'det_noun_agreement_regular' = 'D-N_Reg',
#   'hypernymy' = 'Hyper',
#   'npi' = 'NPI',
#   'subject_verb_agreement' = 'S-V',
#   'wug' = 'Wug',
#   'add_sub_2op_symbolic' = 'A+2S', 'add_sub_2op_verbal' = 'A+2V',
#   'add_sub_3op_symbolic' = 'A+3S', 'add_sub_3op_verbal' = 'A+3V',
#   'mul_div_2op_symbolic' = 'M*2S', 'mul_div_2op_verbal' = 'M*2V',
#   'mul_div_3op_symbolic' = 'M*3S', 'mul_div_3op_verbal' = 'M*3V',
#   'simple_equation' = 'Eq',
#   'logic_first_order' = 'FOL', 'logic_folio' = 'FOLIO',
#   'logic_propositional' = 'PropL', 'logic_syllogism' = 'Syllog',
#   'mbpp_assertion' = 'MBPP',
#   'number_sorting' = 'NumS', 'number_sequence' = 'NumQ',
#   'modulo_grid' = 'Mod',
#   'code_A' = "code_A",
#   'code_B' = 'code_B',
#   'code_conditional' = 'CodeCond',
#   'code_list' = 'CodeList',
#   'code_loop' = 'CodeLoop',
#   'logic_propositional_1' = 'PropL1',
#   'logic_propositional_symbolic' = 'PropLS',
#   'logic_syllogism_1' = 'Syll1',
#   'logic_syllogism_symbolic' = 'SyllS',
#   'physical_reasoning_newton' = 'Newton', 'physical_reasoning_prost' = 'PROST',
#   'physical_material_behav' = 'Material', 'physical_obj_motion_force' = 'Motion',
#   'physical_spatial_relational' = 'Spatial',
#   'phys_newton' = 'newton', 'phys_prost' = 'prost',
#   'physical' = 'physical', 'material' = 'material', 'spatial' = 'spatial',
#   'physics_standard' = 'standard', 'physics_complex' = 'complex',
#   'physics_buoyancy' = 'buoyancy', 'physics_brightness' = 'brightness',
#   'physics_elasticity' = 'elasticity', 'physics_solubility' = 'solubility',
#   'physics_speed' = 'speed', 'physics_stability' = 'stability',
#   'physics_temperature' = 'temperature',
#   'ToM_bigtom' = 'BigToM', 'ToM_Faux' = 'Faux',
#   'ToM_Socialqa' = 'SocQA', 'ToM_agent' = 'Agent',
#   'ToM_social_interactions' = 'SocInt', 'ToM_social_relations' = 'SocRel',
#   'socialqa' = 'SocialQA',
#   'agent' = 'Agent',
#   'social_interactions' = 'SocInteract',
#   'social_relations' = 'SocRelation',
#   'desires_goals' = 'Desires',
#   'primary_emotions' = 'PrimEmotion',
#   'secondary_emotions' = 'SecEmotion',
#   'emotion_fewshot' = 'EmoFewshot',
#   'norm_appropriate' = 'Appropriate',
#   'norm_moral' = 'Moral'
# )

# ============================================================
# NEW canonical task display names (per docs/figure.md)
# ============================================================
task_short_names <- c(
  # Language (8)
  'anaphor_gender_agreement' = 'Anaphor',
  'det_noun_agreement_irregular' = 'DetN-Irr',
  'det_noun_agreement_regular' = 'DetN-Reg',
  'det_noun_agreement_with_adjective' = 'DetN-Adj',
  'hypernymy' = 'Hyper',
  'npi' = 'NPI',
  'subject_verb_agreement' = 'S-V',
  'wug' = 'Wug',

  # MD - Arithmetic (8)
  'add_sub_2op_symbolic' = 'Add2-Sym',
  'add_sub_2op_verbal' = 'Add2-Vrb',
  'add_sub_3op_symbolic' = 'Add3-Sym',
  'add_sub_3op_verbal' = 'Add3-Vrb',
  'mul_div_2op_symbolic' = 'Mul2-Sym',
  'mul_div_2op_verbal' = 'Mul2-Vrb',
  'mul_div_3op_symbolic' = 'Mul3-Sym',
  'mul_div_3op_verbal' = 'Mul3-Vrb',

  # MD - Logic (4)
  'logic_propositional_1' = 'PropL-NL',
  'logic_propositional_symbolic' = 'PropL-Sym',
  'logic_syllogism_1' = 'Syll-NL',
  'logic_syllogism_symbolic' = 'Syll-Sym',

  # MD - Code (5)
  'code_A' = 'CodeA',
  'code_B' = 'CodeB',
  'code_conditional' = 'CodeCond',
  'code_list' = 'CodeList',
  'code_loop' = 'CodeLoop',

  # MD - Algorithmic (3) — merged with Code shade
  'simple_equation' = 'Eq',
  'number_sequence' = 'NumSeq',
  'number_sorting' = 'NumSort',

  # Physical (9)
  'phys_newton' = 'Newton',
  'phys_prost' = 'PROST',
  'physics_brightness' = 'Brightness',
  'physics_buoyancy' = 'Buoyancy',
  'physics_elasticity' = 'Elasticity',
  'physics_solubility' = 'Solubility',
  'physics_speed' = 'Speed',
  'physics_stability' = 'Stability',
  'physics_temperature' = 'Temperature',

  # ToM (9)
  'agent' = 'Agent',
  'desires_goals' = 'Desires',
  'emotion_fewshot' = 'EmotionFS',
  'norm_appropriate' = 'NormApp',
  'norm_moral' = 'NormMoral',
  'primary_emotions' = 'PrimEmo',
  'secondary_emotions' = 'SecEmo',
  'social_interactions' = 'SocInt',
  'social_relations' = 'SocRel'
)

# ============================================================
# OLD domain colors (commented out, kept for reference)
# ============================================================
# domain_colors <- c(
#   'Language' = '#2ECC71',
#   'MD_Arithmetic' = '#E74C3C',
#   'MD_Logic' = '#E67E22',
#   'MD_Code' = '#F1C40F',
#   'Physical' = '#3498DB',
#   'ToM' = '#9B59B6'
# )

# ============================================================
# NEW domain colors (lab convention from Andrea)
# Language = red, MD = blue (3 shades), ToM = green, Physical = orange
# Three blues spaced for visual contrast: very dark / medium / light
# ============================================================
domain_colors <- c(
  'Language'      = '#C0392B',  # red
  'MD_Arithmetic' = '#0E2F44',  # very dark blue (almost navy)
  'MD_Logic'      = '#85C1E9',  # light blue
  'MD_CodeAlgo'   = '#3498DB',  # bright medium blue
  'Physical'      = '#E67E22',  # orange
  'ToM'           = '#27AE60'   # green
)

# ============================================================
# Load data
# ============================================================
cat("Loading overlap data...\n")
if (!file.exists(csv_file)) {
  stop(sprintf("CSV file not found: %s", csv_file))
}
df <- read.csv(csv_file, row.names = 1, check.names = FALSE)
cat(sprintf("Loaded %g tasks\n", nrow(df)))

# Convert to percentage if in ratio form (0-1)
if (max(df, na.rm = TRUE) <= 1.0) {
  df <- df * 100
  cat("Converted ratios to percentages\n")
}

# ============================================================
# Parse labels — our CSV has "domain/task" format
# ============================================================
raw_labels <- rownames(df)
task_domains <- sapply(raw_labels, parse_domain)
task_names <- sapply(raw_labels, parse_task)
task_subdomains <- mapply(task_to_subdomain, task_domains, task_names)

# Get short names
all_short_names <- sapply(task_names, function(t) {
  if (t %in% names(task_short_names)) return(task_short_names[t])
  if (nchar(t) > 12) return(paste0(substr(t, 1, 10), '..'))
  return(t)
})

# Deduplicate short names by appending _1, _2, etc.
name_counts <- table(all_short_names)
dup_names <- names(name_counts[name_counts > 1])
counters <- setNames(rep(1, length(dup_names)), dup_names)
for (i in seq_along(all_short_names)) {
  nm <- all_short_names[i]
  if (nm %in% dup_names) {
    all_short_names[i] <- paste0(nm, '_', counters[nm])
    counters[nm] <- counters[nm] + 1
  }
}

# Order by subdomain for visualization
# OLD: subdomain_order <- c('Language', 'MD_Arithmetic', 'MD_Logic', 'MD_Code', 'Physical', 'ToM')
subdomain_order <- c('Language', 'MD_Arithmetic', 'MD_Logic', 'MD_CodeAlgo', 'Physical', 'ToM')
order_idx <- order(match(task_subdomains, subdomain_order), task_names)

raw_labels_ordered <- raw_labels[order_idx]
short_names_ordered <- all_short_names[order_idx]
subdomains_ordered <- task_subdomains[order_idx]

# Reorder the data frame
df <- df[raw_labels_ordered, raw_labels_ordered]

# ============================================================
# Build connections (above threshold)
# ============================================================
connections <- data.frame()
for (i in 1:(length(raw_labels_ordered) - 1)) {
  for (j in (i + 1):length(raw_labels_ordered)) {
    label1 <- raw_labels_ordered[i]
    label2 <- raw_labels_ordered[j]
    value <- df[label1, label2]
    if (!is.na(value) && value >= OVERLAP_CUTOFF) {
      connections <- rbind(connections, data.frame(
        from = short_names_ordered[i],
        to = short_names_ordered[j],
        value = value,
        stringsAsFactors = FALSE
      ))
    }
  }
}

cat(sprintf("Found %g connections above %g%% overlap\n", nrow(connections), OVERLAP_CUTOFF))
cat(sprintf("Plotting %g tasks\n", length(raw_labels_ordered)))

# ============================================================
# Set colors
# ============================================================
task_colors <- sapply(subdomains_ordered, function(sd) domain_colors[sd])
names(task_colors) <- short_names_ordered

# Width transform
if (nrow(connections) > 0) {
  vals <- connections$value
  if (WIDTH_TRANSFORM == 'sqrt') {
    link_lwd <- 0.5 + 5 * sqrt(vals / max(vals))
  } else if (WIDTH_TRANSFORM == 'log') {
    link_lwd <- 0.5 + 5 * log1p(vals) / log1p(max(vals))
  } else {
    link_lwd <- 0.5 + 5 * (vals / max(vals))
  }
} else {
  link_lwd <- numeric(0)
}

# ============================================================
# Group vector for domain spacing
# ============================================================
group_vector <- sapply(subdomains_ordered, function(sd) {
  gsub('_.*', '', sd)  # MD_Arithmetic -> MD, etc.
})
names(group_vector) <- short_names_ordered

# ============================================================
# Plot
# ============================================================
dir.create(dirname(output_file), recursive = TRUE, showWarnings = FALSE)
# OLD: PDF only
# output_file <- sub('\\.png$', '.pdf', output_file)
# pdf(output_file, width = 15, height = 15)
# NEW: save as SVG (per Andrea's request)
# Use svglite if available (Cairo-free), else fall back to base svg(), else PDF
output_file <- sub('\\.(png|pdf)$', '.svg', output_file)
if (have_svglite) {
  svglite::svglite(output_file, width = 15, height = 15)
} else {
  tryCatch({
    svg(output_file, width = 15, height = 15)
  }, error = function(e) {
    cat("Warning: SVG not available, falling back to PDF\n")
    output_file <<- sub('\\.svg$', '.pdf', output_file)
    pdf(output_file, width = 15, height = 15)
  })
}

circos.clear()

circos.par(
  start.degree = 90,
  track.margin = c(0.01, 0.01),
  points.overflow.warning = FALSE,
  canvas.xlim = c(-1.06, 1.06),
  canvas.ylim = c(-1.12, 1.0)
)

chordDiagram(
  connections,
  grid.col = task_colors,
  order = short_names_ordered,
  group = group_vector,
  transparency = 0.5,
  directional = 0,
  annotationTrack = "grid",
  annotationTrackHeight = 0.02,
  preAllocateTracks = list(track.height = 0.25),
  link.border = 1,
  link.lwd = link_lwd,
  big.gap = 10,
  small.gap = 2
)

# Task labels
circos.track(
  track.index = 1,
  panel.fun = function(x, y) {
    xlim = get.cell.meta.data("xlim")
    ylim = get.cell.meta.data("ylim")
    sector.name = get.cell.meta.data("sector.index")
    circos.text(
      mean(xlim), ylim[1],
      sector.name,
      facing = "clockwise",
      niceFacing = TRUE,
      adj = c(0, 0.5),
      cex = 1.8,
      font = 2
    )
  },
  bg.border = NA
)

# Detect component type from filename
csv_basename <- basename(csv_file)
component_label <- if (grepl('_heads_', csv_basename)) 'Head' else 'Neuron'

# Title
# OLD: sprintf("%s\n%s Overlap (>=%g%%)", model_name, component_label, OVERLAP_CUTOFF)
# Try to extract pct from CSV name
pct_match <- regmatches(csv_basename, regexpr('[0-9.]+pct', csv_basename))
pct_str <- if (length(pct_match) > 0) sub('pct', '', pct_match) else 'unknown'
display_model <- if (model_name == 'average') '6-model average' else model_name
title_str <- sprintf("Cross-task %s overlap (%s, top %s%%, \u2265%g%%)",
                     tolower(component_label), display_model, pct_str, OVERLAP_CUTOFF)
title(title_str, cex.main = 1.8, font.main = 2, line = -3)

# Legend — only show domains that are present
# OLD labels:
# present_labels <- c(
#   'Language' = 'Language', 'MD_Arithmetic' = 'MD (Arithmetic)',
#   'MD_Logic' = 'MD (Logic)', 'MD_Code' = 'MD (Code)',
#   'Physical' = 'Physical', 'ToM' = 'ToM'
# )
present_subdomains <- unique(subdomains_ordered)
present_labels <- c(
  'Language'      = 'Lan',
  'MD_Arithmetic' = 'Formal (Arith)',
  'MD_Logic'      = 'Formal (Logic)',
  'MD_CodeAlgo'   = 'Formal (Code/Algo)',
  'Physical'      = 'Phys',
  'ToM'           = 'Social'
)
legend_idx <- present_subdomains[present_subdomains %in% names(domain_colors)]
legend("bottomleft",
       legend = present_labels[legend_idx],
       fill = domain_colors[legend_idx],
       border = "white", bty = "n", cex = 2.5)

circos.clear()
dev.off()

cat(sprintf("\n✓ Saved: %s\n", output_file))