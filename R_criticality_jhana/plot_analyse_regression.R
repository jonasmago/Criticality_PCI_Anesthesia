library(dplyr)
library(ggplot2)
library(lme4)
library(lmerTest)
library(glue)


####################
### REGRESSION ###
####################

# Define predictors and outcomes
predictors <- c("stability_jhana_mindfulness", "fading_jhana_mindfulness", "MODTAS_sum_rep", "dat")
# outcomes   <- c("AVC_bin_2.0_iei_0.008_mean_iei", "EOS_8_13_OR_mean", "EOS_1_4_OR_mean")
# outcomes   <- outcomes_all

outcomes   <- c("01Chaos_fixed_4_K_median", "AVC_bin_2.0_iei_0.008_mean_iei")


vars_to_process <- c(predictors, outcomes)

# Trimming function (winsorize to ±2.5 SD)
winsorize <- function(x, sd_limit = 2.5) {
  if (all(is.na(x))) return(x)
  mu <- mean(x, na.rm = TRUE)
  sigma <- sd(x, na.rm = TRUE)
  lower <- mu - sd_limit * sigma
  upper <- mu + sd_limit * sigma
  x[x < lower] <- lower
  x[x > upper] <- upper
  return(x)
}

# Apply trimming and z-scoring
merged_scaled <- merged_full %>%
  mutate(across(
    all_of(vars_to_process),
    ~ scale(winsorize(.))[, 1],
    .names = "z_{.col}"
  ))

# Z-scored variable names
z_predictors <- paste0("z_", predictors)
z_outcomes   <- paste0("z_", outcomes)

# Loop over models and plots
for (outcome in outcomes) {
  for (predictor in predictors) {
    cat("🔍 Model:", outcome, "~", predictor, "\n")
    
    formula <- as.formula(paste(outcome, "~", predictor, "+ (1 | sub)"))
    model <- lmer(formula, data = merged_scaled)
    summ <- summary(model)
    
    est   <- signif(summ$coefficients[2, "Estimate"], 3)
    pval  <- signif(summ$coefficients[2, "Pr(>|t|)"], 3)
    tval  <- signif(summ$coefficients[2, "t value"], 3)
    label <- glue("slope = {est}, t = {tval}, p = {pval}")
    
    # Dynamically calculate label y-position (top of y range)
    y_max <- max(merged_scaled[[outcome]], na.rm = TRUE)
    
    # Plot with label at top and subject-colored dots
    print(
      ggplot(merged_scaled, aes_string(x = predictor, y = outcome, color = "sub")) +
        geom_point(alpha = 0.7, size = 2) +
        geom_smooth(method = "lm", se = TRUE, color = "black") +
        annotate("text", x = -1.8, y = 2.5, label = label, hjust = 0, size = 4) +
        labs(
          title = glue("{outcome} vs {predictor}"),
          x = predictor,
          y = outcome
        ) +
        theme_minimal() +
        theme(legend.position = "none")
    )
    
    
    cat("\n-----------------------------\n\n")
  }
}



########################################
### HIERARCICAL LINEAR MODEL ###
########################################

library(dplyr)
library(ggplot2)
library(lme4)
library(lmerTest)
library(glue)


# Loop over outcomes
for (outcome in outcomes) {
  cat("🔍 Group comparison for", outcome, "by condition and day\n")
  
  # Check if the outcome column has all NaN values
  if (all(is.nan(merged_scaled[[outcome]]))) {
    cat("Skipping plot for", outcome, "because all values are NaN.\n")
    next  # Skip to the next iteration of the loop
  }
  
  # Compute mean across days
  mean_data <- merged_scaled %>%
    group_by(sub, condition) %>%
    summarize(!!outcome := mean(.data[[outcome]], na.rm = TRUE), .groups = "drop") %>%
    mutate(day = "mean")
  
  # Combine with original data
  plot_data <- merged_scaled %>%
    select(sub, condition, day, all_of(outcome)) %>%
    bind_rows(mean_data)
  
  # Normalize day as factor with ordering
  plot_data$day <- factor(plot_data$day, levels = c("day1", "day2", "day3", "day4", "mean"))
  
  # Mixed model on full data (to annotate stats)
  model <- lmer(as.formula(paste(outcome, "~ condition + (1 | sub)")), data = merged_scaled)
  summ <- summary(model)
  est  <- signif(summ$coefficients["conditionmindfulness", "Estimate"], 3)
  tval <- signif(summ$coefficients["conditionmindfulness", "t value"], 3)
  pval <- signif(summ$coefficients["conditionmindfulness", "Pr(>|t|)"], 3)
  label <- glue("p = {pval}\nt = {tval}\nΔ = {est}")
  
  # Plot
  y_max <- max(plot_data[[outcome]], na.rm = TRUE)
  
  # Filter out NaN values for the outcome variable only for plotting (but keep original data intact)
  plot_data_filtered <- plot_data %>%
    filter(!is.nan(.data[[outcome]]))
  
  # Plot the violin plot with the filtered data (ignoring NaN for the plot)
  print(
    ggplot(plot_data_filtered, aes(x = condition, y = .data[[outcome]], fill = condition)) +
      geom_violin(trim = FALSE, alpha = 0.6, width = 0.9) +
      geom_point(position = position_jitter(width = 0.1), alpha = 0.6, size = 2) +
      geom_line(aes(group = sub), color = "gray50", alpha = 0.5, position = position_dodge(width = 0.3)) +
      stat_summary(fun = mean, geom = "point", shape = 21, size = 3, fill = "white") +
      annotate("text", x = 1.2, y = y_max * 1.1, label = label, hjust = 0, size = 4) +
      facet_wrap(~ day, nrow = 1) +
      labs(
        title = glue("{outcome} by condition across days + mean"),
        x = "Condition",
        y = outcome
      ) +
      theme_minimal() +
      theme(legend.position = "none")
  )
  
  cat("\n-----------------------------\n\n")
}
