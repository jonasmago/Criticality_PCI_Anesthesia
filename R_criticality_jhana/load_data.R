library(data.table)
library(dplyr)
library(purrr)


##########################
### LOAD RESULTS ###
##########################


# Function to read CSV and drop numeric column names
read_and_select <- function(path) {
  dt <- fread(path)
  
  # Drop columns named 0 to 31
  drop_cols <- as.character(0:31)
  keep_cols <- setdiff(names(dt), drop_cols)
  dt <- dt[, ..keep_cols]
  
  # Extract file identifier (e.g. EOS_8_13 from path)
  file_id <- tools::file_path_sans_ext(basename(path))  # e.g. "EOS_8_13"
  
  # Rename columns except for sub, day, condition
  cols_to_rename <- setdiff(names(dt), c("sub", "day", "condition"))
  setnames(dt, cols_to_rename, paste0(file_id, "_", cols_to_rename))
  
  return(dt)
}


# File paths
file_paths <- c(
  "../data/output/AVC/AVC_bin_2.0_iei_0.008.csv",
  # "../data/output/avc_std_dist/AVC_std_varley_raw.csv",
  # "../data/output/avc_std_dist/AVC_std_varley.csv",
  "../data/output/DFA/DFA_1_4.csv",
  "../data/output/DFA/DFA_1_45.csv",
  "../data/output/DFA/DFA_4_8.csv",
  "../data/output/DFA/DFA_8_13.csv",
  "../data/output/DFA/DFA_13_30.csv",
  "../data/output/DFA/DFA_30_44.csv",
  "../data/output/EOC/01Chaos_fixed_4.csv",
  # "../data/output/EOC/K_space_fixed_4.csv",
  "../data/output/EOS/EOS_1_4.csv",
  "../data/output/EOS/EOS_1_45.csv",
  "../data/output/EOS/EOS_4_8.csv",
  "../data/output/EOS/EOS_8_13.csv",
  "../data/output/EOS/EOS_13_30.csv",
  "../data/output/EOS/EOS_30_45.csv",
  # "../data/output/Slope/Slope_space.csv",
  "../data/output/Slope/Slope.csv"
)

# Load & clean each file
data_list <- lapply(file_paths, read_and_select)

for (i in seq_along(data_list)) {
  cols <- names(data_list[[i]])
  if (!("sub" %in% cols && "day" %in% cols)) {
    cat("❌ File", file_paths[i], "is missing `sub` or `day`\n")
    print(cols)
  } else {
    cat("✅ File", file_paths[i], "looks good\n")
  }
}


# Merge on sub + day
merged_data <- reduce(data_list, full_join, by = c("sub", "day", "condition"))


# Done!
print(dim(merged_data))
head(merged_data)

# Optional: Save
# fwrite(merged_data, "merged_clean.csv")








##########################
### LOAD MAIN ###
##########################


# Load the Excel sheet
phen_data <- read_excel("master_phen_dat.xlsx", sheet = "main")

# Filter only jhana and mindfulness
phen_data <- phen_data %>%
  filter(type %in% c("jhana", "mindfulness"))

# Define MODTAS and word columns to drop
modtas_cols <- paste0("MODTAS", 1:34)
word_cols   <- paste0("word", 1:10)

# Combine and intersect with actual column names
cols_to_drop <- intersect(names(phen_data), c(modtas_cols, word_cols))

# Drop them
phen_data <- phen_data %>%
  select(-all_of(cols_to_drop))


# Create matching keys
phen_data <- phen_data %>%
  mutate(
    sub = paste0("sub", ID),
    day = paste0("day", run),
    condition = type
  )

phen_data <- phen_data %>%
  group_by(sub) %>%
  mutate(MODTAS_sum_rep = MODTAS_sum[day == "day1" & condition == "jhana"]) %>%
  ungroup()


phen_data <- phen_data %>%
  mutate(
    stability_jhana_mindfulness = case_when(
      condition == "jhana" ~ rowMeans(
        mutate(across(starts_with("jhana_stabilit_j"), as.numeric)) %>%
          select(starts_with("jhana_stabilit_j")),
        na.rm = TRUE
      ),
      condition == "mindfulness" ~ as.numeric(mindfulness_stability_before_beeps),
      TRUE ~ NA_real_
    ),
    
    fading_jhana_mindfulness = case_when(
      condition == "jhana" ~ rowMeans(
        mutate(across(starts_with("jhana_fading_j"), as.numeric)) %>%
          select(starts_with("jhana_fading_j")),
        na.rm = TRUE
      ),
      condition == "mindfulness" ~ as.numeric(mindfulness_fading_before_beeps),
      TRUE ~ NA_real_
    )
  )



# Merge with EEG data
merged_full <- full_join(merged_data, phen_data, by = c("sub", "day", "condition"))

# Done!
print(dim(merged_full))
head(merged_full)

rm(data_list, phen_data, merged_data, cols, cols_to_drop, file_paths, i, modtas_cols, word_cols, read_and_select)

cat(names(merged_full), sep = "\n")
# outcomes_all <- names(merged_full)[4:(length(names(merged_full)) - 36)]
# outcomes_all <- outcomes_all[sapply(merged_full[, ..outcomes_all], is.numeric)]


