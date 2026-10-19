# =========================================================================
# Plot Spatialization and Tree Summary Workflow
# =========================================================================
#
# This script:
#   1. Reads a CSV containing plot coordinates
#   2. Converts the plot coordinates to a spatial point layer
#   3. Reprojects the points to a specified output CRS
#   4. Saves the plot locations as a shapefile
#   5. Reads a CSV containing individual tree measurements
#   6. Calculates individual-tree biomass and volume
#   7. Summarises DBH, height, tree count, density, biomass, and volume by plot
#   8. Joins the tree summaries to the spatial plot layer
#   9. Saves the final shapefile containing the plot summaries
#
# =========================================================================


# =========================================================================
# Install required packages (if not already done)
# =========================================================================

pkgs <- c(
  "sf",
  "dplyr",
  "readr"
)

# Install required packages if not already done
for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) {
    install.packages(p)
  }
}


# Clear and Setup Environment
rm(list = ls(globalenv()))


# Load required packages ---------------------------------------------------

library(sf)       # Spatial vector data and coordinate reference systems
library(dplyr)    # Data manipulation and summarising
library(readr)    # Reading CSV files


# =========================================================================
# VOLUME AND BIOMASS COEFFICIENTS
# =========================================================================


# -------------------------------------------------------------------------
# Volume coefficients
# -------------------------------------------------------------------------
#
# Volume equation:
#
# Volume = b0 + b1 * DBH^b2 * Height^b3
#

volume_coeffs <- list(

  Sw = c(
    b0 = 0.04316,
    b1 = 3.1526e-5,
    b2 = 2,
    b3 = 1
  ),

  Sb = c(
    b0 = 0.00432,
    b1 = 3.5718e-5,
    b2 = 2,
    b3 = 1
  ),

  At = c(
    b0 = 0.04591,
    b1 = 3.1133e-5,
    b2 = 2,
    b3 = 1
  ),

  Pl = c(
    b0 = 0,
    b1 = 4.421585e-5,
    b2 = 1.926909,
    b3 = 1.00304
  ),

  Bw = c(
    b0 = 0,
    b1 = 5.634793e-5,
    b2 = 1.976455,
    b3 = 0.803794
  ),

  All_softwoods = c(
    b0 = 0,
    b1 = 4.5e-5,
    b2 = 1.95,
    b3 = 1
  ),

  All_hardwoods = c(
    b0 = 0,
    b1 = 5.5e-5,
    b2 = 1.95,
    b3 = 0.9
  )
)


# -------------------------------------------------------------------------
# Biomass coefficients
# -------------------------------------------------------------------------
#
# Biomass equation:
#
# Biomass = b1 * DBH^b2 * Height^b3
#
# Wood, bark, branch, and foliage biomass are calculated separately
# and then summed to obtain total above-ground biomass.
#

biomass_coeffs <- list(

  Pl = list(
    wood     = c(0.0239, 1.6827, 1.1878),
    bark     = c(0.0117, 1.6398, 0.6524),
    branches = c(0.0285, 3.3764, -1.4395),
    foliage  = c(0.0769, 2.6834, -1.2484)
  ),

  Sw = list(
    wood     = c(0.0252, 1.7819, 1.0022),
    bark     = c(0.0096, 1.6901, 0.7393),
    branches = c(0.0322, 2.8961, -0.9203),
    foliage  = c(0.1832, 2.4144, -1.0948)
  ),

  Sb = list(
    wood     = c(0.0335, 1.7389, 0.9835),
    bark     = c(0.0132, 1.7657, 0.5775),
    branches = c(0.0405, 3.1917, -1.3674),
    foliage  = c(0.2078, 2.5517, -1.3453)
  ),

  At = list(
    wood     = c(0.0143, 1.9369, 1.0579),
    bark     = c(0.0063, 2.0744, 0.6691),
    branches = c(0.0150, 2.9068, -0.6306),
    foliage  = c(0.0284, 1.6020, 0.0)
  ),

  Bw = list(
    wood     = c(0.0333, 2.0794, 0.6811),
    bark     = c(0.0079, 1.9905, 0.6553),
    branches = c(0.0253, 3.1518, -0.9083),
    foliage  = c(0.1361, 2.2978, -1.0934)
  ),

  Subalpine_fir = list(
    wood     = c(0.0220, 1.6469, 1.1714),
    bark     = c(0.0061, 1.8603, 0.7693),
    branches = c(0.0265, 3.6747, -1.5958),
    foliage  = c(0.0509, 2.9909, -1.2271)
  ),

  Red_alder_cottonwood = list(
    wood     = c(0.0051, 1.0697, 2.2748),
    bark     = c(0.0009, 1.3061, 2.0109),
    branches = c(0.0131, 2.5760, 0.0),
    foliage  = c(0.0224, 1.8368, 0.0)
  ),

  All_softwoods = list(
    wood     = c(0.0276, 1.6868, 1.0953),
    bark     = c(0.0101, 1.8486, 0.5525),
    branches = c(0.0313, 2.9974, -1.0383),
    foliage  = c(0.1379, 2.3981, -1.0418)
  ),

  All_hardwoods = list(
    wood     = c(0.0353, 2.0249, 0.7048),
    bark     = c(0.0090, 1.8677, 0.7144),
    branches = c(0.0448, 2.6855, -0.5911),
    foliage  = c(0.0869, 1.8541, -0.5491)
  )
)


# =========================================================================
# FUNCTION 1: Calculate biomass for one tree
# =========================================================================

calc_agb <- function(species, dbh, height) {

  # Return NA if species or tree measurements are invalid
  if (
    is.na(species) ||
    is.na(dbh) ||
    is.na(height) ||
    dbh <= 0 ||
    height <= 0 ||
    !(species %in% names(biomass_coeffs))
  ) {
    return(NA_real_)
  }

  # Get biomass coefficients for the specified species
  species_coeffs <- biomass_coeffs[[species]]

  # Calculate biomass for each tree component and sum the components
  agb <- sum(
    vapply(
      species_coeffs,
      function(x) {

        b1 <- x[1]
        b2 <- x[2]
        b3 <- x[3]

        b1 * dbh^b2 * height^b3
      },
      numeric(1)
    )
  )

  return(agb)
}


# =========================================================================
# FUNCTION 2: Calculate volume for one tree
# =========================================================================

calc_volume <- function(species, dbh, height) {

  # Return NA if species or tree measurements are invalid
  if (
    is.na(species) ||
    is.na(dbh) ||
    is.na(height) ||
    dbh <= 0 ||
    height <= 0 ||
    !(species %in% names(volume_coeffs))
  ) {
    return(NA_real_)
  }

  # Get volume coefficients for the specified species
  coefs <- volume_coeffs[[species]]

  # Calculate individual-tree volume
  volume <- coefs["b0"] +
    coefs["b1"] *
    dbh^coefs["b2"] *
    height^coefs["b3"]

  return(as.numeric(volume))
}


# =========================================================================
# FUNCTION 3: Convert CSV coordinates to a shapefile
# =========================================================================

csv_to_shp <- function(
    csv_path,                  # Path to the input CSV containing coordinates
    output_shp,                # Path for the output shapefile
    x_col = "longitude",       # Name of the x-coordinate column
    y_col = "latitude",        # Name of the y-coordinate column
    plot_id = NULL,            # Optional plot ID column used to group duplicate plots
    input_crs,                 # CRS of the coordinates in the CSV
    output_crs                 # CRS for the output spatial layer
) {

  # Read the CSV file
  df <- readr::read_csv(
    csv_path,
    show_col_types = FALSE
  )

  # Check that the coordinate columns exist
  required <- c(x_col, y_col)

  # If a plot ID is provided, also check that it exists
  if (!is.null(plot_id)) {
    required <- c(required, plot_id)
  }

  missing <- setdiff(required, names(df))

  # Stop if any required columns are missing
  if (length(missing) > 0) {
    stop(
      paste0(
        "Missing required columns: ",
        paste(missing, collapse = ", "),
        "\nCheck that x_col = '", x_col,
        "' and y_col = '", y_col, "'",
        if (!is.null(plot_id)) {
          paste0(" and plot_id = '", plot_id, "'")
        } else {
          ""
        },
        " match the column names in the CSV."
      )
    )
  }

  # If a plot ID column is supplied, group duplicate plot IDs
  # and calculate the mean x and y coordinates for each plot
  if (!is.null(plot_id)) {

    df <- df |>
      dplyr::group_by(.data[[plot_id]]) |>
      dplyr::summarise(
        "{x_col}" := mean(.data[[x_col]], na.rm = TRUE),
        "{y_col}" := mean(.data[[y_col]], na.rm = TRUE),
        .groups = "drop"
      )
  }

  # Convert the data frame to an sf spatial point layer
  out <- sf::st_as_sf(
    df,
    coords = c(x_col, y_col),
    crs = input_crs,
    remove = FALSE
  )

  # Transform the points to the desired output CRS
  out <- sf::st_transform(
    out,
    crs = output_crs
  )

  # Save the spatial layer as an ESRI Shapefile
  sf::st_write(
    out,
    output_shp,
    driver = "ESRI Shapefile",
    delete_layer = TRUE,
    quiet = TRUE
  )

  # Return the spatial layer for continued use in R
  return(out)
}


# =========================================================================
# FUNCTION 4: Summarise individual tree measurements by plot
# =========================================================================

summarize_plots <- function(
    tree_csv,                  # Path to CSV containing individual tree records
    plot_id = "nfm_id",        # Column identifying the plot for each tree
    dbh_col = "dbh",           # Column containing DBH measurements
    height_col = "height",     # Column containing tree height measurements
    species_col = "species",   # Column containing species information
    plot_area_ha = 0.04        # Plot area in hectares
) {

  # Read the individual tree data
  trees <- readr::read_csv(
    tree_csv,
    show_col_types = FALSE
  )

  # Check that all required columns exist in the tree data
  required <- c(
    plot_id,
    dbh_col,
    height_col,
    species_col
  )

  missing <- setdiff(required, names(trees))

  # Stop if any required columns are missing
  if (length(missing) > 0) {
    stop(
      paste0(
        "Missing required columns: ",
        paste(missing, collapse = ", "),
        "\nCheck that plot_id = '", plot_id,
        "', dbh_col = '", dbh_col,
        "', height_col = '", height_col,
        "', and species_col = '", species_col,
        "' match the column names in the CSV."
      )
    )
  }


  # -----------------------------------------------------------------------
  # Standardize species codes
  # -----------------------------------------------------------------------
  #
  # Converts species codes to consistent capitalization so that
  # differences in capitalization do not affect coefficient matching.
  #
  # Examples:
  #   "SW" -> "Sw"
  #   "sw" -> "Sw"
  #   "PL" -> "Pl"
  #   "AT" -> "At"
  #

  trees <- trees |>
    dplyr::mutate(
      "{species_col}" := paste0(
        toupper(substr(.data[[species_col]], 1, 1)),
        tolower(substr(
          .data[[species_col]],
          2,
          nchar(.data[[species_col]])
        ))
      )
    )


  # -----------------------------------------------------------------------
  # Calculate biomass and volume for each individual tree
  # -----------------------------------------------------------------------

  trees <- trees |>
    dplyr::rowwise() |>
    dplyr::mutate(

      # Above-ground biomass in kilograms
      AGB_kg = calc_agb(
        species = .data[[species_col]],
        dbh = .data[[dbh_col]],
        height = .data[[height_col]]
      ),

      # Tree volume in cubic metres
      Volume_m3 = calc_volume(
        species = .data[[species_col]],
        dbh = .data[[dbh_col]],
        height = .data[[height_col]]
      )
    ) |>
    dplyr::ungroup()


  # -----------------------------------------------------------------------
  # Group trees by plot and calculate plot-level summaries
  # -----------------------------------------------------------------------

  summary <- trees |>

    # Group all trees belonging to the same plot
    dplyr::group_by(.data[[plot_id]]) |>

    # Calculate summary statistics for each plot
    dplyr::summarise(

      # Number of trees measured in the plot
      n_trees = dplyr::n(),

      # DBH summary statistics
      dbh_mean = mean(.data[[dbh_col]], na.rm = TRUE),
      dbh_min  = min(.data[[dbh_col]], na.rm = TRUE),
      dbh_max  = max(.data[[dbh_col]], na.rm = TRUE),

      # Height summary statistics
      ht_mean = mean(.data[[height_col]], na.rm = TRUE),
      ht_min  = min(.data[[height_col]], na.rm = TRUE),
      ht_max  = max(.data[[height_col]], na.rm = TRUE),

      # Total above-ground biomass in the plot (kg)
      agb_kg = sum(
        AGB_kg,
        na.rm = TRUE
      ),

      # Total tree volume in the plot (m3)
      volume_m3 = sum(
        Volume_m3,
        na.rm = TRUE
      ),

      # Remove grouping after summarising
      .groups = "drop"
    ) |>

    # ---------------------------------------------------------------------
    # Convert plot-level totals to per-hectare values
    # ---------------------------------------------------------------------

    dplyr::mutate(

      # Trees per hectare
      density_tpha =
        n_trees / plot_area_ha,

      # Above-ground biomass in tonnes per hectare
      agb_t_ha =
        (agb_kg / 1000) /
        plot_area_ha,

      # Volume in cubic metres per hectare
      vlm_m3_ha =
        volume_m3 /
        plot_area_ha
    ) |>

    # Remove intermediate plot totals
    dplyr::select(
      -agb_kg,
      -volume_m3
    )

  # Return one row per plot containing the calculated summaries
  return(summary)
}


# =========================================================================
# USER INPUTS
# =========================================================================
#
# Update the values below for your dataset.
#


# Plot CSV containing coordinates
plot_csv <- "F:\\yukon\\plot_generation\\2026_Nahanni_LiDAR_Calibration_Plots.csv"

# Tree CSV containing individual tree measurements
tree_csv <- "F:\\yukon\\plot_generation\\2026_Nahanni_LiDAR_Calibration_Plots.csv"

# Intermediate shapefile containing only plot locations
plot_shp <- "F:\\yukon\\plot_generation\\plots.shp"

# Final shapefile containing plot locations + tree summaries
output_shp <- "F:\\yukon\\plot_generation\\plots_summary.shp"


# -------------------------------------------------------------------------
# Column names
# -------------------------------------------------------------------------

# Coordinate columns
x_col <- "Longitude"
y_col <- "Latitude"

# Column identifying each plot
plot_id <- "Plot ID"

# Individual tree columns
dbh_col <- "DBH (cm)"
height_col <- "Height (m)"

# Species column
#
# Values in this column must correspond to names in the biomass
# and volume coefficient lists (e.g., Sw, Sb, At, Pl, Bw).
species_col <- "Species"


# -------------------------------------------------------------------------
# Coordinate reference systems
# -------------------------------------------------------------------------
#
# EPSG:4326 = WGS84 latitude/longitude
# EPSG:32608 = WGS84 / UTM Zone 8N
#

input_crs <- 4326
output_crs <- 32608


# -------------------------------------------------------------------------
# Plot area in hectares
# -------------------------------------------------------------------------
#
# 400 m2 plot = 0.04 ha
#

plot_area_ha <- 0.04


# =========================================================================
# RUN WORKFLOW
# =========================================================================


# -------------------------------------------------------------------------
# 1. Convert plot coordinates from CSV to a spatial shapefile
# -------------------------------------------------------------------------
#
# If multiple records have the same Plot ID, the mean latitude and
# longitude are used to create a single point for that plot.
#

plots <- csv_to_shp(
  csv_path = plot_csv,
  output_shp = plot_shp,
  x_col = x_col,
  y_col = y_col,
  plot_id = plot_id,
  input_crs = input_crs,
  output_crs = output_crs
)


# -------------------------------------------------------------------------
# 2. Calculate tree metrics and summarise measurements by plot
# -------------------------------------------------------------------------

plot_summary <- summarize_plots(
  tree_csv = tree_csv,
  plot_id = plot_id,
  dbh_col = dbh_col,
  height_col = height_col,
  species_col = species_col,
  plot_area_ha = plot_area_ha
)


# -------------------------------------------------------------------------
# 3. Join tree summaries to the spatial plot layer
# -------------------------------------------------------------------------
#
# The Plot ID must exist in both datasets.
#

plots_summary <- plots |>
  dplyr::left_join(
    plot_summary,
    by = plot_id
  )


# -------------------------------------------------------------------------
# 4. Save the final spatial plot layer
# -------------------------------------------------------------------------

sf::st_write(
  plots_summary,
  output_shp,
  driver = "ESRI Shapefile",
  delete_layer = TRUE,
  quiet = TRUE
)


# -------------------------------------------------------------------------
# 5. Display the completed spatial layer
# -------------------------------------------------------------------------

print(plots_summary)