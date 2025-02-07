library(tidyverse)
library(viridis)


# N = [Z, X, Y]

get_x_mortality <- function(N, params) {
  mort <- 0.5 * N[1] * (N[2] ^ 2) / (params$v_x ^ 2 + 
                                       params$h_x * N[2] ^ 2 + 
                                       params$h_y * N[3] ^ 2)
}

get_y_mortality <- function(N, params) {
  mort <- 0.5 * N[1] * (N[3] ^ 2) / (params$v_y ^ 2 + 
                                       params$h_x * N[2] ^ 2 + 
                                       params$h_y * N[3] ^ 2)
}

get_mort_df <- function(params) {
  XY <- expand.grid(X = seq(0, 1, 0.05), Y = seq(0, 1, 0.05))
  mort_df <- as.data.frame(matrix(NA, nrow = nrow(XY), ncol = 5))
  colnames(mort_df) <- c("Z", "X", "Y", "mort_X", "mort_Y")
  mort_df$Z <- params$Z
  mort_df[, c("X", "Y")] <- XY
  
  mort_df$mort_X <- apply(mort_df[, c("Z", "X", "Y")], 1, 
                          function(row) get_x_mortality(as.numeric(row), params))
  
  mort_df$mort_Y <- apply(mort_df[, c("Z", "X", "Y")], 1, 
                          function(row) get_y_mortality(as.numeric(row), params))
  
  return(mort_df)
}

get_plot <- function(mort_df) {
  mort_X <- ggplot(data = mort_df) +
    geom_tile(aes(x = X, y = Y, fill = mort_X)) + 
    scale_fill_viridis()
  mort_Y <- ggplot(data = mort_df) +
    geom_tile(aes(x = X, y = Y, fill = mort_Y)) +
    scale_fill_viridis()
  final <- mort_X + mort_Y + plot_layout(nrow = 1)
  return(final)
}

