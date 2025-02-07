library(shiny)
library(tidyverse)
library(patchwork)

source('functional_response_shiny.R')
ui <- fluidPage(
  hr(),
  withMathJax(  
    
    p("\\( mort_X = 0.5 * Z * \\frac{X^2}{v_x^2 + h_x * X^2 + h_y * Y^2} \\)"),
    
    p("\\( mort_Y = 0.5 * Z * \\frac{Y^2}{v_y^2 + h_x * X^2 + h_y * Y^2} \\)")
    
  ),
  hr(),
  sidebarPanel(
    sliderInput("v_x", "v_x:",
                min = 0, max = 2,
                value = 0.5, step = 0.01),
    sliderInput("v_y", "v_y:",
                min = 0, max = 2,
                value = 0.5, step = 0.01),
    sliderInput("h_x", "h_x:",
                min = 0, max = 2,
                value = 0.5, step = 0.01),
    sliderInput("h_y", "h_y:",
                min = 0, max = 2,
                value = 0.5, step = 0.01),
    sliderInput("Z", "Z:",
                min = 0, max = 0.5,
                value = 0.1, step = 0.05),
  ),
  mainPanel(
    plotOutput('plot')
    )
  )

server <- function(input, output, session){
  
  #all model types -- plot
  output$plot <- renderPlot({
    get_plot(mort_df = get_mort_df(params = list(v_x = input$v_x,
                                                 v_y = input$v_y,
                                                 h_x = input$h_x,
                                                 h_y = input$h_y,
                                                 Z = input$Z)))
    
  })
}

shinyApp(ui = ui, server = server)