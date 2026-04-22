# 1. Instalar o pacote se ainda não o tiveres

if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("TCGAbiolinks")

library(TCGAbiolinks)

query <- GDCquery(
  project = "TCGA-PAAD", 
  data.category = "Clinical", 
  data.type = "Clinical Supplement" 
)

GDCdownload(query)

clinical_paad_list <- GDCprepare(query)

clinical_patient_paad <- clinical_paad_list$clinical_patient_paad

View(clinical_patient_paad)

write.csv(clinical_patient_paad, "tcga_paad_clinical.csv", row.names = FALSE)

getwd()
