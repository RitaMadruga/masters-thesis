# 1. Instalar o pacote se ainda não o tiveres

if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("TCGAbiolinks")

# 2. Carregar a biblioteca
library(TCGAbiolinks)

# 3. Fazer a Query (Exatamente como descrito na imagem)
query <- GDCquery(
  project = "TCGA-PAAD", 
  data.category = "Clinical", 
  data.type = "Clinical Supplement" 
)

# 4. Fazer o Download dos ficheiros
GDCdownload(query)

# 5. Preparar os dados (Transformar em tabelas utilizáveis no R)
clinical_paad_list <- GDCprepare(query)

# 6. Extrair a tabela específica de pacientes
# O GDCprepare para "Clinical Supplement" devolve uma lista com várias tabelas 
# (patient, drug, radiation, etc.). A professora guardou a tabela 'clinical_patient_paad'.
clinical_patient_paad <- clinical_paad_list$clinical_patient_paad

# 7. Visualizar as colunas de histologia
# É aqui que vais encontrar a coluna 'histological_diagnosis' ou similar
View(clinical_patient_paad)

write.csv(clinical_patient_paad, "tcga_paad_clinical.csv", row.names = FALSE)

getwd()
