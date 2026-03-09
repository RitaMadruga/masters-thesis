library(TCGAbiolinks)
library(SummarizedExperiment)

#mrna
query_mrna <- GDCquery(
  project = "TCGA-PAAD",
  data.category = "Transcriptome Profiling",
  data.type = "Gene Expression Quantification", 
  workflow.type = "STAR - Counts"
)

GDCdownload(query_mrna)

data_mrna <- GDCprepare(query_mrna)

mrna_tpm <- assay(data_mrna, "tpm_unstrand")

saveRDS(mrna_tpm, file = "TCGA_PAAD_mRNA_TPM.rds")

#miRNA

query_mirna <- GDCquery(
  project = "TCGA-PAAD",
  data.category = "Transcriptome Profiling",
  data.type = "miRNA Expression Quantification"
)

GDCdownload(query_mirna)

mirna_data <- GDCprepare(query_mirna)

saveRDS(mirna_data, file = "TCGA_PAAD_miRNA_Raw.rds")

#DNA Methylation

query_met <- GDCquery(
  project = "TCGA-PAAD",
  data.category = "DNA Methylation",
  data.type = "Methylation Beta Value",
  platform = "Illumina Human Methylation 450"
)

GDCdownload(query_met)

data_met <- GDCprepare(query_met)

saveRDS(data_met, file = "TCGA_PAAD_Methylation_450k.rds")

#save as CSV files

mirna_data <- readRDS("TCGA_PAAD_miRNA_Raw.rds")
write.csv(mirna_data, "TCGA_PAAD_miRNA_Raw.csv", row.names = TRUE)

mrna_data <- readRDS("TCGA_PAAD_mRNA_TPM.rds")
write.csv(mrna_data, "TCGA_PAAD_mRNA_TPM.csv", row.names = TRUE)

meth_data <- readRDS("TCGA_PAAD_Methylation_450k.rds")

meth_matrix <- assay(meth_data)
write.csv(meth_matrix, "TCGA_PAAD_Methylation.csv", row.names = TRUE)