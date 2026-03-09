library(IlluminaHumanMethylation450kanno.ilmn12.hg19)
ann <- getAnnotation(IlluminaHumanMethylation450kanno.ilmn12.hg19)

# 1. Definir a pasta de destino (Ajusta o nome de utilizador se necessário)
setwd("C:/Users/ritam/Desktop/thesis_rita/data/annotations")

# 2. Carregar as bibliotecas
#library(methylGSA)
#library(IlluminaHumanMethylation450kanno.ilmn12.hg19)

# --- FASE A: Dicionário CpG -> Gene (O que o Gonçalo usou) ---
#data("CpG2Genetoy", package = "methylGSA")
#write.csv(CpG2Genetoy, "CpG2Gene_methylGSA.csv", row.names = FALSE)

# --- FASE B: Extrair Anotação para Filtros (X, Y e SNPs) ---
#ann <- getAnnotation(IlluminaHumanMethylation450kanno.ilmn12.hg19)

# Sondas nos Cromossomas X e Y
xy_probes <- rownames(ann[ann$chr %in% c("chrX", "chrY"), ])
write.csv(data.frame(Probe = xy_probes), "meth_ChrXY_probes.csv", row.names = FALSE)

# Sondas com SNPs
snps <- getSnpInfo(IlluminaHumanMethylation450kanno.ilmn12.hg19)
snp_probes <- rownames(snps[!is.na(snps$Probe_rs) | !is.na(snps$CpG_rs), ])
write.csv(data.frame(Probe = snp_probes), "meth_SNP_probes.csv", row.names = FALSE)

print("Feito! Verifica a tua pasta Desktop > thesis_rita > data > annotations")


# Instalar o pacote de dados do ChAMP (se ainda não o tiveres)
BiocManager::install("ChAMPdata")

library(ChAMPdata)

# Carregar a lista oficial de sondas multi-hit (cross-reactive)
data("multi.hit")

# Guardar em CSV na tua pasta annotations
write.csv(data.frame(Probe = multi.hit), "meth_cross_reactive_probes.csv", row.names = FALSE)

print("Lista de sondas Cross-Reactive gerada com sucesso na pasta annotations!")


# Criar uma tabela simples com o ID da sonda e o Gene correspondente
my_mapping <- data.frame(
  CpG = rownames(ann),
  Gene = ann$UCSC_RefGene_Name
)

# Gravar na tua pasta annotations
write.csv(my_mapping, "C:/Users/ritam/Desktop/thesis_rita/data/annotations/illumina_mapping.csv", row.names = FALSE)
