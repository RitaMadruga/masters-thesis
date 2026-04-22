library(IlluminaHumanMethylation450kanno.ilmn12.hg19)
ann <- getAnnotation(IlluminaHumanMethylation450kanno.ilmn12.hg19)

setwd("C:/Users/ritam/Desktop/thesis_rita/data/annotations")

xy_probes <- rownames(ann[ann$chr %in% c("chrX", "chrY"), ])
write.csv(data.frame(Probe = xy_probes), "meth_ChrXY_probes.csv", row.names = FALSE)

# Sondas com SNPs
snps <- getSnpInfo(IlluminaHumanMethylation450kanno.ilmn12.hg19)
snp_probes <- rownames(snps[!is.na(snps$Probe_rs) | !is.na(snps$CpG_rs), ])
write.csv(data.frame(Probe = snp_probes), "meth_SNP_probes.csv", row.names = FALSE)

print("Feito! Verifica a tua pasta Desktop > thesis_rita > data > annotations")


BiocManager::install("ChAMPdata")

library(ChAMPdata)

data("multi.hit")

write.csv(data.frame(Probe = multi.hit), "meth_cross_reactive_probes.csv", row.names = FALSE)

print("Lista de sondas Cross-Reactive gerada com sucesso na pasta annotations!")

my_mapping <- data.frame(
  CpG = rownames(ann),
  Gene = ann$UCSC_RefGene_Name
)

write.csv(my_mapping, "C:/Users/ritam/Desktop/thesis_rita/data/annotations/illumina_mapping.csv", row.names = FALSE)
