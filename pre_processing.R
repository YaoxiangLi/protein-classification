rm(list = ls())
setwd("/home/bach/Dropbox/2017fall/BIST532ML/project/data/")

# library(seqinr)
# smallFastaFile <- system.file("chloroplast.fasta", package = "seqinr")
# mySmallProtein <- read.fasta(file = smallFastaFile, as.string = TRUE, seqtype = "AA")[[1]]
# stopifnot(mySmallProtein == "SEQINRSEQINRSEQINRSEQINR*")
# read.fasta(file = system.file("chloroplast.fasta", package = "seqinr"), 
#            seqtype = c("DNA", "AA"), as.string = FALSE, forceDNAtolower = TRUE,
#            set.attributes = TRUE, legacy.mode = TRUE, seqonly = FALSE, strip.desc = FALSE,
#            bfa = FALSE, sizeof.longlong = .Machine$sizeof.longlong,
#            endian = .Platform$endian, apply.mask = TRUE)

# source("https://bioconductor.org/biocLite.R")
# biocLite("Biostrings")
library("Biostrings")
library("plyr")
library("stringr")

load_seq <- function(seq_data){
    fastaFile = readDNAStringSet(seq_data)
    name = names(fastaFile)
    sequence = paste(fastaFile)
    df = data.frame(name, sequence)
    df$label = seq_data
    return(df)
}

chloroplast <- load_seq("chloroplast.fasta")
cytoplasmic <- load_seq("cytoplasmic.fasta")
ER <- load_seq("ER.fasta")
extracellular <- load_seq("extracellular.fasta")
Golgi <- load_seq("Golgi.fasta")
lysosomal <- load_seq("lysosomal.fasta")
mitochondrial <- load_seq("mitochondrial.fasta")
nuclear <- load_seq("nuclear.fasta")
peroxisomal <- load_seq("peroxisomal.fasta")
plasma_membrane <- load_seq("plasma_membrane.fasta")
vacuolar <- load_seq("vacuolar.fasta")

proteins <- rbind(chloroplast,
                      cytoplasmic,
                      ER,
                      extracellular,
                      Golgi,
                      lysosomal,
                      mitochondrial,
                      nuclear,
                      peroxisomal,
                      plasma_membrane,
                      vacuolar)

rm(chloroplast,
      cytoplasmic,
      ER,
      extracellular,
      Golgi,
      lysosomal,
      mitochondrial,
      nuclear,
      peroxisomal,
      plasma_membrane,
      vacuolar)
proteins <- proteins[, c(3,1,2)]

proteins$sequence <- as.character(proteins$sequence)
proteins$name <- as.character(proteins$name)
proteins$sequence <- gsub('N', '', proteins$sequence)
proteins$sequence <- gsub('C', '', proteins$sequence)

counts <- rep(NA, nrow(proteins))
for (i in 1:nrow((proteins))){
    counts[i] <- str_length(proteins$sequence[i])
}
max(counts)
min(counts)

proteins <- proteins[-order(counts)[1:4696],]
counts <- rep(NA, nrow(proteins))

for (i in 1:nrow((proteins))){
    counts[i] <- str_length(proteins$sequence[i])
}

max(counts)
min(counts)
for (i in 1:nrow((proteins))){
    proteins$sequence[i] <- strtrim(proteins$sequence[i], 401)
}

counts <- rep(NA, nrow(proteins))

for (i in 1:nrow((proteins))){
    counts[i] <- str_length(proteins$sequence[i])
}

max(counts)
min(counts)
count(proteins$label)
write.csv(proteins, "proteins.csv")
