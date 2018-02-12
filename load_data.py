import Bio
from Bio import SeqIO


def main():
    print("Biopython v" + Bio.__version__)
    # overall: 5959 sequences
    len_chloroplast = 449
    len_cytoplasmic = 1411
    len_ER = 198
    len_extracellular = 843
    len_Golgi = 150
    len_lysosomal = 103
    len_mitochondrial = 510
    len_nuclear = 837
    len_peroxisomal = 157
    len_plasma_membrane = 1238
    len_vacuolar = 63

    PATH = "data/"
    chloroplast = get_seq(PATH, "chloroplast.fasta", len_chloroplast)
    cytoplasmic = get_seq(PATH, "cytoplasmic.fasta", len_cytoplasmic)
    ER = get_seq(PATH, "ER.fasta", len_ER)
    extracellular = get_seq(PATH, "extracellular.fasta", len_extracellular)
    Golgi = get_seq(PATH, "Golgi.fasta", len_Golgi)
    lysosomal = get_seq(PATH, "lysosomal.fasta", len_lysosomal)
    mitochondrial = get_seq(PATH, "mitochondrial.fasta", len_mitochondrial)
    nuclear = get_seq(PATH, "nuclear.fasta", len_nuclear)
    peroxisomal = get_seq(PATH, "peroxisomal.fasta", len_peroxisomal)
    plasma_membrane = get_seq(PATH, "plasma_membrane.fasta",
                              len_plasma_membrane)
    vacuolar = get_seq(PATH, "vacuolar.fasta", len_vacuolar)

    chloroplast.__len__()
    cytoplasmic.__len__()
    ER.__len__()
    extracellular.__len__()
    Golgi.__len__()
    lysosomal.__len__()
    mitochondrial.__len__()
    nuclear.__len__()
    peroxisomal.__len__()
    plasma_membrane.__len__()
    vacuolar.__len__()


def get_seq(PATH, data_name, length):
    # Here we are setting up an array to save our sequences for the next step
    sequences = []
    for seq_record in SeqIO.parse(PATH + data_name, "fasta"):
        count = 0
        if (count <= length):
            sequences.append(seq_record)
            print("Id: " +
                  seq_record.id +
                  " \t " +
                  "Length: " +
                  str("{:,d}".format(len(seq_record))))
            print(repr(seq_record.seq) + "\n")
            count += 1
    return sequences


if __name__ == '__main__':
    main()
