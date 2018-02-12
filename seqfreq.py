import pylab as pl
import numpy as np
string1 = """MRAVVVLLLVAVASAKVYDRCELARALKASGMDGYAGNSLPNWVCLSKWESSYNTQATNRNTDGSTDYGIFQINSRYWCDDGRTPGAKNVCGIRCSQLLTADLTVAIRCAKRVVLDPNGIGAWVAWRLHCQNQDLRSYVAGCGV
"""
alphabet = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]


def frequencies(string, letters):
    list_frequencies = []
    for letter in letters:
        freq = 0
        for i in string:
            if i == letter:
                freq += 1
        if freq != 0:
            list_frequencies.append(letter)
            list_frequencies.append(freq)
    return list_frequencies


print(frequencies(string1, alphabet))


def fix_lists_letter(list_1):
    list_letters = []
    list_letters.append(list_1[0])
    list_freq = []
    for i in range(1, len(list_1)):
        if i % 2 == 0:
            list_letters.append(list_1[i])
        else:
            list_freq.append(list_1[i])
    if len(list_letters) != len(list_freq):
        return "Some error occurred"
    else:
        final_list = [list_letters, list_freq]
        return final_list


first_count = frequencies(string1, alphabet)
final = fix_lists_letter(first_count)
letter_s = final[0]
freq = final[1]
print("Number of character used:", sum(freq), sep=" ")


# Relative frequencies

def get_rel_freq(list_1):
    list_to_ret = []
    for i in list_1:
        list_to_ret.append(i / sum(list_1))
    return list_to_ret


freq = get_rel_freq(freq)


fig = pl.figure()
ax = pl.subplot(111)
width = 0.8
ax.bar(range(len(letter_s)), freq, width=width)
ax.set_xticks(np.arange(len(letter_s)) + width / 2)
ax.set_xticklabels(letter_s, rotation=45)
pl.show()
