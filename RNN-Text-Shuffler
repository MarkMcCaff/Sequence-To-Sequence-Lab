import random

outputtesttext = []

def main():
    file = open("geo-train-unique-spelled.txt", "r")
    outputfile = open("shuffled-geo-train-lm.txt", "a")
    outputtestfile = open("shuffled-geo-test-lm.txt", "w")
    with file as file:
        for line in file:
            lmsu(line,outputfile,outputtestfile)
    readfile = open("shuffled-geo-train-lm.txt", "r")
    shuffler = []
    with readfile as file:
        for line in readfile:
            shuffler.append(line)

    i = shuffler[0:1000]
    for x in i:
        outputtestfile.write(x)


def lmsu(line,outputfile,outputtestfile):
    words = [line.split()]
    for i in words:
        if len(i) <= 6:
            shuffled = i.copy()
            random.shuffle(shuffled)
            shuffledtext = ""
            for a in shuffled:
                shuffledtext = shuffledtext + " " + a
            sentence = ""
            for z in i:
                firstword = z
                outputtext = firstword + "\t" + shuffledtext.lstrip() + " <s>" + sentence + "\n"
                outputfile.write(outputtext)
                sentence = sentence + " " + firstword
            outputtext = "</s>\t" + shuffledtext + "<s>" + sentence + "\n"
            outputfile.write(outputtext)


main()
