def tinbaDGA(idomain, seed):
    """
    l'algoritmo genera esadecimali casuali a partire dal seed e idomain. se il carattere (contenuto in al) fa parte dell'alfabeto minuscolo allora viene sommato alla stringa buf che diventera' il dominio.
    :param idomain:
    :param seed:
    :return:
    """
    suffix = ".com"  # TLD
    domains = []

    count = 100000
    eax = 0
    edx = 0
    for i in range(count):
        buf = ''
        esi = seed
        ecx = 0x10
        eax = 0  # registro accumulatore
        edx = 0  # registro dati
        for s in range(len(seed)):
            # Given a string of length one, return an integer representing the Unicode code. For example, ord('a') returns the integer 97
            eax = ord(seed[s])
            edx += eax  # somma degli int dei caratteri del seed.
        # edi = idomain
        ecx = 0x0C  # contatore al char FORM FEED . DETERMINA LA LUNGHEZZA DEL DOMINIO
        d = 0  # contatore per while
        while (ecx > 0):
            al = eax & 0xFF  # registro accumulatore 8 bit
            dl = edx & 0xFF  # registro dati 8 bit
            al = al + ord(idomain[d])  # somma a se stesso il valore int della d-esima lettera del dominio
            al = al ^ dl  # ^ Binary XOR	It copies the bit if it is set in one operand but not both.
            al += ord(idomain[d + 1])  # somma il valore ordinale del char successivo
            al = al & 0xFF  # estrae il valore a 8 bit
            eax = (eax & 0xFFFFFF00) + al  # riporta il registro a 32bit
            edx = (edx & 0xFFFFFF00) + dl  # uguale a eax
            if al > 0x61:  # se al e' maggiore del char a = 97
                if al < 0x7A:  # se al e' minore di z = 122
                    eax = (eax & 0xFFFFFF00) + al  # riporta il registro a 32bit
                    buf += chr(al)  # buffer stringa che genera il dominio sommando le lettere
                    d += 1  # contatore del while
                    ecx -= 1  # decrementa ecx
                    continue  # ricomincia il while
            dl += 1  # incrementa dl
            dl = dl & 0xFF  # estrae i primi 8 bit
            edx = (edx & 0xFFFFFF00) + dl  # concatena i primi 24 bit di edx a dl

        domain = buf + suffix  # concatena dominio al TLD
        domains.append(domain)  # lista domini
        idomain = domain  # il dominio viene usato come seed del prossimo dominio
    return domains


def init():
    harddomain = "ssrgwnrmgrxe.com"
    seed = "oGkS3w3sGGOGG7oc"
    domains = tinbaDGA(harddomain, seed)
    index = 0
    fp = open("../tinba.txt", "w")
    for domain in domains:
        index += 1
        fp.write(domain + '\n')
    fp.close()


if __name__ == "__main__":
    init()
