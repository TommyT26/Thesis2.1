"""
Ποιοι hosts των προφίλ ΔΕΝ είναι hosts του διαδικτύου: ιδιωτικές διευθύνσεις
RFC1918 και διευθύνσεις ειδικής χρήσης (multicast, εκπομπή, link-local κ.λπ.).

ΓΙΑΤΙ ΧΡΕΙΑΖΕΤΑΙ ΞΕΧΩΡΙΣΤΟ ΒΗΜΑ. Μετά την ψευδωνυμοποίηση δεν υπάρχει IP, μόνο
HMAC-SHA256. Τα hashes δεν αντιστρέφονται, οπότε ο μόνος τρόπος να απαντηθεί
«είναι αυτός ο host ιδιωτική διεύθυνση;» είναι να υπολογιστεί προς τα εμπρός
το hash κάθε διεύθυνσης του εύρους και να ελεγχθεί αν εμφανίζεται στα
δεδομένα. Γίνεται ΜΙΑ φορά, γράφεται ο κατάλογος των ευρημάτων, και τα στάδια
διαβάζουν αυτόν.

ΤΙ ΒΡΗΚΕ ΤΟ RFC1918. Το πρώτο τρέξιμο έδειξε 85 ιδιωτικές διευθύνσεις εκτός των
τριών υποδικτύων της ανάλυσης, με 186.178 ροές (0,73% του συνόλου), από τις
οποίες 20 ήταν μέσα στις 1.846 ανωμαλίες του Isolation Forest. Δεν είναι
θόρυβος: κίνηση OpenFlow (6653/6633) προς το 192.168.8.x, DNS και NTP,
Kubernetes API (6443), γέφυρες Docker (172.17/172.18), Splashtop (6783) και
MikroTik Neighbor Discovery (5678). Είναι υποδομή, όχι hosts του διαδικτύου.
Η πρώτη σάρωση δεν είχε το 10.0.0.0/8, που περιείχε μόνο του 56 διευθύνσεις.

ΤΙ ΒΡΗΚΑΝ ΤΑ ΕΙΔΙΚΑ ΕΥΡΗ (19/11/2026). Η 255.255.255.255 δεχόταν 416.981 ροές
και η εκπομπή του εξωτερικού /24 του router 656.579, χωρίς να στέλνουν καμία.
Και οι δύο ήταν μέσα στις ανωμαλίες του Isolation Forest ως «εξωτερικοί
hosts». Βρέθηκαν επίσης η διεύθυνση δικτύου (.0) του ίδιου /24, μία
διεύθυνση multicast του 224.0.0.x, η 0.0.0.0, τέσσερις link-local
(169.254/16) και επτά του 100.64/10, 16 συνολικά. Μια διεύθυνση
multicast ή εκπομπής δεν είναι μηχανή, και οι υπόλοιπες δεν δρομολογούνται
στο διαδίκτυο, οπότε βγαίνουν από τον εξωτερικό πληθυσμό όπως οι ιδιωτικές.

ΚΟΣΤΟΣ. Τα ειδικά εύρη είναι περίπου 306 εκατομμύρια διευθύνσεις, σχεδόν όλες
του multicast (224.0.0.0/4). Η πρώτη σάρωση κράτησε περίπου 10 λεπτά. Ο κατάλογος
κρατά ένα αποτύπωμα των hosts των προφίλ και των ρυθμίσεων, και αν δεν έχουν
αλλάξει η σάρωση δεν ξαναγίνεται. Το αποτέλεσμα εξαρτάται μόνο από αυτά.

ΑΣΦΑΛΕΙΑ. Ο κατάλογος κρατά hash και ετικέτα /24, ποτέ το τελευταίο οκτέτο,
ίδιος κανόνας με όλα τα άλλα αποτελέσματα. Για το εξωτερικό /24 ούτε καν
αυτό: η ετικέτα του είναι γενική, γιατί το πρόθεμα είναι δημόσια διεύθυνση
και μένει μόνο στο .env. Το αποτύπωμα του .state είναι HMAC με το κλειδί,
ώστε να μην αποκαλύπτει το πρόθεμα. Η σάρωση γίνεται με streaming. Καμία αντιστροφή,
κανένας πίνακας αντιστοίχισης δεν γράφεται στον δίσκο.
"""
import os
import csv
import hashlib
import hmac

from src.config import PATHS, BASE_DIR, SUBNET_PREFIXES, OUTER_PREFIX, SALT_KEY_BYTES
from src.utils.anonymize import prefix_hashes

OUTPUT = os.path.join(BASE_DIR, 'results', 'private_hosts.csv')

# Τα τρία μπλοκ του RFC1918, ως πλήθος /24 το καθένα:
#   10.0.0.0/8      -> 65.536 υποδίκτυα /24
#   172.16.0.0/12   ->  4.096
#   192.168.0.0/16  ->    256
TOTAL_PREFIXES = 65536 + 4096 + 256

# Διευθύνσεις ειδικής χρήσης που δεν είναι hosts του διαδικτύου. Είναι
# υποσύνολο του RFC 6890: λείπουν τα 192.0.0.0/24, τα τρία TEST-NET,
# το 198.18.0.0/15 και το 240.0.0.0/4 (εκτός από την 255.255.255.255, που
# ελέγχεται χωριστά). Δεν έχει μετρηθεί αν εμφανίζονται στα δεδομένα.
# Ως (πρώτο οκτέτο, δεύτερα οκτέτα) ανά κατηγορία, για να βγουν τα /24.
SPECIAL_RANGES = {
    'this-network': [(0, range(256))],                     # 0.0.0.0/8
    'CGNAT': [(100, range(64, 128))],                      # 100.64.0.0/10
    'loopback': [(127, range(256))],                       # 127.0.0.0/8
    'link-local': [(169, range(254, 255))],                # 169.254.0.0/16
    'multicast': [(a, range(256)) for a in range(224, 240)],  # 224.0.0.0/4
    'single': [('255.255.255.255', 'broadcast')],
}

# Η ετικέτα του εξωτερικού /24. Γενική επίτηδες: δεν περιέχει το πρόθεμα.
OUTER_LABEL = 'εξωτερικό-δίκτυο.x'


def private_prefixes():
    """Κάθε /24 του RFC1918, ως πρόθεμα με τελεία στο τέλος."""
    for a in range(256):
        for b in range(256):
            yield f'10.{a}.{b}.'
    for a in range(16, 32):
        for b in range(256):
            yield f'172.{a}.{b}.'
    for b in range(256):
        yield f'192.168.{b}.'


def special_prefixes():
    """Κάθε /24 των ειδικών ευρών, ως (πρόθεμα, κατηγορία)."""
    for category, blocks in SPECIAL_RANGES.items():
        if category == 'single':
            continue
        for first, seconds in blocks:
            for b in seconds:
                for c in range(256):
                    yield f'{first}.{b}.{c}.', category


def scan(hosts, report=None, prefixes=None):
    """
    {hash: ετικέτα /24} για κάθε host των δεδομένων που είναι RFC1918.

    hosts: σύνολο των hashes που εμφανίζονται στα προφίλ.
    report: προαιρετική συνάρτηση προόδου, καλείται ανά 4.096 προθέματα.
    prefixes: ποια /24 να σαρωθούν· προεπιλογή όλο το RFC1918. Υπάρχει ως
    παράμετρος για τους ελέγχους, που δεν έχουν λόγο να πληρώνουν 17,9
    εκατομμύρια hashes για να επαληθεύσουν τη λογική της αντιστοίχισης.
    """
    found = {}
    for index, prefix in enumerate(private_prefixes() if prefixes is None else prefixes):
        for host in prefix_hashes(prefix):
            if host in hosts:
                found[host] = prefix + 'x'
        if report is not None and index % 4096 == 0:
            report(index, len(found))
    return found


def scan_special(hosts, report=None, prefixes=None, outer_prefix=None):
    """
    {hash: (ετικέτα, κατηγορία)} για τους hosts των δεδομένων που είναι
    διευθύνσεις ειδικής χρήσης.

    prefixes: λίστα (πρόθεμα, κατηγορία)· προεπιλογή όλα τα ειδικά εύρη.
    outer_prefix: το εξωτερικό /24 (προεπιλογή από το .env). Από αυτό
    ελέγχονται μόνο η .0 και η .255, με γενική ετικέτα.
    """
    found = {}
    source = special_prefixes() if prefixes is None else prefixes
    for index, (prefix, category) in enumerate(source):
        label = prefix + 'x'
        for host in prefix_hashes(prefix):
            if host in hosts:
                found[host] = (label, category)
        if report is not None and index % 65536 == 0:
            report(index, len(found))

    for ip, category in SPECIAL_RANGES['single']:
        prefix, octet = ip.rsplit('.', 1)
        host = prefix_hashes(prefix + '.')[int(octet)]
        if host in hosts:
            found[host] = (prefix + '.x', category)

    outer = OUTER_PREFIX if outer_prefix is None else outer_prefix
    if outer:
        hashes = prefix_hashes(outer)
        for octet, category in ((0, 'outer-network'), (255, 'outer-broadcast')):
            if hashes[octet] in hosts:
                found[hashes[octet]] = (OUTER_LABEL, category)
    return found


def total_special_prefixes():
    return sum(len(seconds) * 256
               for category, blocks in SPECIAL_RANGES.items() if category != 'single'
               for _, seconds in blocks)


def profile_hosts():
    """Τα hashes των προφίλ. Μόνο η στήλη Host, χωρίς pandas."""
    path = os.path.join(PATHS['features'], 'features_global.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Δεν βρέθηκε το {path}. Τρέξε πρώτα:  python main.py --step features")
    with open(path, encoding='utf-8-sig', newline='') as handle:
        return {row['Host'] for row in csv.DictReader(handle)}


def fingerprint(hosts, key=None):
    """
    Αποτύπωμα όσων καθορίζουν τον κατάλογο: hosts, εύρη, εξωτερικό /24.

    Είναι HMAC-SHA256 με το μυστικό κλειδί και όχι σκέτο SHA256. Το πρόθεμα
    του εξωτερικού /24 έχει μόνο 2^24 πιθανές τιμές και οι hosts φαίνονται στα
    αποτελέσματα, οπότε ένα αποτύπωμα χωρίς κλειδί θα αποκάλυπτε το πρόθεμα
    με δοκιμή όλων των τιμών σε λίγα δευτερόλεπτα (έλεγχος κώδικα 20/11/2026).
    Με το κλειδί η δοκιμή δεν γίνεται χωρίς το .env, και το αποτύπωμα αλλάζει
    επίσης όταν αλλάξει το κλειδί, οπότε η σάρωση ξαναγίνεται τότε.
    """
    key = SALT_KEY_BYTES if key is None else key
    digest = hmac.new(key, digestmod=hashlib.sha256)
    for host in sorted(hosts):
        digest.update(host.encode('ascii') + b'\n')
    digest.update(repr(sorted((k, repr(v)) for k, v in SPECIAL_RANGES.items())).encode())
    digest.update(b'outer:' + OUTER_PREFIX.encode('ascii'))
    digest.update(b'v3')
    return digest.hexdigest()


def state_path():
    return OUTPUT + '.state'


def unchanged(hosts):
    """Αληθές αν ο κατάλογος υπάρχει και φτιάχτηκε από τα ίδια δεδομένα."""
    if not (os.path.exists(OUTPUT) and os.path.exists(state_path())):
        return False
    with open(state_path(), encoding='ascii') as handle:
        return handle.read().strip() == fingerprint(hosts)


def load():
    """
    Ο κατάλογος από τον δίσκο: {hash: ετικέτα /24}. Άδειος αν λείπει.

    Το άδειο δεν είναι σιωπηλό. Όποιος χωρίζει πληθυσμούς χωρίς αυτόν τον
    κατάλογο ξαναβάζει τις ιδιωτικές διευθύνσεις στους «εξωτερικούς hosts του
    διαδικτύου», που είναι ακριβώς το λάθος που το βήμα αυτό διορθώνει.
    """
    if not os.path.exists(OUTPUT):
        return {}
    with open(OUTPUT, encoding='utf-8-sig', newline='') as handle:
        return {row['Host']: row['Υποδίκτυο'] for row in csv.DictReader(handle)}


def ours():
    """Τα προθέματα των υποδικτύων της ανάλυσης, για να εξαιρεθούν."""
    return tuple(SUBNET_PREFIXES)


def others():
    """
    Τα hashes των διευθύνσεων του καταλόγου ΕΚΤΟΣ των δικών μας υποδικτύων.

    Αυτό είναι το σύνολο που βγαίνει από τον εξωτερικό πληθυσμό: ιδιωτικές
    άλλων /24 και διευθύνσεις ειδικής χρήσης. Τα δικά μας υποδίκτυα τα
    χειρίζεται το populations.py και δεν περνούν από εδώ.
    """
    mine = ours()
    return frozenset(host for host, label in load().items()
                     if not label.rstrip('x').startswith(mine))


def run():
    print("\n--- Έναρξη: Εντοπισμός Ιδιωτικών και Ειδικών Διευθύνσεων ---")
    hosts = profile_hosts()
    print(f" -> {len(hosts):,} hosts στα προφίλ")

    # Ο διαχωρισμός κρατά τον κατάλογο στη μνήμη (lru_cache). Καθαρίζεται εδώ,
    # ώστε στο διαδραστικό μενού το επόμενο στάδιο να διαβάσει τον νέο.
    from src.models.data_split import private_hosts
    private_hosts.cache_clear()

    if unchanged(hosts):
        print(" -> Ο κατάλογος φτιάχτηκε ήδη από τους ίδιους hosts και τις ίδιες "
              "ρυθμίσεις. Η σάρωση παραλείπεται.")
        print(f" -> Κατάλογος: {OUTPUT}")
        print("--- Ολοκληρώθηκε ---")
        return

    print(f" -> Σάρωση {TOTAL_PREFIXES * 256:,} διευθύνσεων του RFC1918 "
          f"(10/8, 172.16/12, 192.168/16)")
    print("    Υπολογίζονται hashes προς τα εμπρός· καμία αντιστροφή.")

    def report(index, hits):
        print(f"    - {index / TOTAL_PREFIXES:5.1%} | ευρήματα ως τώρα: {hits}", flush=True)

    found = scan(hosts, report=report)

    total = total_special_prefixes()
    print(f" -> Σάρωση {total * 256 + 1:,} διευθύνσεων ειδικής χρήσης "
          f"(0/8, 100.64/10, 127/8, 169.254/16, 224/4, 255.255.255.255)")
    print("    " + ("Ελέγχονται και η .0 και η .255 του εξωτερικού /24 (από το .env)."
                    if OUTER_PREFIX else
                    "Δεν δόθηκε εξωτερικό /24 στο .env (THESIS_OUTER_PREFIX)."))

    def report_special(index, hits):
        print(f"    - {index / total:5.1%} | ευρήματα ως τώρα: {hits}", flush=True)

    special = scan_special(hosts, report=report_special)

    mine = ours()
    inside = {h: l for h, l in found.items() if l.rstrip('x').startswith(mine)}
    outside = {h: l for h, l in found.items() if h not in inside}

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, 'w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['Host', 'Υποδίκτυο', 'Στα_υποδίκτυα_της_αναλυσης', 'Κατηγορία'])
        for host, label in sorted(found.items(), key=lambda item: item[1]):
            writer.writerow([host, label, host in inside, 'RFC1918'])
        for host, (label, category) in sorted(special.items(), key=lambda item: item[1]):
            writer.writerow([host, label, False, category])
    with open(state_path(), 'w', encoding='ascii') as handle:
        handle.write(fingerprint(hosts))

    print(f"\n -> Βρέθηκαν {len(found)} ιδιωτικές διευθύνσεις στα δεδομένα:")
    print(f"    {len(inside)} στα υποδίκτυα της ανάλυσης ({', '.join(mine)})")
    print(f"    {len(outside)} ΕΚΤΟΣ αυτών — βγαίνουν από τον εξωτερικό πληθυσμό")

    groups = {}
    for label in outside.values():
        groups[label] = groups.get(label, 0) + 1
    for label, count in sorted(groups.items(), key=lambda item: -item[1]):
        print(f"      {label:<18} {count}")

    print(f"\n -> Βρέθηκαν {len(special)} διευθύνσεις ειδικής χρήσης "
          f"— βγαίνουν κι αυτές από τον εξωτερικό πληθυσμό:")
    categories = {}
    for _, category in special.values():
        categories[category] = categories.get(category, 0) + 1
    for category, count in sorted(categories.items()):
        print(f"      {category:<18} {count}")

    print(f"\n -> Ο κατάλογος αποθηκεύτηκε στο: {OUTPUT}")
    print("--- Ολοκληρώθηκε ---")


if __name__ == "__main__":
    run()
