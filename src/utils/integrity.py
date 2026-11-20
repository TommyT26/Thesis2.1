"""
Ελέγχει ότι δεν χάθηκαν δεδομένα στη συνένωση των ροών.

Η Φάση 2 συμπιέζει εκατομμύρια μονόδρομες ροές σε λίγες εκατοντάδες χιλιάδες
αμφίδρομες. Αν κάτι πάει λάθος εκεί, όλα τα επόμενα αποτελέσματα είναι άκυρα
χωρίς να φαίνεται πουθενά. Οπότε συγκρίνω τα συνολικά bytes και τις ροές πριν
και μετά, και πρέπει να βγαίνουν ακριβώς ίδια.

Ελέγχω όλα τα δεδομένα, όχι δείγμα: τα αθροίσματα είναι φθηνά και ένα δείγμα
μπορεί εύκολα να προσπεράσει σφάλμα που αφορά λίγες γραμμές.
"""
import os
import glob

import pandas as pd

from src.config import PATHS, CHUNK_SIZE

# Ανοχή για το σφάλμα κινητής υποδιαστολής όταν αθροίζονται εκατομμύρια τιμές
RELATIVE_TOLERANCE = 1e-9


def clean_path_for(bidir_path):
    return os.path.join(PATHS['processed_data'],
                        os.path.basename(bidir_path).replace("bidir_", "clean_", 1))


def temporal_path_for(bidir_path):
    return os.path.join(PATHS['bidirectional_data'],
                        os.path.basename(bidir_path).replace("bidir_", "temporal_", 1))


def clean_totals(path):
    """Αθροίσματα του καθαρισμένου αρχείου, σε chunks ώστε να μη φορτωθεί όλο μαζί."""
    total_bytes = 0.0
    rows = 0
    rows_with_hosts = 0
    same_rows = 0
    same_bytes = 0.0
    for chunk in pd.read_csv(path, usecols=['Bytes', 'Src_Host', 'Dst_Host'],
                             chunksize=CHUNK_SIZE, low_memory=False):
        values = pd.to_numeric(chunk['Bytes'].astype(str).str.strip(),
                               errors='coerce').fillna(0)
        total_bytes += float(values.sum())
        rows += len(chunk)
        rows_with_hosts += int((chunk['Src_Host'].notna() & chunk['Dst_Host'].notna()).sum())
        # Ροές με τον ίδιο host και στις δύο άκρες: ο χρονικός πίνακας τις μετρά
        # μία φορά (βλ. aggregate_flows.host_time_table), όχι δύο.
        same = chunk['Src_Host'].notna() & (chunk['Src_Host'] == chunk['Dst_Host'])
        same_rows += int(same.sum())
        same_bytes += float(values[same].sum())
    return {'bytes': total_bytes, 'rows': rows, 'rows_with_hosts': rows_with_hosts,
            'same_rows': same_rows, 'same_bytes': same_bytes}


def compare(label, expected, actual, unit=""):
    """Επιστρέφει (ok, γραμμή αναφοράς)."""
    diff = actual - expected
    scale = max(abs(expected), 1.0)
    ok = abs(diff) / scale <= RELATIVE_TOLERANCE
    mark = "OK " if ok else "ΑΠΟΤΥΧΙΑ"
    line = (f"      {mark} {label:<34} αναμενόμενο {expected:>18,.0f}{unit}"
            f" | βρέθηκε {actual:>18,.0f}{unit}")
    if not ok:
        line += f"  (διαφορά {diff:+,.0f})"
    return ok, line


def check_day(bidir_path):
    """Έλεγχος ενός ζεύγους clean/bidir (και του temporal, αν υπάρχει)."""
    name = os.path.basename(bidir_path)
    clean_path = clean_path_for(bidir_path)
    if not os.path.exists(clean_path):
        print(f"   -> {name}: δεν βρέθηκε το αντίστοιχο clean αρχείο. Παράλειψη.")
        return None

    print(f"   -> {name}")
    clean = clean_totals(clean_path)
    bidir = pd.read_csv(bidir_path)

    results = []
    results.append(compare("Σύνολο bytes", clean['bytes'],
                           float(bidir['Bytes_AtoB'].sum() + bidir['Bytes_BtoA'].sum())))
    results.append(compare("Πλήθος ροών", clean['rows_with_hosts'],
                           float(bidir['Flows_AtoB'].sum() + bidir['Flows_BtoA'].sum())))

    temporal_path = temporal_path_for(bidir_path)
    if os.path.exists(temporal_path):
        temporal = pd.read_csv(temporal_path)
        # Κάθε ροή μετριέται μία φορά για κάθε άκρο της, άρα διπλά, εκτός από
        # τις ροές με τον ίδιο host και στις δύο άκρες, που μετρούν μία φορά.
        results.append(compare("Ροές στον χρονικό πίνακα",
                               2 * clean['rows_with_hosts'] - clean['same_rows'],
                               float(temporal['Flows'].sum())))
        results.append(compare("Bytes στον χρονικό πίνακα",
                               2 * clean['bytes'] - clean['same_bytes'],
                               float(temporal['Bytes'].sum())))

    for ok, line in results:
        print(line)
    return all(ok for ok, _ in results)


def run():
    print("\n--- Έναρξη: Έλεγχος Ακεραιότητας Δεδομένων ---")
    bidir_files = sorted(glob.glob(os.path.join(PATHS['bidirectional_data'], "bidir_*.csv")))
    if not bidir_files:
        # Εξαίρεση και όχι προειδοποίηση, για τον ίδιο λόγο με την αποτυχία
        # παρακάτω: ένας έλεγχος που δεν έλεγξε τίποτα δεν είναι [OK].
        raise RuntimeError(f"Δεν βρέθηκαν αρχεία bidir στον φάκελο "
                           f"{PATHS['bidirectional_data']}· ο έλεγχος δεν έγινε.")

    print(f" -> Έλεγχος {len(bidir_files)} ημερών (πλήρη αθροίσματα, όχι δείγμα)")
    outcomes = [check_day(path) for path in bidir_files]
    checked = [o for o in outcomes if o is not None]

    print("\n" + "-" * 60)
    if checked and all(checked):
        print(f"   Και οι {len(checked)} ημέρες πέρασαν: δεν χάθηκαν bytes ούτε ροές.")
    elif not checked:
        print("   Δεν ελέγχθηκε καμία ημέρα.")
        print("-" * 60)
        raise RuntimeError("Δεν βρέθηκε κανένα clean αρχείο για τις ημέρες του "
                           "bidir· ο έλεγχος ακεραιότητας δεν έγινε.")
    else:
        failed = len(checked) - sum(checked)
        print(f"   [!] {failed} από {len(checked)} ημέρες ΑΠΕΤΥΧΑΝ. Τα αποτελέσματα των")
        print("       επόμενων φάσεων δεν είναι αξιόπιστα μέχρι να διορθωθεί.")
        print("-" * 60)
        # Εξαίρεση και όχι απλή προειδοποίηση: το main.py σταματά στο πρώτο
        # στάδιο που αποτυγχάνει. Με προειδοποίηση, μια ανεπίβλεπτη εκτέλεση
        # συνέχιζε και τελείωνε «επιτυχώς» πάνω σε αλλοιωμένα δεδομένα.
        raise RuntimeError(f"Ο έλεγχος ακεραιότητας απέτυχε σε {failed} από "
                           f"{len(checked)} ημέρες· η εκτέλεση σταματά εδώ.")
    print("-" * 60)
    print("--- Ολοκληρώθηκε ο Έλεγχος Ακεραιότητας ---")


if __name__ == "__main__":
    run()
