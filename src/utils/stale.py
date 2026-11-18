# src/utils/stale.py
import os


def remove_stale(*paths):
    """
    Σβήνει αποτελέσματα προηγούμενης εκτέλεσης όταν ένα στάδιο δεν μπορεί
    να τρέξει (π.χ. λείπουν τα δεδομένα εισόδου).

    Αλλιώς ο παλιός πίνακας μένει στον δίσκο και το manifest τον καταγράφει
    σαν αποτέλεσμα της τρέχουσας εκτέλεσης. Όταν το στάδιο τρέχει αλλά δεν
    βρίσκει τίποτα, γράφεται κενός πίνακας, όχι αυτή η συνάρτηση: το «κανένας»
    είναι αποτέλεσμα, το «δεν έτρεξε» δεν είναι.
    """
    removed = [path for path in paths if os.path.exists(path)]
    for path in removed:
        os.remove(path)
    if removed:
        names = ', '.join(os.path.basename(path) for path in removed)
        print(f"    - Σβήστηκαν παλιά αποτελέσματα που δεν ισχύουν πια: {names}")
    return removed
