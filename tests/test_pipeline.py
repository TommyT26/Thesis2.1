"""
Έλεγχοι των σημείων όπου ένα λάθος θα περνούσε ΑΘΟΡΥΒΑ στα αποτελέσματα.

Δεν ελέγχω τα πάντα. Ελέγχω τις συναρτήσεις όπου ένα σφάλμα δεν ρίχνει το
pipeline αλλά αλλάζει σιωπηλά τους αριθμούς της αναφοράς — αυτά που δεν θα τα
έπιανα κοιτώντας την έξοδο. Τρία από αυτά έχουν ήδη συμβεί:

  - το Out Byte διαβαζόταν ως «Packets» και περνούσε στα μοντέλα ως σταθερά
    μηδενικό χαρακτηριστικό (νεκρή διάσταση στις αποστάσεις),
  - στην αντιστοίχιση ετικετών, όταν μια ομάδα της αναφοράς έλειπε από μια
    ημέρα, οι hosts εκείνης της ημέρας γίνονταν κενά και το ARI άλλαζε,
  - η ημερομηνία ως float32 στρογγυλοποιούνταν και ημέρες συγχωνεύονταν.

Κάθε ένα από αυτά έχει εδώ τον έλεγχο που θα το είχε πιάσει.

Εκτέλεση από τη ρίζα του project:

    py -m unittest discover -s tests -v
"""
import io
import os
import shutil
import tempfile
import contextlib
import unittest
from unittest import mock

# Το config απαιτεί κλειδί. Αν δεν υπάρχει .env (π.χ. σε καθαρό μηχάνημα),
# βάζω ένα δοκιμαστικό ΠΡΙΝ εισαχθεί οτιδήποτε από το src.
os.environ.setdefault('THESIS_SALT_KEY', 'test-key-only-for-unit-tests')

import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS, INTERNAL_SUBNET, SUBNET_PREFIXES
from src.utils.anonymize import (anonymize_ip, anonymize_series,
                                 get_internal_hashes, get_network_hashes)
from src.utils.scaling import scale_features
from src.preprocessing.aggregate_flows import (normalize_protocol, add_time_columns,
                                               host_time_table)
from src.models.dbscan_clustering import deduplicate, resolve_min_samples
from src.models.data_split import split_internal_external
from src.models.daily_stability import align_day
from src.models.peer_recurrence import recurrence, consecutive_share
from src.models.hourly_profile import hourly_shares, partial_days
from src.models import port_semantics
from src.models.port_semantics import NO_PORTS, port_pair_cases
from src.models import subnet_analysis
from src.models.subnet_analysis import write_peers, file_label
from src.models import populations as pops
from src.utils.anonymize import get_subnet_hashes, label_subnets, subnet_labels


def profile_frame(rows):
    """Ελάχιστο DataFrame προφίλ με όλες τις στήλες που θέλει το scale_features."""
    frame = pd.DataFrame(rows)
    for column in FEATURE_COLUMNS:
        if column not in frame:
            frame[column] = 1.0
    return frame


class TestAnonymize(unittest.TestCase):

    def test_ιδια_ip_δινει_ιδιο_hash(self):
        self.assertEqual(anonymize_ip('192.168.7.10'), anonymize_ip('192.168.7.10'))

    def test_διαφορετικες_ip_δινουν_διαφορετικα_hash(self):
        self.assertNotEqual(anonymize_ip('192.168.7.10'), anonymize_ip('192.168.7.11'))

    def test_το_hash_δεν_περιεχει_την_ip(self):
        # Αν η ανωνυμοποίηση σπάσει και επιστρέψει την είσοδο, θα φανεί εδώ.
        self.assertNotIn('192.168', anonymize_ip('192.168.7.10'))
        self.assertEqual(len(anonymize_ip('192.168.7.10')), 16)

    def test_series_συμφωνει_με_μεμονωμενο(self):
        ips = pd.Series(['10.0.0.1', '10.0.0.2', '10.0.0.1'])
        got = anonymize_series(ips)
        self.assertEqual(list(got), [anonymize_ip(ip) for ip in ips])

    def test_εσωτερικα_hashes_καλυπτουν_ολα_τα_υποδικτυα(self):
        # Το εσωτερικό δίκτυο ΔΕΝ είναι πια ένα υποδίκτυο: είναι όλα όσα
        # δηλώνει το SUBNET_PREFIXES, 254 hosts το καθένα.
        internal = get_internal_hashes()
        self.assertEqual(len(internal), 254 * len(SUBNET_PREFIXES))
        for prefix in SUBNET_PREFIXES:
            self.assertIn(anonymize_ip(f'{prefix}1'), internal, prefix)
            self.assertIn(anonymize_ip(f'{prefix}254'), internal, prefix)

    def test_σχημα_και_κανονικοποιηση(self):
        frame = profile_frame({'Bytes_In': [1, 10, 100, 1000],
                               'Total_Flows': [1, 2, 3, 400]})
        X = scale_features(frame)
        self.assertEqual(X.shape, (4, len(FEATURE_COLUMNS)))
        varying = X[:, 0]
        self.assertAlmostEqual(float(varying.mean()), 0.0, places=9)
        self.assertAlmostEqual(float(varying.std()), 1.0, places=9)

    def test_σταθερο_χαρακτηριστικο_αναφερεται(self):
        # Ο έλεγχος που θα είχε πιάσει το Total_Packets (πάντα 0).
        frame = profile_frame({'Bytes_In': [1, 10, 100], 'Total_Flows': [0, 0, 0]})
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            scale_features(frame)
        self.assertIn('Total_Flows', out.getvalue())
        self.assertIn('σταθερά', out.getvalue())

    def test_λειπον_χαρακτηριστικο_δινει_καθαρο_σφαλμα(self):
        frame = profile_frame({'Bytes_In': [1, 2]}).drop(columns=['Active_Hours'])
        with self.assertRaises(KeyError) as ctx:
            scale_features(frame)
        self.assertIn('Active_Hours', str(ctx.exception))


class TestAggregate(unittest.TestCase):

    def test_πρωτοκολλα_και_αριθμητικοι_κωδικοι(self):
        got = normalize_protocol(pd.Series(['tcp', 'TCP', '6', 'udp', '17', 'icmp', '1']))
        self.assertEqual(list(got),
                         ['TCP', 'TCP', 'TCP', 'UDP', 'UDP', 'OTHER', 'OTHER'])

    def test_ημερομηνια_δεν_χανει_ακριβεια(self):
        # Ο έλεγχος για το float32: το 20251203 δεν χωράει σε 24 bits mantissa.
        frame = pd.DataFrame({'Time': ['2025-12-03 23:59:36.210',
                                       '2025-12-02 00:00:01.000']})
        out, invalid = add_time_columns(frame)
        self.assertEqual(invalid, 0)
        self.assertEqual(list(out['Date']), [20251203, 20251202])
        self.assertEqual(list(out['Hour']), [23, 0])
        self.assertEqual(out['Date'].dtype.name, 'Int32')

    def test_ολα_τα_χρονικα_ακυρα_δινει_σφαλμα(self):
        with self.assertRaises(ValueError):
            add_time_columns(pd.DataFrame({'Time': ['οχι ημερομηνια', 'ουτε αυτο']}))

    def test_καθε_ροη_μετραει_και_για_τα_δυο_ακρα(self):
        frame = pd.DataFrame({'Src_Host': ['a'], 'Dst_Host': ['b'],
                              'Date': [20251127], 'Hour': [10],
                              'DayOfWeek': [3], 'Bytes': [500]})
        table = host_time_table(frame)
        self.assertEqual(set(table['Host']), {'a', 'b'})
        self.assertEqual(list(table['Flows']), [1, 1])
        self.assertEqual(list(table['Bytes']), [500, 500])


class TestDedup(unittest.TestCase):

    def test_ταυτοσημα_σημεια_παιρνουν_βαρος(self):
        X = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
        uniq, inverse, weights = deduplicate(X)
        self.assertEqual(len(uniq), 2)
        self.assertEqual(sorted(weights.tolist()), [1.0, 3.0])
        # Το inverse πρέπει να ανασυνθέτει το αρχικό σύνολο.
        np.testing.assert_allclose(uniq[inverse], X)

    def test_min_samples_κλιμακωνεται_και_φρεναρει(self):
        self.assertEqual(resolve_min_samples(64), 5)        # κάτω όριο
        self.assertEqual(resolve_min_samples(121_222), 182)  # αναλογικό
        self.assertEqual(resolve_min_samples(10_000_000), 500)  # άνω όριο


class TestAlignment(unittest.TestCase):
    """Η αντιστοίχιση ετικετών: εδώ ζούσε ένα πραγματικό σφάλμα."""

    def test_αντιστρεφει_μεταθεση(self):
        hosts = [f'h{i}' for i in range(6)]
        reference = pd.Series([0, 0, 1, 1, 2, 2], index=hosts)
        # Ίδια ομαδοποίηση, άλλη αρίθμηση: 0->2, 1->0, 2->1
        day = pd.Series([2, 2, 0, 0, 1, 1], index=hosts)
        translated, common, agreement, _ = align_day(day, reference)
        self.assertEqual(common, 6)
        self.assertEqual(agreement, 1.0)
        self.assertEqual(list(translated), list(reference))

    def test_δεν_παραγονται_ετικετες_εκτος_αναφορας(self):
        # ΤΟ ΣΦΑΛΜΑ, στην πραγματική του μορφή.
        #
        # Ο πίνακας επικάλυψης παίρνει στήλες μόνο από τις ετικέτες που
        # εμφανίζονται στους ΚΟΙΝΟΥΣ hosts. Εδώ η ομάδα 0 της αναφοράς ζει
        # μόνο στον h5, που λείπει από την ημέρα, οπότε χωρίς το reindex ο
        # πίνακας βγαίνει 3x2: τρεις ομάδες ημέρας, δύο στήλες. Η τρίτη μένει
        # αταίριαστη και παίρνει «εφεδρικό» αριθμό max+1 = 3 — ετικέτα που
        # ΔΕΝ υπάρχει στη συνολική ομαδοποίηση. Το αποτέλεσμα είναι ομάδα
        # φάντασμα στα γραφήματα σύνθεσης και στους πίνακες συμφωνίας, ενώ
        # η σωστή απάντηση ήταν να αντιστοιχιστεί στην ομάδα 0.
        reference = pd.Series({'h0': 1, 'h1': 1, 'h2': 2,
                               'h3': 2, 'h4': 2, 'h5': 0})
        day = pd.Series({'h0': 0, 'h1': 0, 'h2': 1, 'h3': 1, 'h4': 2})

        translated, _, _, _ = align_day(day, reference)
        self.assertEqual(translated.isna().sum(), 0)
        self.assertTrue(
            set(translated.dropna()).issubset(set(reference.unique())),
            f"ετικέτες εκτός αναφοράς: {set(translated.dropna())} "
            f"έναντι {set(reference.unique())}")

    def test_επιπλεον_ομαδα_ημερας_παιρνει_νεο_αριθμο(self):
        hosts = [f'h{i}' for i in range(4)]
        reference = pd.Series([0, 0, 1, 1], index=hosts)
        day = pd.Series([0, 1, 2, 2], index=hosts)
        translated, _, _, _ = align_day(day, reference)
        self.assertEqual(translated.isna().sum(), 0)
        # Καμία ετικέτα δεν χάνεται με το να στριμωχτεί πάνω σε άλλη.
        self.assertEqual(len(set(translated)), 3)


class TestRecurrence(unittest.TestCase):

    def test_λογος_επαναληψης(self):
        frame = pd.DataFrame({
            'Host': ['a', 'a', 'a', 'b', 'b'],
            'Service_Port': [80, 80, 443, 22, 23],
            'Ημέρα': [1, 2, 1, 1, 2],
        })
        table = recurrence(frame, 'Service_Port')
        # Ο 'a' είδε δύο θύρες, η μία σε δύο ημέρες -> 1/2
        self.assertEqual(table.loc['a', 'Πλήθος'], 2)
        self.assertAlmostEqual(table.loc['a', 'Λόγος_Επανάληψης'], 0.5)
        # Ο 'b' είδε δύο θύρες, καμία δύο φορές -> 0
        self.assertAlmostEqual(table.loc['b', 'Λόγος_Επανάληψης'], 0.0)

    def test_ποσοστο_διαδοχικων_θυρων(self):
        frame = pd.DataFrame({
            'Host': ['scan'] * 4 + ['service'] * 3,
            'Service_Port': [80, 81, 82, 83, 22, 443, 8080],
            'Ημέρα': [1] * 7,
        })
        shares = consecutive_share(frame, {'scan', 'service'})
        self.assertAlmostEqual(shares['scan'], 1.0)     # 80,81,82,83
        self.assertAlmostEqual(shares['service'], 0.0)  # διάσπαρτες

    def test_ενα_μονο_στοιχειο_δεν_οριζεται(self):
        frame = pd.DataFrame({'Host': ['x'], 'Service_Port': [80], 'Ημέρα': [1]})
        shares = consecutive_share(frame, {'x'})
        self.assertTrue(np.isnan(shares['x']))


class TestPortSemantics(unittest.TestCase):
    """Το Service_Port == 0: πρωτόκολλο χωρίς θύρες, όχι θύρα υπηρεσίας."""

    def setUp(self):
        self.folder = tempfile.mkdtemp()
        pd.DataFrame({
            # δύο ροές ICMP/IGMP (θύρα 0 και στις δύο πλευρές), που ΔΕΝ πρέπει
            # να μετρηθούν ως «δύο καταχωρημένες θύρες»
            'Src_Port': [0, 0, 443, 50000, 22],
            'Dst_Port': [0, 0, 51000, 52000, 80],
        }).to_csv(os.path.join(self.folder, 'clean_20251127.csv'), index=False)

    def tearDown(self):
        shutil.rmtree(self.folder, ignore_errors=True)

    def test_οι_ροες_χωρις_θυρες_μετριουνται_χωριστα(self):
        # ΤΟ ΣΦΑΛΜΑ: με fillna(-1) και σκέτο «< 1024», η θύρα 0 του ICMP/IGMP
        # περνούσε ως καταχωρημένη και οι δύο αυτές ροές έμπαιναν στο «δύο
        # γνωστές θύρες», φουσκώνοντας ακριβώς τον αριθμό που μπαίνει στην
        # αναφορά. Εδώ πρέπει να βγουν σε δική τους γραμμή.
        with mock.patch.dict(port_semantics.PATHS, {'processed_data': self.folder}):
            table, total, with_ports = port_pair_cases()

        self.assertEqual(total, 5)
        self.assertEqual(with_ports, 3)
        counts = dict(zip(table['Περίπτωση'], table['Ροές']))
        self.assertEqual(counts[NO_PORTS], 2)
        self.assertEqual(counts['Δύο γνωστές θύρες'], 1)   # 22 / 80
        self.assertEqual(counts['Μία γνωστή θύρα'], 1)     # 443 / 51000
        self.assertEqual(counts['Καμία γνωστή θύρα'], 1)   # 50000 / 52000

    def test_τα_ποσοστα_ειναι_επι_των_ροων_με_θυρες(self):
        # Αλλιώς τα τρία ποσοστά δεν αθροίζουν σε 1 και ο πίνακας της
        # αναφοράς διαβάζεται λάθος.
        with mock.patch.dict(port_semantics.PATHS, {'processed_data': self.folder}):
            table, total, with_ports = port_pair_cases()
        rest = table[table['Περίπτωση'] != NO_PORTS]
        # places=3 και όχι 6: τα ποσοστά στρογγυλοποιούνται σε 4 δεκαδικά για
        # το CSV, οπότε τρία τρίτα δίνουν 0,9999 και όχι ακριβώς 1.
        self.assertAlmostEqual(float(rest['Ποσοστό'].sum()), 1.0, places=3)
        portless = table.loc[table['Περίπτωση'] == NO_PORTS, 'Ποσοστό'].iloc[0]
        self.assertAlmostEqual(float(portless), 2 / 5, places=6)


class TestHourly(unittest.TestCase):

    def test_τεραστιος_host_δεν_κυριαρχει_στην_ομαδα(self):
        # Η κανονικοποίηση ανά host: αλλιώς η καμπύλη της ομάδας θα ήταν η
        # καμπύλη της πύλης, που έχει 18,3 εκατ. ροές.
        rows = []
        for hour in range(24):
            rows.append(('flat', hour, 10))
            rows.append(('GIANT', hour, 100_000))
        rows.append(('spiky', 9, 1_000))  # ένας host με όλη την κίνηση στις 09:00
        frame = pd.DataFrame(rows, columns=['Host', 'Hour', 'Flows'])
        labels = pd.Series({'flat': 0, 'GIANT': 0, 'spiky': 0})

        curve, hosts = hourly_shares(frame, labels)
        self.assertEqual(len(hosts), 3)
        # Δύο επίπεδοι (1/24 ο καθένας) και ένας με όλα στις 09:00:
        # μέσος όρος στις 09:00 = (1/24 + 1/24 + 1)/3
        self.assertAlmostEqual(curve.loc[0, 9], (2 / 24 + 1) / 3, places=9)
        # Αν κυριαρχούσε ο GIANT, η καμπύλη θα ήταν επίπεδη στο 1/24.
        self.assertGreater(curve.loc[0, 9], 0.3)
        # Κάθε καμπύλη ομάδας αθροίζει σε 1.
        self.assertAlmostEqual(float(curve.loc[0].sum()), 1.0, places=9)

    def test_μερικη_ημερα_εντοπιζεται_απο_τα_δεδομενα(self):
        frame = pd.DataFrame({
            'Host': ['a'] * 4,
            'Date': [20251126, 20251126, 20251127, 20251127],
            'Hour': [9, 23, 0, 23],
            'DayOfWeek': [2, 2, 3, 3],
            'Flows': [1, 1, 1, 1],
        })
        found, _ = partial_days(frame)
        self.assertIn(20251126, found)
        self.assertNotIn(20251127, found)


class TestSubnets(unittest.TestCase):
    """Το στάδιο των ιδιωτικών υποδικτύων: ψευδωνυμοποίηση και μέτρηση ροών."""

    def test_οι_διευθυνσεις_εκπομπης_δεν_ειναι_ουτε_εσωτερικες_ουτε_εξωτερικες(self):
        # ΔΕΣΜΕΥΤΙΚΟ. Οι δύο πληθυσμοί δεν είναι συμπληρωματικοί, επίτηδες:
        # μια διεύθυνση εκπομπής δεν είναι μηχανή για να ομαδοποιηθεί, ούτε
        # host του διαδικτύου για να μετρηθεί ως ανωμαλία. Όσο ήταν
        # συμπληρωματικοί, η 192.168.7.255 ήταν μία από τις 1.847 ανωμαλίες
        # του Isolation Forest.
        internal, network = get_internal_hashes(), get_network_hashes()
        for prefix in SUBNET_PREFIXES:
            for octet in (0, 255):
                h = anonymize_ip(f'{prefix}{octet}')
                self.assertNotIn(h, internal, f'{prefix}{octet}')
                self.assertIn(h, network, f'{prefix}{octet}')

        frame = pd.DataFrame({'Host': [
            anonymize_ip(f'{INTERNAL_SUBNET}5'),      # host
            anonymize_ip(f'{INTERNAL_SUBNET}255'),    # εκπομπή
            anonymize_ip('8.8.8.8'),                  # διαδίκτυο
        ]})
        inside, outside = split_internal_external(frame, other_private=frozenset())
        self.assertEqual(len(inside), 1)
        self.assertEqual(len(outside), 1)
        broadcast = anonymize_ip(f'{INTERNAL_SUBNET}255')
        self.assertNotIn(broadcast, set(inside['Host']))
        self.assertNotIn(broadcast, set(outside['Host']))

    def test_get_subnet_hashes_ειναι_προς_τα_εμπρος(self):
        # Δεν αντιστρέφει hashes: τα ξαναϋπολογίζει με την ΙΔΙΑ συνάρτηση που
        # ανωνυμοποίησε τα δεδομένα. Αν κάποτε αλλάξει η anonymize_ip, τα δύο
        # πρέπει να αλλάξουν μαζί, αλλιώς το στάδιο δεν βρίσκει κανέναν host.
        hashes = get_subnet_hashes('192.168.8.')
        self.assertEqual(len(hashes), 254)
        self.assertIn(anonymize_ip('192.168.8.195'), hashes)
        self.assertNotIn(anonymize_ip('192.168.9.195'), hashes)

    def test_οι_διευθυνσεις_εκπομπης_δεν_ειναι_hosts(self):
        # ΔΕΣΜΕΥΤΙΚΟ, και το ίδιο εύρος για ΟΛΑ τα υποδίκτυα. Με 0-255 στα
        # υποδίκτυα και 1-254 στο εργαστήριο, η 192.168.8.255 ομαδοποιούνταν
        # ως ένας από τους 26 «hosts» ενώ η 192.168.7.255 όχι. Και δεν είναι
        # ακίνδυνο: η .255 έχει Out_Ratio 0, TCP_Ratio 0, 36 bytes ανά ροή και
        # 562.975 ροές, δηλαδή ακραίο σημείο και το 10% του υποδικτύου.
        for prefix in ('192.168.7.', '192.168.8.', '192.168.1.'):
            hashes = get_subnet_hashes(prefix)
            self.assertEqual(len(hashes), 254, prefix)
            self.assertNotIn(anonymize_ip(f'{prefix}0'), hashes, prefix)
            self.assertNotIn(anonymize_ip(f'{prefix}255'), hashes, prefix)
        # Η ΕΤΙΚΕΤΑ όμως πρέπει να τις καλύπτει: μια εκπομπή προς το
        # 192.168.8.255 είναι κίνηση προς το 192.168.8.x, και ο πίνακας
        # επαφών θα την έχανε αλλιώς.
        labelled = label_subnets(pd.Series([anonymize_ip('192.168.8.255')]),
                                 ['192.168.8.'])
        self.assertEqual(labelled.iloc[0], '192.168.8.x')

    def test_οι_ετικετες_δεν_αποκαλυπτουν_το_τελευταιο_οκτετο(self):
        # Η ψευδωνυμοποίηση σπάει αν ένα CSV των αποτελεσμάτων γράψει
        # «192.168.8.195». Η ετικέτα είναι του /24 και τίποτα παρακάτω: αυτό
        # που χρειάζεται η ανάλυση είναι η ομάδα, όχι η διεύθυνση.
        mapping = subnet_labels(['192.168.8.'])
        self.assertEqual(set(mapping.values()), {'192.168.8.x'})
        self.assertEqual(len(mapping), 256)

        hosts = pd.Series([anonymize_ip('192.168.8.195'),
                           anonymize_ip('192.168.1.4'),
                           anonymize_ip('8.8.8.8')])
        labels = label_subnets(hosts, ['192.168.8.', '192.168.1.'])
        self.assertEqual(labels.iloc[0], '192.168.8.x')
        self.assertEqual(labels.iloc[1], '192.168.1.x')
        self.assertTrue(pd.isna(labels.iloc[2]))   # το διαδίκτυο μένει εκτός
        self.assertNotIn('195', str(labels.iloc[0]))

    def test_το_εσωτερικο_μπαινει_πρωτο_και_μια_φορα(self):
        # Το INTERNAL_SUBNET είναι η ομάδα αναφοράς και δεν γράφεται στο
        # SUBNET_PREFIXES. Αν κάποιος το γράψει και εκεί, δεν πρέπει να
        # εμφανιστεί δύο φορές (διπλή στήλη στον σταυροπίνακα).
        with mock.patch.object(subnet_analysis, 'SUBNET_PREFIXES',
                               [INTERNAL_SUBNET, '192.168.8.']):
            found = subnet_analysis.prefixes()
        self.assertEqual(found, [INTERNAL_SUBNET, '192.168.8.'])

    def test_οι_ροες_εντος_υποδικτυου_μετριουνται_μια_φορα(self):
        # ΤΟ ΣΦΑΛΜΑ, στην πραγματική του μορφή: το μερίδιο ροών των
        # υποδικτύων είχε βγει 31% από άθροισμα των Total_Flows ανά host.
        # Κάθε ροή με ΚΑΙ ΤΑ ΔΥΟ άκρα μέσα στο ίδιο υποδίκτυο μετριέται δύο
        # φορές έτσι, και το σωστό ήταν 28,8%. Εδώ ο ίδιος πίνακας πρέπει να
        # μετρήσει τις 100 ενδοϋποδικτυακές ροές μία φορά, όχι διακόσιες.
        frame = pd.DataFrame({
            'Α': ['192.168.8.x', '192.168.8.x'],
            'Β': ['192.168.8.x', '192.168.7.x'],
            'Ροές': [100.0, 10.0],
            'Μπάιτ': [1000.0, 100.0],
        })
        folder = tempfile.mkdtemp()
        try:
            with mock.patch.object(subnet_analysis, 'BASE_DIR', folder):
                table = write_peers(frame)
        finally:
            shutil.rmtree(folder, ignore_errors=True)

        eight = table[table['Υποδίκτυο'] == '192.168.8.x']
        self.assertEqual(float(eight['Ροές'].sum()), 110.0)
        intra = eight[eight['Μιλά_με'] == '192.168.8.x']
        self.assertEqual(float(intra['Ροές'].iloc[0]), 100.0)
        # Τα ποσοστά του υποδικτύου αθροίζουν σε 100, αλλιώς ο πίνακας
        # της αναφοράς διαβάζεται λάθος.
        self.assertAlmostEqual(float(eight['Ποσοστό_του_υποδικτύου_%'].sum()),
                               100.0, places=2)


    def test_καθε_υποδικτυο_ειναι_πληθυσμος_με_δικο_του_κλειδι(self):
        # Το κλειδί γίνεται πρόθεμα αρχείων (subnet_192_168_8_global.csv) και
        # level στα στάδια kselect, micro και stability. Αν αλλάξει η μορφή
        # του, τα στάδια ψάχνουν αρχεία που δεν γράφτηκαν ποτέ.
        self.assertEqual(pops.key_of('192.168.8.'), 'subnet_192_168_8')
        with mock.patch.object(pops, 'SUBNET_PREFIXES', ['192.168.8.']):
            self.assertEqual(pops.extra_levels(),
                             [('subnet_192_168_8', 'Υποδίκτυο 192.168.8.x')])

    def test_καθε_υποδικτυο_εχει_δικο_του_level_και_το_εργαστηριο_μαζι(self):
        # Το 'micro' είναι το ΕΣΩΤΕΡΙΚΟ ΔΙΚΤΥΟ και σύντομα θα περιέχει και τα
        # τρία υποδίκτυα μαζί. Αν το εργαστήριο δεν είχε και δικό του
        # subnet_<x> level, η ξεχωριστή ανάλυση του 192.168.7.x θα
        # εξαφανιζόταν σιωπηλά από τα kselect, stability και visualize τη
        # μέρα που θα άλλαζε η σημασία του 'micro'.
        with mock.patch.object(pops, 'SUBNET_PREFIXES',
                               [INTERNAL_SUBNET, '192.168.8.', '192.168.1.']):
            keys = [k for k, _ in pops.extra_levels()]
            subnets = [k for k, _, _ in pops.subnet_populations()]
        self.assertEqual(keys, ['subnet_192_168_7', 'subnet_192_168_8',
                                'subnet_192_168_1'])
        self.assertEqual(subnets, keys)

    def test_το_εργαστηριο_μπαινει_πρωτο_και_κραταει_το_κλειδι_micro(self):
        # ΔΕΣΜΕΥΤΙΚΟ: τα αρχεία του εργαστηρίου λέγονται micro_global.csv και
        # τα διαβάζουν τα hourly, stability και visualize. Αν το κλειδί του
        # γινόταν subnet_192_168_7, όλα αυτά θα έψαχναν αρχείο που δεν
        # υπάρχει — σιωπηλά, αφού ελέγχουν την ύπαρξη και παραλείπουν.
        with mock.patch.object(pops, 'SUBNET_PREFIXES', ['192.168.8.']):
            found = pops.populations()
        self.assertEqual(found[0][0], 'micro')
        self.assertEqual([k for k, _, _ in found], ['micro', 'subnet_192_168_8'])

    def test_το_εργαστηριο_υπαρχει_και_ως_δικτυο_και_ως_υποδικτυο(self):
        # ΔΥΟ ΑΞΟΝΕΣ, όχι διπλοεγγραφή: το 'micro' είναι το εσωτερικό δίκτυο
        # και το 'subnet_192_168_7' το ίδιο το /24. Σήμερα τα δύο σύνολα
        # ταυτίζονται και τα αρχεία βγαίνουν ίδια· όταν το εσωτερικό δίκτυο
        # γίνει η ένωση των τριών, αποκλίνουν και η ξεχωριστή ανάλυση του
        # εργαστηρίου συνεχίζει να υπάρχει χωρίς άλλη αλλαγή.
        with mock.patch.object(pops, 'SUBNET_PREFIXES', [INTERNAL_SUBNET, '192.168.8.']):
            keys = [k for k, _, _ in pops.populations()]
        self.assertEqual(keys, ['micro', 'subnet_192_168_7', 'subnet_192_168_8'])

    def test_ο_διαχωρισμος_χωρις_ορισμα_μενει_ο_ιδιος(self):
        # ΕΛΕΓΧΟΣ ΠΑΛΙΝΔΡΟΜΗΣΗΣ της παραμετροποίησης: το split_internal_external
        # δέχεται πλέον hashes, αλλά χωρίς όρισμα πρέπει να δίνει ΑΚΡΙΒΩΣ ό,τι
        # έδινε πριν, αλλιώς αλλάζουν σιωπηλά όλα τα macro και micro.
        frame = pd.DataFrame({
            'Host': [anonymize_ip(f'{INTERNAL_SUBNET}5'),
                     anonymize_ip(f'{INTERNAL_SUBNET}6'),
                     anonymize_ip('8.8.8.8')],
        })
        internal, external = split_internal_external(frame, other_private=frozenset())
        self.assertEqual(len(internal), 2)
        self.assertEqual(len(external), 1)
        explicit, _ = split_internal_external(frame, get_internal_hashes(),
                                              other_private=frozenset())
        pd.testing.assert_frame_equal(internal, explicit)

    def test_file_label(self):
        self.assertEqual(file_label('192.168.8.'), '192_168_8')
        self.assertEqual(file_label('192.168.8'), '192_168_8')


class TestKConfig(unittest.TestCase):
    """
    Το k κάθε πληθυσμού και η συνέπεια του διαχωρισμού.

    Γιατί υπάρχει αυτή η κλάση: τα στάδια ρωτούν για το k όταν τρέχουν σε
    τερματικό και αλλιώς παίρνουν σιωπηλά την προεπιλογή. Όσο η προεπιλογή
    ήταν ένα MICRO_CLUSTERS για όλους, ένα `py main.py --step all > log.txt`
    έτρεχε με άλλα k από αυτά της αναφοράς χωρίς να το πει πουθενά. Δεν είναι
    σφάλμα που σκάει, είναι σφάλμα που βγάζει διαφορετικά αποτελέσματα
    σιωπηλά — άρα το πιάνει μόνο έλεγχος.
    """

    def test_καθε_πληθυσμος_εχει_δικο_του_k(self):
        from src.config import K_PER_LEVEL
        from src.models.populations import populations

        levels = ['macro'] + [key for key, _, _ in populations()]
        missing = [level for level in levels if level not in K_PER_LEVEL]
        self.assertEqual(
            missing, [],
            f"Πληθυσμοί χωρίς ρητό k στο K_PER_LEVEL: {missing}. Θα έτρεχαν "
            f"με το MICRO_CLUSTERS, δηλαδή με k που δεν τεκμηριώνεται πουθενά.")

    def test_τα_k_ειναι_μεσα_στο_ευρος_που_δοκιμαζεται(self):
        # Ένα k έξω από το εύρος του kselect δεν υπάρχει στον πίνακα των
        # δεικτών, οπότε η choose_k το απορρίπτει και ξαναρωτά — ή, σε μη
        # διαδραστική εκτέλεση, τρέχει με νούμερο που κανένα κριτήριο δεν
        # αξιολόγησε.
        from src.config import K_PER_LEVEL
        from src.models.select_k import MACRO_K_RANGE, MICRO_K_RANGE

        for level, k in K_PER_LEVEL.items():
            allowed = MACRO_K_RANGE if level == 'macro' else MICRO_K_RANGE
            with self.subTest(level=level):
                self.assertIn(k, list(allowed),
                              f"Το k={k} του '{level}' είναι έξω από το εύρος "
                              f"που δοκιμάζει το στάδιο kselect.")

    def test_η_μη_διαδραστικη_εκτελεση_παιρνει_τα_k_της_αναφορας(self):
        # Η default_k είναι το «Enter» της ερώτησης και η τιμή που ισχύει
        # χωρίς τερματικό. Πρέπει να συμφωνεί με το config σε κάθε level,
        # αλλιώς υπάρχουν δύο προεπιλογές και μετράει η κρυφή.
        from src.config import K_PER_LEVEL
        from src.models.populations import default_k
        from src.models.select_k import LEVELS

        for level, k in K_PER_LEVEL.items():
            with self.subTest(level=level):
                self.assertEqual(default_k(level), k)
                self.assertEqual(LEVELS[level][1], k)

    def test_εξοδος_σε_αρχειο_δεν_ρωταει(self):
        # `py main.py --step all > log.txt` από κονσόλα: η είσοδος ΕΙΝΑΙ
        # τερματικό, η έξοδος όχι. Ο παλιός έλεγχος κοίταζε μόνο την είσοδο,
        # οπότε η ερώτηση γραφόταν μέσα στο log.txt και το πρόγραμμα περίμενε
        # σιωπηλά απάντηση που κανείς δεν έβλεπε να ζητείται.
        from src.config import K_PER_LEVEL
        from src.models import select_k

        fake_sys = mock.MagicMock()
        fake_sys.stdin.isatty.return_value = True
        fake_sys.stdout.isatty.return_value = False
        table = pd.DataFrame({'k': [2, 3, 4], 'Inertia': [9.0, 5.0, 4.0],
                              'Silhouette': [0.1, 0.3, 0.2],
                              'Calinski_Harabasz': [10.0, 12.0, 11.0],
                              'Davies_Bouldin': [1.2, 0.9, 1.0],
                              'Gap': [0.5, 0.7, 0.71], 'Gap_s': [0.05] * 3,
                              'Stability_ARI': [0.8, 0.9, 0.7]})

        with mock.patch.object(select_k, 'sys', fake_sys), \
                mock.patch.object(select_k, 'load_k_table', return_value=table), \
                mock.patch.object(select_k, 'ask',
                                  side_effect=AssertionError('ρώτησε ενώ η έξοδος είναι αρχείο')), \
                contextlib.redirect_stdout(io.StringIO()):
            chosen = select_k.choose_k('macro')

        self.assertEqual(chosen, K_PER_LEVEL['macro'])
        self.assertEqual(select_k.K_USED['macro']['source'], 'config')

    def test_το_εσωτερικο_υποδικτυο_ειναι_παραγωγο_της_λιστας(self):
        # Ήταν δύο ανεξάρτητες ρυθμίσεις και μπορούσαν να αποκλίνουν: αλλαγή
        # της SUBNET_PREFIXES χωρίς αλλαγή του INTERNAL_SUBNET έδινε στο
        # στάδιο subnets ομάδα αναφοράς εκτός ανάλυσης, χωρίς καμία ένδειξη.
        from src.models.subnet_analysis import prefixes

        self.assertEqual(INTERNAL_SUBNET, SUBNET_PREFIXES[0])
        self.assertEqual(prefixes(), list(SUBNET_PREFIXES))


class TestPrivateScan(unittest.TestCase):
    """
    Ο εντοπισμός των ιδιωτικών διευθύνσεων RFC1918 στα προφίλ.

    Τα hashes δεν αντιστρέφονται, οπότε η μόνη απάντηση στο «είναι ιδιωτική;»
    είναι το forward hash κάθε διεύθυνσης του RFC1918. Το βήμα αυτό το κάνει
    μία φορά και γράφει κατάλογο. Αυτό που ελέγχεται εδώ είναι ότι ο κατάλογος
    καλύπτει και τα τρία μπλοκ, ότι δεν διαρρέει το τελευταίο οκτέτο, και ότι
    οι διευθύνσεις που βρίσκει βγαίνουν πράγματι από τον εξωτερικό πληθυσμό.
    """

    def test_καλυπτονται_και_τα_τρια_μπλοκ_του_rfc1918(self):
        # Η πρώτη σάρωση κάλυπτε μόνο 192.168/16 και 172.16/12, και το 10/8
        # περιείχε μόνο του 56 διευθύνσεις με περισσότερες ροές από τις άλλες
        # δύο ζώνες μαζί. Ο έλεγχος κλειδώνει ότι δεν θα ξαναλείψει μπλοκ.
        from src.utils.private_scan import private_prefixes, TOTAL_PREFIXES

        seen = set()
        count = 0
        for prefix in private_prefixes():
            count += 1
            seen.add(prefix.split('.')[0])
        self.assertEqual(count, TOTAL_PREFIXES)
        self.assertEqual(count, 65536 + 4096 + 256)
        self.assertEqual(seen, {'10', '172', '192'})

    def test_η_σαρωση_βρισκει_οσες_υπαρχουν_και_μονο_αυτες(self):
        from src.utils.private_scan import scan

        mesa = ['10.1.2.3', '172.20.5.6', '192.168.44.7']
        exo = ['8.8.8.8', '1.1.1.1']
        hosts = {anonymize_ip(ip) for ip in mesa + exo}
        # Τα προθέματα δίνονται ρητά: η πληρότητα του RFC1918 ελέγχεται
        # χωριστά, εδώ ελέγχεται η αντιστοίχιση.
        prefixes = ['10.1.2.', '172.20.5.', '192.168.44.', '10.9.9.']

        found = scan(hosts, prefixes=prefixes)
        self.assertEqual(len(found), len(mesa))
        self.assertEqual(set(found.values()),
                         {'10.1.2.x', '172.20.5.x', '192.168.44.x'})
        for ip in exo:
            self.assertNotIn(anonymize_ip(ip), found)

    def test_ο_καταλογος_δεν_αποκαλυπτει_το_τελευταιο_οκτετο(self):
        from src.utils.private_scan import scan

        hosts = {anonymize_ip('10.1.2.200')}
        labels = list(scan(hosts, prefixes=['10.1.2.']).values())
        self.assertEqual(labels, ['10.1.2.x'])
        self.assertNotIn('200', labels[0])

    def test_οι_ιδιωτικες_αλλων_υποδικτυων_βγαινουν_απο_το_macro(self):
        from src.models.data_split import split_internal_external

        xeni = anonymize_ip('10.1.2.3')
        frame = pd.DataFrame({'Host': [
            anonymize_ip(f'{INTERNAL_SUBNET}5'),   # εσωτερικός
            anonymize_ip('8.8.8.8'),               # εξωτερικός
            xeni,                                  # ιδιωτική αλλού
        ]})
        internal, external = split_internal_external(
            frame, other_private=frozenset([xeni]))
        self.assertEqual(len(internal), 1)
        self.assertEqual(len(external), 1)
        self.assertNotIn(xeni, set(external['Host']))
        self.assertNotIn(xeni, set(internal['Host']))

    def test_χωρις_καταλογο_ο_διαχωρισμος_μενει_ο_παλιος(self):
        # ΕΛΕΓΧΟΣ ΠΑΛΙΝΔΡΟΜΗΣΗΣ: με ρητό internal_hashes ο καλών έχει ορίσει
        # μόνος του τι θεωρεί δικό του, οπότε δεν αφαιρούνται ιδιωτικές και το
        # αποτέλεσμα είναι ακριβώς ό,τι ήταν πριν το βήμα privatescan.
        from src.models.data_split import split_internal_external
        from src.utils.anonymize import get_internal_hashes

        frame = pd.DataFrame({'Host': [
            anonymize_ip(f'{INTERNAL_SUBNET}5'),
            anonymize_ip('10.1.2.3'),
        ]})
        internal, external = split_internal_external(frame, get_internal_hashes())
        self.assertEqual(len(internal), 1)
        self.assertEqual(len(external), 1)

class TestKPlots(unittest.TestCase):
    """
    Τα γραφήματα των δεικτών εγκυρότητας.

    Δεν ελέγχεται πώς φαίνεται μια εικόνα — αυτό δεν ελέγχεται με κώδικα. Αυτό
    που ελέγχεται είναι τα δύο σημεία όπου ένα γράφημα μπορεί να πει ψέματα:
    ποιους πληθυσμούς σχεδιάζει (αν η λίστα είναι γραμμένη με το χέρι, μένει
    πίσω και βγάζει ορφανές εικόνες — ακριβώς αυτό είχε συμβεί) και ποιο k
    δηλώνει ως επιλεγμένο (αν δεν είναι το ίδιο με του config, το γράφημα
    δείχνει άλλη απόφαση από εκείνη που εκτελέστηκε).
    """

    def test_σχεδιαζονται_οσοι_πληθυσμοι_εχουν_πινακα(self):
        from src.visualizations.plot_k import levels_found

        with tempfile.TemporaryDirectory() as folder:
            for key in ('micro', 'macro', 'subnet_192_168_7', 'ο_νεος'):
                with open(os.path.join(folder, f'k_selection_{key}.csv'), 'w',
                          encoding='utf-8') as handle:
                    handle.write('k,Silhouette\n2,0.5\n')
            found = levels_found(folder)

        # Ό,τι υπάρχει στον φάκελο, με τη σειρά των LEVELS και τα άγνωστα στο
        # τέλος: ένας καινούργιος πληθυσμός σχεδιάζεται χωρίς να πειραχτεί
        # κώδικας εδώ.
        self.assertEqual(found, ['macro', 'micro', 'subnet_192_168_7', 'ο_νεος'])

    def test_το_γραφημα_δηλωνει_το_k_που_εκτελεστηκε(self):
        # Χωρίς K_USED και χωρίς αρχεία αποτελεσμάτων: το k του config. Τα
        # αρχεία μπλοκάρονται ρητά, αλλιώς το αποτέλεσμα εξαρτάται από το αν
        # έχει τρέξει το pipeline στον φάκελο.
        from src.visualizations.plot_k import chosen_k
        from src.models import daily_stability, select_k
        from src.config import K_PER_LEVEL

        with mock.patch.dict(select_k.K_USED, {}, clear=True), \
                mock.patch.object(daily_stability, 'reference_labels', return_value=None):
            for level, expected in K_PER_LEVEL.items():
                k, source = chosen_k(level)
                self.assertEqual(k, expected, f'λάθος k στο γράφημα του {level}')
                self.assertEqual(source, 'config')

    def test_χωρις_k_used_το_γραφημα_παιρνει_το_k_του_αρχειου(self):
        # Αν το micro έτρεξε σε άλλη διεργασία, το k των ετικετών στον δίσκο
        # είναι αυτό που εκτελέστηκε, όχι απαραίτητα του config.
        from src.visualizations.plot_k import chosen_k
        from src.models import daily_stability, select_k

        reference = pd.Series([0, 1, 2, 3, 0], index=list('abcde'))
        with mock.patch.dict(select_k.K_USED, {}, clear=True), \
                mock.patch.object(daily_stability, 'reference_labels', return_value=reference):
            self.assertEqual(chosen_k('micro'), (4, 'αρχείο αποτελεσμάτων'))

    def test_επιπεδο_χωρις_επιλεγμενο_k_δεν_εφευρισκει_ενα(self):
        from src.visualizations.plot_k import chosen_k

        # Το macro_dedup είναι έλεγχος πάνω στον macro, όχι πληθυσμός με δική
        # του ομαδοποίηση. Αν το γράφημα τού έβαζε γραμμή σε κάποιο k, θα
        # ισχυριζόταν απόφαση που δεν πάρθηκε ποτέ.
        self.assertEqual(chosen_k('macro_dedup'), (None, None))

class TestAblation(unittest.TestCase):
    """
    Το ablation συγκρίνει οικογένειες χαρακτηριστικών μεταξύ τους.

    Η σύγκριση έχει νόημα μόνο αν οι οικογένειες είναι ΙΣΕΣ και καλύπτουν
    ΑΚΡΙΒΩΣ τα FEATURE_COLUMNS. Αν προστεθεί χαρακτηριστικό στο config και
    ξεχαστεί εδώ, το πείραμα θα έτρεχε και θα έβγαζε πίνακα — απλώς η «πλήρης»
    παραλλαγή δεν θα ήταν πλήρης και κάθε σύγκριση θα ήταν λάθος σιωπηλά.
    """

    def test_οι_οικογενειες_καλυπτουν_ακριβως_τα_χαρακτηριστικα(self):
        from src.models.ablation import GROUPS, check_groups
        from src.config import FEATURE_COLUMNS

        check_groups()  # σκάει μόνο του αν κάτι λείπει ή περισσεύει
        mine = [name for names in GROUPS.values() for name in names]
        self.assertEqual(sorted(mine), sorted(FEATURE_COLUMNS))

    def test_οι_οικογενειες_ειναι_ισομεγεθεις(self):
        from src.models.ablation import GROUPS

        # Αλλιώς το «χωρίς τα χρονικά χαλάει περισσότερο» μπορεί να σημαίνει
        # απλώς ότι αφαιρέθηκαν περισσότερες διαστάσεις.
        sizes = {name: len(names) for name, names in GROUPS.items()}
        self.assertEqual(len(set(sizes.values())), 1, f'άνισες οικογένειες: {sizes}')

    def test_καθε_παραλλαγη_εχει_το_σωστο_πληθος(self):
        from src.models.ablation import GROUPS, variants
        from src.config import FEATURE_COLUMNS

        size = len(next(iter(GROUPS.values())))
        got = {name: len(columns) for name, columns in variants()}
        self.assertEqual(got['όλα'], len(FEATURE_COLUMNS))
        for family in GROUPS:
            self.assertEqual(got[f'μόνο {family}'], size)
            self.assertEqual(got[f'χωρίς {family}'], len(FEATURE_COLUMNS) - size)

    def test_η_σειρα_των_στηλων_ειναι_η_σειρα_των_ονοματων(self):
        # Η κλιμάκωση στοιβάζει log, linear, ratio· το ablation ονομάζει τη
        # στήλη j ως FEATURE_COLUMNS[j]. Αν οι δύο σειρές αποκλίνουν, κάθε
        # όνομα οικογένειας δείχνει λάθος χαρακτηριστικό, αθόρυβα.
        from src.config import (FEATURE_COLUMNS, LOG_FEATURES,
                                LINEAR_FEATURES, RATIO_FEATURES)

        # Το scale_features στοιβάζει log, linear, ratio με αυτή τη σειρά.
        self.assertEqual(list(FEATURE_COLUMNS),
                         list(LOG_FEATURES) + list(LINEAR_FEATURES)
                         + list(RATIO_FEATURES))


class TestIntegrityStops(unittest.TestCase):
    """
    Αν η συνένωση χάσει bytes ή ροές, το pipeline πρέπει να ΣΤΑΜΑΤΑ.

    Μέχρι τώρα ο έλεγχος τύπωνε προειδοποίηση και επέστρεφε κανονικά: το
    στάδιο σημειωνόταν [OK], η εκτέλεση τελείωνε «επιτυχώς» και κάθε επόμενος
    αριθμός χτιζόταν πάνω σε αλλοιωμένα δεδομένα.
    """

    def run_with(self, outcomes):
        from src.utils import integrity
        with tempfile.TemporaryDirectory() as folder:
            for day in range(len(outcomes)):
                open(os.path.join(folder, f'bidir_2025112{day}.csv'), 'w').close()
            with mock.patch.dict(integrity.PATHS, {'bidirectional_data': folder}), \
                    mock.patch.object(integrity, 'check_day', side_effect=outcomes), \
                    contextlib.redirect_stdout(io.StringIO()):
                integrity.run()

    def test_ημερα_που_αποτυγχανει_σταματα_την_εκτελεση(self):
        with self.assertRaises(RuntimeError):
            self.run_with([True, False, True])

    def test_ολες_οι_ημερες_σωστες_δεν_σταματουν_τιποτα(self):
        self.run_with([True, True])


class TestServicePortArtefact(unittest.TestCase):
    """
    Το τεχνούργημα που έκανε 21 εξυπηρετητές NTP να μοιάζουν με σαρωτές.

    Η αλυσίδα: η συσκευή NAT ξαναγράφει τη θύρα προέλευσης σε διαδοχικές
    χαμηλές τιμές, το Service_Port = min(Src, Dst) τις διαλέγει επειδή είναι
    μικρότερες από το 123, και ο δείκτης διαδοχικότητας βγάζει τέλεια
    ακολουθία. Ο έλεγχος κλειδώνει τον κανόνα που το λύνει: θύρα υπηρεσίας
    υπάρχει ΜΟΝΟ όταν ακριβώς μία από τις δύο είναι κάτω από το όριο.
    """

    def clean_file(self, folder, rows):
        path = os.path.join(folder, 'clean_20251128.csv')
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_ροη_με_δυο_γνωστες_θυρες_δεν_δινει_θυρα_υπηρεσιας(self):
        from src.models import port_semantics

        with tempfile.TemporaryDirectory() as folder:
            # Ακριβώς το μοτίβο του τεχνουργήματος: NTP (123) απέναντι σε
            # θύρες που έγραψε το NAT, όλες κάτω από το 1024.
            self.clean_file(folder, [
                {'Src_Port': port, 'Dst_Port': 123,
                 'Src_Host': 'ntp_client', 'Dst_Host': 'server'}
                for port in range(40, 60)
            ])
            with mock.patch.dict(port_semantics.PATHS,
                                 {'processed_data': folder}):
                ports, partners = port_semantics.unambiguous_service_ports(
                    ['ntp_client'])

        # Καμία θύρα δεν περνά, άρα δεν υπάρχει διαδοχικότητα να μετρηθεί.
        self.assertEqual(ports['ntp_client'], set())
        self.assertTrue(np.isnan(port_semantics.consecutive(ports['ntp_client'])))
        # Η αντίπερα θύρα όμως καταγράφεται, για να φαίνεται τι ήταν η κίνηση.
        self.assertEqual(max(partners['ntp_client'], key=partners['ntp_client'].get), 123)

    def test_ροη_με_μια_γνωστη_θυρα_δινει_αυτην(self):
        from src.models import port_semantics

        with tempfile.TemporaryDirectory() as folder:
            self.clean_file(folder, [
                {'Src_Port': 123, 'Dst_Port': 45000,
                 'Src_Host': 'ntp_client', 'Dst_Host': 'peer'},
                {'Src_Port': 51000, 'Dst_Port': 443,
                 'Src_Host': 'ntp_client', 'Dst_Host': 'peer'},
            ])
            with mock.patch.dict(port_semantics.PATHS,
                                 {'processed_data': folder}):
                ports, _ = port_semantics.unambiguous_service_ports(['ntp_client'])

        self.assertEqual(ports['ntp_client'], {123, 443})

    def test_ο_ελεγχος_μιας_ημερας_δεν_σβηνει_την_επαληθευση_των_21(self):
        # ΕΛΕΓΧΟΣ ΠΑΛΙΝΔΡΟΜΗΣΗΣ. Η single_day_candidates() καλούσε τη
        # verify_candidates(), που έγραφε ΠΑΝΤΑ στο sequential_verified.csv:
        # το αρχείο-τεκμήριο των 21 αντικαθίστατο από τον πίνακα ενός host,
        # αθόρυβα, στο ίδιο τρέξιμο που το είχε μόλις γράψει.
        from src.models import port_semantics as ps

        with tempfile.TemporaryDirectory() as root:
            feat, bidir, clean = (os.path.join(root, d) for d in 'fbc')
            ports_dir = os.path.join(root, 'results', 'ports')
            peers_dir = os.path.join(root, 'results', 'peers')
            for folder in (feat, bidir, clean, ports_dir, peers_dir):
                os.makedirs(folder)

            # Ένας host στον πίνακα επανεμφάνισης, ένας μίας ημέρας με το
            # μοτίβο NTP/NAT.
            pd.DataFrame({'Host': ['tested', 'gap'],
                          'Well_Known_Ratio': [0.9, 1.0],
                          'Unique_Ports': [6, 6],
                          'P2P_Port_Ratio': [0, 0]}).to_csv(
                os.path.join(feat, 'features_global.csv'), index=False)
            pd.DataFrame({'Host': ['tested'], 'Πλήθος': [6], 'Ημέρες': [2],
                          'Ποσοστό_Διαδοχικών': [0.9]}).to_csv(
                os.path.join(peers_dir, 'recurrence_ports.csv'),
                index=False, encoding='utf-8-sig')
            pd.DataFrame({'IP_A': ['gap'] * 6, 'IP_B': ['srv'] * 6,
                          'Service_Port': range(100, 106)}).to_csv(
                os.path.join(bidir, 'bidir_20251127.csv'), index=False)
            pd.DataFrame({'Src_Port': range(100, 106), 'Dst_Port': [123] * 6,
                          'Src_Host': ['gap'] * 6, 'Dst_Host': ['srv'] * 6}).to_csv(
                os.path.join(clean, 'clean_20251127.csv'), index=False)

            evidence = os.path.join(ports_dir, 'sequential_verified.csv')
            with open(evidence, 'w', encoding='utf-8-sig') as handle:
                handle.write('Host\nτο_τεκμήριο_των_21\n')

            with mock.patch.dict(ps.PATHS, {'features': feat,
                                            'bidirectional_data': bidir,
                                            'processed_data': clean}), \
                    mock.patch.object(ps, 'BASE_DIR', root), \
                    contextlib.redirect_stdout(io.StringIO()):
                ps.single_day_candidates()

            with open(evidence, encoding='utf-8-sig') as handle:
                self.assertIn('το_τεκμήριο_των_21', handle.read(),
                              'το αρχείο των 21 αντικαταστάθηκε')
            single = pd.read_csv(os.path.join(ports_dir, 'sequential_single_day.csv'),
                                 encoding='utf-8-sig')
            self.assertEqual(list(single['Host']), ['gap'])

    def test_υποψηφιος_χωρις_ροες_δεν_ελεγχθηκε_δεν_απορριφθηκε(self):
        # Σε συνεπή εκτέλεση δεν συμβαίνει: οι υποψήφιοι βγαίνουν από τα bidir,
        # που φτιάχνονται από τις ίδιες στήλες Src_Host/Dst_Host των ίδιων clean.
        # Αν όμως τα clean αλλάξουν μετά τα bidir, ένας υποψήφιος μπορεί να μη
        # βρεθεί. Τότε το ποσοστό διαδοχικών είναι NaN, και ο παλιός κώδικας το
        # κατέγραφε ως «δεν επιβεβαιώνεται ως σαρωτής»: απόρριψη λόγω απουσίας.
        from src.models import port_semantics as ps

        with tempfile.TemporaryDirectory() as root:
            clean = os.path.join(root, 'clean')
            os.makedirs(clean)
            self.clean_file(clean, [
                {'Src_Port': port, 'Dst_Port': 123,
                 'Src_Host': 'ntp_client', 'Dst_Host': 'server'}
                for port in range(40, 60)
            ])
            flagged = pd.DataFrame({'Πλήθος': [20, 30],
                                    'Ποσοστό_Διαδοχικών': [1.0, 1.0]},
                                   index=pd.Index(['ntp_client', 'αόρατος'], name='Host'))
            with mock.patch.dict(ps.PATHS, {'processed_data': clean}), \
                    mock.patch.object(ps, 'BASE_DIR', root), \
                    contextlib.redirect_stdout(io.StringIO()):
                table = ps.verify_candidates(flagged, filename='δοκιμή.csv')

        self.assertEqual(table.loc['ntp_client', 'Κατάσταση'], 'Απορρίπτεται')
        self.assertEqual(table.loc['αόρατος', 'Κατάσταση'], 'Δεν ελέγχθηκε')
        self.assertFalse(table['Επιβεβαιώνεται'].any())

    def test_η_διαδοχικοτητα_μετραει_σωστα(self):
        from src.models.port_semantics import consecutive

        self.assertAlmostEqual(consecutive({10, 11, 12, 13}), 1.0)
        self.assertAlmostEqual(consecutive({10, 20, 30}), 0.0)
        # Με μία θύρα δεν ορίζεται — ΟΧΙ μηδέν, που θα περνούσε ως «καθαρός».
        self.assertTrue(np.isnan(consecutive({123})))
        self.assertTrue(np.isnan(consecutive(set())))

class TestResolvePorts(unittest.TestCase):
    """Η θύρα ακολουθεί τον host μετά την επίλυση NAT (aggregate_flows)."""

    def frame(self):
        # Απάντηση NTP: εξυπηρετητής:123 -> router:57, και η NAT στήλη δείχνει
        # τον εσωτερικό πελάτη στη 123. Μετά την επίλυση ο Dst_Host είναι ο
        # πελάτης, άρα και η θύρα του προορισμού πρέπει να γίνει 123.
        return pd.DataFrame({
            'Src_IP': ['srv', 'pc'], 'Src_Host': ['srv', 'pc'],
            'Dst_IP': ['router', 'srv'], 'Dst_Host': ['pc', 'srv'],
            'Src_Port': [123, 123], 'Dst_Port': [57, 123],
            'NAT_Src_Port': ['123', ' 57'], 'NAT_Dst_Port': [' 123', '123'],
        })

    def test_η_θυρα_του_επιλυμενου_host_ερχεται_απο_τη_nat_στηλη(self):
        from src.preprocessing.aggregate_flows import resolve_ports
        df, stats = resolve_ports(self.frame())
        self.assertEqual(df.loc[0, 'Dst_Port'], 123)
        self.assertEqual(np.minimum(df['Src_Port'], df['Dst_Port']).tolist(), [123, 123])
        self.assertEqual(stats['Dst'], {'moved': 1, 'changed': 1, 'no_nat': 0})
        self.assertEqual(stats['Src'], {'moved': 0, 'changed': 0, 'no_nat': 0})

    def test_μη_επιλυμενος_host_κραταει_τη_θυρα_του(self):
        from src.preprocessing.aggregate_flows import resolve_ports
        # Η εξερχόμενη ροή έχει NAT_Src_Port 57 (η θύρα του router), αλλά ο
        # host δεν άλλαξε, οπότε η θύρα του πελάτη (123) μένει.
        df, _ = resolve_ports(self.frame())
        self.assertEqual(df.loc[1, 'Src_Port'], 123)

    def test_χωρις_στηλες_nat_σταματα_αντι_να_προσπερασει(self):
        from src.preprocessing.aggregate_flows import resolve_ports
        frame = self.frame().drop(columns=['NAT_Dst_Port'])
        with self.assertRaises(KeyError):
            resolve_ports(frame)
        _, stats = resolve_ports(frame, required=False)
        self.assertIsNone(stats)

    def test_κενος_host_δεν_μετραει_ως_επιλυμενος(self):
        from src.preprocessing.aggregate_flows import resolve_ports
        frame = self.frame()
        frame.loc[0, ['Dst_IP', 'Dst_Host']] = np.nan
        df, stats = resolve_ports(frame)
        self.assertEqual(df.loc[0, 'Dst_Port'], 57)
        self.assertEqual(stats['Dst']['moved'], 0)

    def test_το_portcheck_βλεπει_τις_ιδιες_θυρες_με_το_aggregate(self):
        # Port forwarding: εξωτερικός:5555 -> router:8443, που το NAT στέλνει
        # στον εσωτερικό pc:443. Το aggregate δίνει Service_Port = 443. Χωρίς
        # επίλυση το portcheck θα έβλεπε 5555/8443 και καμία θύρα υπηρεσίας.
        from src.models import port_semantics
        from src.preprocessing.aggregate_flows import resolve_ports
        rows = pd.DataFrame({
            'Src_IP': ['ext'], 'Src_Host': ['ext'], 'Src_Port': [5555],
            'Dst_IP': ['router'], 'Dst_Host': ['pc'], 'Dst_Port': [8443],
            'NAT_Src_Port': ['5555'], 'NAT_Dst_Port': [' 443'],
        })
        resolved, _ = resolve_ports(rows.copy())
        self.assertEqual(int(min(resolved.loc[0, 'Src_Port'],
                                 resolved.loc[0, 'Dst_Port'])), 443)
        with tempfile.TemporaryDirectory() as folder:
            rows.to_csv(os.path.join(folder, 'clean_20251127.csv'), index=False)
            with mock.patch.dict(port_semantics.PATHS, {'processed_data': folder}):
                ports, _ = port_semantics.unambiguous_service_ports(['pc'])
        self.assertEqual(ports['pc'], {443})


class TestNoStaleEvidence(unittest.TestCase):
    """Χωρίς υποψήφιους ο πίνακας επαλήθευσης γράφεται άδειος, δεν μένει ο παλιός."""

    def test_αδειος_ελεγχος_σβηνει_τον_παλιο_πινακα(self):
        from src.models import port_semantics as ps
        with tempfile.TemporaryDirectory() as folder:
            stale = os.path.join(folder, 'sequential_verified.csv')
            pd.DataFrame({'Host': ['παλιός'], 'Κατάσταση': ['Απορρίπτεται']}).to_csv(
                stale, index=False)
            with mock.patch.object(ps, 'output_dir', return_value=folder):
                ps.verify_candidates(pd.DataFrame())
            self.assertTrue(pd.read_csv(stale).empty)


class TestExternalPopulationEverywhere(unittest.TestCase):
    """Το peers/portcheck παίρνει τον ίδιο εξωτερικό πληθυσμό με το macro."""

    def test_εκπομπη_και_ιδιωτικοι_δεν_ειναι_εξωτερικοι(self):
        from src.models.data_split import split_internal_external
        inside = anonymize_ip(INTERNAL_SUBNET + '10')
        broadcast = anonymize_ip(INTERNAL_SUBNET + '255')
        frame = pd.DataFrame({'Host': [inside, broadcast, 'other_private', 'public']})
        _, external = split_internal_external(frame, other_private=frozenset({'other_private'}))
        self.assertEqual(external['Host'].tolist(), ['public'])



class TestDirectionsSymmetricPorts(unittest.TestCase):
    """
    Ο πίνακας κατεύθυνσης του subnets δεν μετρά απαντήσεις ως κλήσεις.

    Μετά την επίλυση των θυρών μέσω NAT, η απάντηση NTP srv:123 -> pc:123
    έχει Dst_Port 123 και προορισμό τον εσωτερικό πελάτη. Αν μετρούσε, ο
    πελάτης θα εμφανιζόταν ως ακροατής στη 123 που τον καλούν από έξω.
    """

    def test_απαντηση_σε_ιδια_θυρα_δεν_κανει_τον_πελατη_ακροατη(self):
        pc, srv, ext = (anonymize_ip('192.168.8.10'), anonymize_ip('8.8.8.8'),
                        anonymize_ip('9.9.9.9'))
        rows = pd.DataFrame({
            # ερώτηση NTP, απάντηση NTP, ερώτηση DNS, port forwarding 8443 -> 443
            'Src_IP':   [pc, srv, pc, ext],
            'Src_Host': [pc, srv, pc, ext],
            'Src_Port': [123, 123, 50000, 5555],
            'Dst_IP':   [srv, 'router', srv, 'router'],
            'Dst_Host': [srv, pc, srv, pc],
            'Dst_Port': [123, 57, 53, 8443],
            'NAT_Src_Port': ['123', '123', '50000', '5555'],
            'NAT_Dst_Port': ['123', ' 123', '53', ' 443'],
        })
        ports_table = pd.DataFrame({'Service_Port': [53, 123, 443]})
        with tempfile.TemporaryDirectory() as root:
            clean = os.path.join(root, 'clean')
            os.makedirs(clean)
            rows.to_csv(os.path.join(clean, 'clean_20251127.csv'), index=False)
            with mock.patch.dict(subnet_analysis.PATHS, {'processed_data': clean}), \
                    mock.patch.object(subnet_analysis, 'BASE_DIR', root), \
                    contextlib.redirect_stdout(io.StringIO()):
                table = subnet_analysis.write_directions(ports_table)

        listens = {(int(r['Θύρα']), r['Ακούει'], r['Συνδέεται'])
                   for _, r in table.iterrows()}
        # Η 123 είναι ίδια και στις δύο πλευρές: καμία από τις δύο ροές δεν μετρά.
        self.assertFalse(any(port == 123 for port, _, _ in listens))
        self.assertIn((53, subnet_analysis.EXTERNAL, '192.168.8.x'), listens)
        self.assertIn((443, '192.168.8.x', subnet_analysis.EXTERNAL), listens)


class TestStaleOutputsRemoved(unittest.TestCase):
    """Όταν λείπουν τα δεδομένα εισόδου, οι παλιοί πίνακες δεν μένουν στον δίσκο."""

    def test_χωρις_αποτελεσματα_peers_σβηνονται_οι_παλιοι_πινακες(self):
        from src.models import port_semantics as ps
        with tempfile.TemporaryDirectory() as root, \
                tempfile.TemporaryDirectory() as out:
            stale = [os.path.join(out, name) for name in
                     ('sequential_well_known.csv', 'sequential_verified.csv',
                      'single_day_gap.csv', 'sequential_single_day.csv')]
            for path in stale:
                pd.DataFrame({'Host': ['παλιός']}).to_csv(path, index=False)
            with mock.patch.dict(ps.PATHS, {'features': root}), \
                    mock.patch.object(ps, 'BASE_DIR', root), \
                    mock.patch.object(ps, 'output_dir', return_value=out), \
                    contextlib.redirect_stdout(io.StringIO()):
                ps.hidden_scanners()
                ps.single_day_candidates()
            for path in stale:
                self.assertFalse(os.path.exists(path), os.path.basename(path))


class TestSameHostFlows(unittest.TestCase):
    """
    Ροές με τον ίδιο host και στις δύο άκρες (hairpin NAT) μετρούν μία φορά.

    Στα δεδομένα είναι περίπου 8.500 ροές, σχεδόν όλες της πύλης του 192.168.7.x.
    Πριν μετρούσαν δύο φορές για τον ίδιο host, και ο host ήταν peer του εαυτού του.
    """

    def test_ο_χρονικος_πινακας_μετρα_μια_φορα(self):
        frame = pd.DataFrame({'Src_Host': ['a', 'g'], 'Dst_Host': ['b', 'g'],
                              'Date': [20251127] * 2, 'Hour': [10] * 2,
                              'DayOfWeek': [3] * 2, 'Bytes': [500, 70]})
        table = host_time_table(frame).set_index('Host')
        self.assertEqual(int(table.loc['g', 'Flows']), 1)
        self.assertEqual(int(table.loc['g', 'Bytes']), 70)
        self.assertEqual(int(table['Flows'].sum()), 2 * 2 - 1)

    def test_ο_ελεγχος_ακεραιοτητας_συμφωνει_με_τον_χρονικο_πινακα(self):
        # Ό,τι περιμένει το integrity πρέπει να είναι ακριβώς ό,τι γράφει το
        # host_time_table, αλλιώς το verify σταματά την εκτέλεση.
        from src.utils import integrity
        clean = pd.DataFrame({'Bytes': [500, 70, 30], 'Src_Host': ['a', 'g', 'g'],
                              'Dst_Host': ['b', 'g', 'c']})
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'clean_20251127.csv')
            clean.to_csv(path, index=False)
            totals = integrity.clean_totals(path)
        self.assertEqual(totals['same_rows'], 1)
        self.assertEqual(totals['same_bytes'], 70.0)
        timed = clean.assign(Date=20251127, Hour=10, DayOfWeek=3)
        table = host_time_table(timed)
        self.assertEqual(float(table['Flows'].sum()),
                         2 * totals['rows_with_hosts'] - totals['same_rows'])
        self.assertEqual(float(table['Bytes'].sum()),
                         2 * totals['bytes'] - totals['same_bytes'])

    def profiles(self, bidir):
        from src.features.extract_features import build_profiles
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'bidir_20251127.csv')
            bidir.to_csv(path, index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                return build_profiles([path], temporal_files=[])

    def test_τα_προφιλ_μετρανε_τις_ροες_μια_φορα_και_κρατουν_τα_bytes(self):
        # Ο g μιλά με τον a (443) και με τον εαυτό του (53, 4 ροές, 70 bytes,
        # γραμμένα στο BtoA όπως τα γράφει το aggregate).
        profiles = self.profiles(pd.DataFrame({
            'IP_A': ['a', 'g'], 'IP_B': ['g', 'g'],
            'Service_Port': [443, 53], 'Protocol': ['TCP', 'UDP'],
            'Bytes_AtoB': [100, 0], 'Bytes_BtoA': [300, 70],
            'Flows_AtoB': [2, 0], 'Flows_BtoA': [1, 4],
        }))
        g = profiles.loc['g']
        self.assertEqual(int(g['Total_Flows']), 3 + 4)       # όχι 3 + 8
        self.assertEqual(int(g['Unique_Peers']), 1)           # μόνο ο a
        # Τα bytes του hairpin μετρούν και ως εξερχόμενα και ως εισερχόμενα,
        # όπως πριν τη διόρθωση: ο host στέλνει και λαμβάνει κάθε byte.
        self.assertEqual(float(g['Bytes_Out']), 300 + 70)
        self.assertEqual(float(g['Bytes_In']), 100 + 70)
        self.assertEqual(int(profiles.loc['a', 'Unique_Peers']), 1)
        self.assertEqual(profiles['Unique_Peers'].dtype.name, 'int64')

    def test_host_μονο_με_τον_εαυτο_του(self):
        profiles = self.profiles(pd.DataFrame({
            'IP_A': ['a', 'h'], 'IP_B': ['b', 'h'],
            'Service_Port': [443, 53], 'Protocol': ['TCP', 'UDP'],
            'Bytes_AtoB': [100, 0], 'Bytes_BtoA': [50, 70],
            'Flows_AtoB': [1, 0], 'Flows_BtoA': [1, 2],
        }))
        h = profiles.loc['h']
        self.assertEqual(int(h['Unique_Peers']), 0)
        self.assertEqual(int(h['Total_Flows']), 2)
        self.assertAlmostEqual(float(h['Out_Ratio']), 0.5)
        self.assertEqual(profiles['Unique_Peers'].dtype.name, 'int64')


class TestFeedbackFixes(unittest.TestCase):
    """Μικρές διορθώσεις της 19/11/2026."""

    def test_votes_χωρις_τιμες_ευσταθειας_δεν_σκαει(self):
        from src.models.select_k import votes
        df = pd.DataFrame({'k': [3, 4, 5], 'Inertia': [30.0, 20.0, 15.0],
                           'Silhouette': [0.3, 0.4, 0.35],
                           'Calinski_Harabasz': [10.0, 12.0, 11.0],
                           'Davies_Bouldin': [1.0, 0.8, 0.9],
                           'Gap': [0.1, 0.2, 0.3], 'Gap_s': [0.05] * 3,
                           'Stability_ARI': [np.nan] * 3})
        picked = votes(df)
        self.assertIsNone(picked['Ευστάθεια ARI'])
        self.assertEqual(picked['Silhouette'], 4)
        self.assertEqual(picked['Davies-Bouldin'], 4)

    def test_το_manifest_καλυπτει_subnets_και_ablation(self):
        from src.utils.manifest import OUTPUT_FOLDERS
        self.assertIn('subnets', OUTPUT_FOLDERS)
        self.assertIn('ablation', OUTPUT_FOLDERS)

    def test_ablation_δεν_μετρα_το_μειον_ενα_ως_ομαδα(self):
        # Μεικτό αρχείο: τρεις ομάδες συν hosts με Cluster = -1. Πριν, το -1
        # μετρούσε ως τέταρτη ομάδα, το k δεν ταίριαζε και ο πληθυσμός παραλειπόταν.
        from src.models import ablation
        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.random((30, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
        df['Cluster'] = [0] * 9 + [1] * 9 + [2] * 9 + [-1] * 3
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'subnet_192_168_1_global.csv')
            df.to_csv(path, index=False)
            with mock.patch.object(ablation, 'default_k', return_value=3), \
                    contextlib.redirect_stdout(io.StringIO()):
                rows = ablation.analyse('subnet_192_168_1', 'test', path)
        self.assertTrue(rows)

    def test_το_sensitivity_παιρνει_το_k_του_macro(self):
        from src.models import sensitivity
        self.assertFalse(hasattr(sensitivity, 'MACRO_CLUSTERS'))
        self.assertTrue(hasattr(sensitivity, 'default_k'))


class TestOpenItems1911(unittest.TestCase):
    """
    Τα θέματα που είχαν μείνει ανοιχτά στις 19/11/2026.

    Το καθένα έχει εδώ τον έλεγχο που αποτυγχάνει στον παλιό κώδικα.
    """

    # --- Σταθερότητα επικρατέστερης ομάδας ---------------------------------

    def test_η_ευσταθεια_δεν_βγαζει_πια_σταθεροτητα_επικρατεστερης(self):
        # Μετριόταν πάνω σε ετικέτες πριν την αντιστοίχιση, άρα δεν είχε νόημα.
        from src.models import daily_stability as ds
        hosts = [f'h{i}' for i in range(6)]
        labels = {
            '20251127': pd.Series([0, 0, 1, 1, 2, 2], index=hosts),
            '20251128': pd.Series([1, 1, 0, 0, 2, 2], index=hosts),
        }
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as folder:
            stale = os.path.join(folder, 'consistency_micro.csv')
            open(stale, 'w').close()
            with mock.patch.object(ds, 'output_dir', return_value=folder), \
                    mock.patch.object(ds, 'align_all', return_value=None), \
                    contextlib.redirect_stdout(out):
                ds.analyse('micro', labels, 3)
            self.assertFalse(os.path.exists(stale))
            self.assertFalse(any(name.startswith('consistency_')
                                 for name in os.listdir(folder)))
        self.assertNotIn('επικρατέστερ', out.getvalue())
        self.assertFalse(hasattr(ds, 'consistency'))

    # --- --k-micro στο stability --------------------------------------------

    def test_το_k_micro_ισχυει_και_για_τα_υποδικτυα_στο_stability(self):
        # Το micro εφαρμόζει το --k-micro σε ΟΛΟΥΣ τους πληθυσμούς. Το
        # stability το έδινε μόνο στο 'micro' και τα υποδίκτυα έπαιρναν το k
        # του config, ενώ αντιστοιχίζονταν σε αρχεία φτιαγμένα με άλλο k.
        from src.models import daily_stability as ds
        fake = [('micro', 't', set()), ('subnet_192_168_7', 't', set())]
        with mock.patch.object(ds, 'reference_labels', return_value=None), \
                contextlib.redirect_stdout(io.StringIO()):
            ks = ds.levels_k(None, 4, fake)
        self.assertEqual(ks['micro'], 4)
        self.assertEqual(ks['subnet_192_168_7'], 4)

    def test_χωρις_ορισμα_το_stability_παιρνει_το_k_του_αρχειου_αναφορας(self):
        # Αν το micro έτρεξε σε άλλη διεργασία με άλλο k, το K_USED είναι
        # άδειο. Το σωστό k είναι αυτό του αρχείου στο οποίο αντιστοιχίζουμε.
        from src.models import daily_stability as ds
        from src.models import select_k
        reference = pd.Series([0, 1, 2, 3, 4, 0, 1], index=list('abcdefg'))
        with mock.patch.dict(select_k.K_USED, {}, clear=True), \
                mock.patch.object(ds, 'reference_labels', return_value=reference), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ds.resolve_k('micro', None, 7), 5)

    def test_το_k_απο_τη_γραμμη_εντολων_καταγραφεται(self):
        # Με --k-micro δεν καλείται το choose_k, οπότε το manifest και τα
        # γραφήματα έπεφταν στο k του config.
        from src.models import micro_profiling, select_k
        with mock.patch.dict(select_k.K_USED, {}, clear=True), \
                contextlib.redirect_stdout(io.StringIO()):
            micro_profiling.analyse_population('subnet_192_168_7', 't', set(), [], 4)
            self.assertEqual(select_k.K_USED['subnet_192_168_7'],
                             {'k': 4, 'source': 'cli'})

    # --- Έλεγχος ακεραιότητας ----------------------------------------------

    def test_ο_ελεγχος_σταματα_οταν_δεν_υπαρχουν_αρχεια(self):
        from src.utils import integrity
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.dict(integrity.PATHS, {'bidirectional_data': folder}), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError):
                integrity.run()

    def test_ο_ελεγχος_σταματα_οταν_δεν_ελεγχθηκε_καμια_ημερα(self):
        from src.utils import integrity
        with tempfile.TemporaryDirectory() as folder:
            open(os.path.join(folder, 'bidir_20251127.csv'), 'w').close()
            with mock.patch.dict(integrity.PATHS, {'bidirectional_data': folder}), \
                    mock.patch.object(integrity, 'check_day', return_value=None), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(RuntimeError):
                    integrity.run()

    # --- Μερικές ημέρες -----------------------------------------------------

    def test_ημερα_που_τελειωνει_νωρις_ειναι_μερικη(self):
        rows = []
        for date, hours in ((20251127, range(24)), (20251203, range(0, 20))):
            rows += [{'Date': date, 'Hour': h} for h in hours]
        found, _ = partial_days(pd.DataFrame(rows))
        self.assertEqual(found, [20251203])

    # --- Κατάλογος ιδιωτικών και ειδικών διευθύνσεων ------------------------

    def test_το_γρηγορο_hash_ταυτιζεται_με_το_anonymize_ip(self):
        from src.utils.anonymize import prefix_hashes
        hashes = prefix_hashes('203.0.113.')
        self.assertEqual(len(hashes), 256)
        for octet in (0, 7, 99, 255):
            self.assertEqual(hashes[octet], anonymize_ip(f'203.0.113.{octet}'))
        # Κλειδί μεγαλύτερο από 64 bytes: το HMAC το κάνει πρώτα hash.
        import hmac, hashlib
        key = b'k' * 100
        self.assertEqual(
            prefix_hashes('10.0.0.', key=key)[5],
            hmac.new(key, b'10.0.0.5', hashlib.sha256).hexdigest()[:16])

    def test_τα_ειδικα_ευρη_καλυπτονται_ολοκληρα(self):
        from src.utils.private_scan import SPECIAL_RANGES, special_prefixes
        counts = {}
        for prefix, category in special_prefixes():
            counts[category] = counts.get(category, 0) + 1
        self.assertEqual(counts['multicast'], 16 * 65536)
        self.assertEqual(counts['link-local'], 256)
        self.assertEqual(counts['CGNAT'], 64 * 256)
        self.assertEqual(counts['loopback'], 65536)
        self.assertEqual(counts['this-network'], 65536)
        self.assertIn('255.255.255.255', [ip for ip, _ in SPECIAL_RANGES['single']])

    def test_η_σαρωση_βρισκει_multicast_εκπομπη_και_link_local(self):
        from src.utils.private_scan import scan_special
        found_ips = ['224.0.0.251', '255.255.255.255', '169.254.1.1', '0.0.0.0']
        hosts = {anonymize_ip(ip) for ip in found_ips + ['8.8.8.8']}
        prefixes = [('224.0.0.', 'multicast'), ('169.254.1.', 'link-local'),
                    ('0.0.0.', 'this-network'), ('9.9.9.', 'multicast')]
        found = scan_special(hosts, prefixes=prefixes, outer_prefix='')
        self.assertEqual(len(found), 4)
        self.assertNotIn(anonymize_ip('8.8.8.8'), found)
        self.assertEqual(found[anonymize_ip('255.255.255.255')][1], 'broadcast')
        for label, _ in found.values():
            self.assertTrue(label.endswith('x'))

    def test_η_εκπομπη_του_εξωτερικου_δικτυου_χωρις_το_προθεμα(self):
        # Το πρόθεμα του εξωτερικού /24 είναι δημόσια διεύθυνση. Μένει στο
        # .env και δεν γράφεται ποτέ στον κατάλογο ή στα αποτελέσματα.
        from src.utils.private_scan import scan_special
        hosts = {anonymize_ip('203.0.113.255'), anonymize_ip('203.0.113.0'),
                 anonymize_ip('203.0.113.9')}
        found = scan_special(hosts, prefixes=[], outer_prefix='203.0.113.')
        self.assertEqual(len(found), 2)
        self.assertNotIn(anonymize_ip('203.0.113.9'), found)
        for label, category in found.values():
            self.assertNotIn('203', label)
            self.assertNotIn('113', label)

    def test_οι_ειδικες_διευθυνσεις_βγαινουν_απο_τους_εξωτερικους(self):
        from src.utils import private_scan
        broadcast = anonymize_ip('255.255.255.255')
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'private_hosts.csv')
            with open(path, 'w', encoding='utf-8-sig', newline='') as handle:
                handle.write('Host,Υποδίκτυο,Στα_υποδίκτυα_της_αναλυσης,Κατηγορία\n')
                handle.write(f'{broadcast},255.255.255.x,False,broadcast\n')
                handle.write(f'{anonymize_ip(INTERNAL_SUBNET + "5")},'
                             f'{INTERNAL_SUBNET}x,True,RFC1918\n')
            with mock.patch.object(private_scan, 'OUTPUT', path):
                others = private_scan.others()
        self.assertEqual(others, frozenset([broadcast]))

    def test_το_privatescan_καθαριζει_τη_μνημη_του_διαχωρισμου(self):
        # Στο διαδραστικό μενού ο κατάλογος έμενε στη μνήμη (lru_cache), οπότε
        # macro, privatescan και ξανά macro έβλεπαν τον παλιό κατάλογο.
        from src.utils import private_scan
        from src.models import data_split
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(private_scan, 'OUTPUT', os.path.join(folder, 'p.csv')), \
                mock.patch.object(private_scan, 'profile_hosts', return_value={'a'}), \
                mock.patch.object(private_scan, 'scan', return_value={}), \
                mock.patch.object(private_scan, 'scan_special', return_value={}), \
                contextlib.redirect_stdout(io.StringIO()):
            data_split.private_hosts()
            self.assertEqual(data_split.private_hosts.cache_info().currsize, 1)
            private_scan.run()
            self.assertEqual(data_split.private_hosts.cache_info().currsize, 0)

    def test_το_privatescan_δεν_ξανασαρωνει_ιδια_δεδομενα(self):
        from src.utils import private_scan
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(private_scan, 'OUTPUT', os.path.join(folder, 'p.csv')), \
                mock.patch.object(private_scan, 'profile_hosts', return_value={'a', 'b'}), \
                mock.patch.object(private_scan, 'scan', return_value={}) as rfc, \
                mock.patch.object(private_scan, 'scan_special', return_value={}) as special, \
                contextlib.redirect_stdout(io.StringIO()):
            private_scan.run()
            private_scan.run()
            self.assertEqual(rfc.call_count, 1)
            self.assertEqual(special.call_count, 1)
            with mock.patch.object(private_scan, 'profile_hosts', return_value={'a', 'c'}):
                private_scan.run()
            self.assertEqual(rfc.call_count, 2)


    # --- Ablation: σχεδόν ισοδύναμες λύσεις του K-Means ---------------------

    def test_οι_μειωμενες_παραλλαγες_ψαχνουν_με_50_αρχικοποιησεις(self):
        # Στο macro το «χωρίς θυρών» έχει δύο λύσεις με διαφορά αδράνειας
        # 0,04% (ARI 0,880 και 0,560). Με 10 αρχικοποιήσεις το αποτέλεσμα
        # άλλαζε ανάλογα με το ποια έβρισκε πρώτη. Η «όλα» μένει με τις
        # ρυθμίσεις του σταδίου, ώστε να αναπαράγει τις ετικέτες (ARI 1,000).
        from src.models import ablation
        seen = {}
        real = ablation.KMeans

        def spy(*args, **kwargs):
            seen.setdefault(kwargs['n_init'], 0)
            seen[kwargs['n_init']] += 1
            return real(*args, **kwargs)

        rng = np.random.default_rng(1)
        df = pd.DataFrame(rng.random((40, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
        df['Cluster'] = [0, 1, 2, 3] * 10
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'micro_global.csv')
            df.to_csv(path, index=False)
            with mock.patch.object(ablation, 'KMeans', side_effect=spy), \
                    mock.patch.object(ablation, 'default_k', return_value=4), \
                    contextlib.redirect_stdout(io.StringIO()):
                ablation.analyse('micro', 'test', path)
        self.assertEqual(seen.get(10), 1)
        self.assertEqual(seen.get(50), 6)


class TestPrefixNotRecoverable(unittest.TestCase):
    """
    Το πρόθεμα του εξωτερικού /24 δεν πρέπει να βγαίνει από το .state.

    Ο έλεγχος κώδικα της 20/11 έδειξε ότι ένα αποτύπωμα SHA256 χωρίς κλειδί
    αποκάλυπτε το πρόθεμα με δοκιμή των 2^24 τιμών, αφού οι hosts φαίνονται
    στα αποτελέσματα.
    """

    def test_το_αποτυπωμα_εξαρταται_απο_το_κλειδι(self):
        from src.utils import private_scan as scan
        hosts = {'aaaa', 'bbbb'}
        with mock.patch.object(scan, 'OUTER_PREFIX', '203.0.113.'):
            first = scan.fingerprint(hosts, key=b'key-one')
            second = scan.fingerprint(hosts, key=b'key-two')
            again = scan.fingerprint(hosts, key=b'key-one')
        self.assertNotEqual(first, second)
        self.assertEqual(first, again)

    def test_χωρις_το_κλειδι_το_προθεμα_δεν_ξαναβγαινει(self):
        # Αυτό που θα έκανε κάποιος με τα αποτελέσματα αλλά χωρίς το .env:
        # το ίδιο περιεχόμενο με σκέτο SHA256 δεν δίνει το αποτύπωμα.
        import hashlib
        from src.utils import private_scan as scan
        hosts = {'aaaa', 'bbbb'}
        with mock.patch.object(scan, 'OUTER_PREFIX', '203.0.113.'):
            stored = scan.fingerprint(hosts, key=b'secret')
        plain = hashlib.sha256()
        for host in sorted(hosts):
            plain.update(host.encode('ascii') + b'\n')
        plain.update(repr(sorted((k, repr(v)) for k, v in
                                 scan.SPECIAL_RANGES.items())).encode())
        plain.update(b'outer:203.0.113.')
        for version in (b'v2', b'v3'):
            candidate = plain.copy()
            candidate.update(version)
            self.assertNotEqual(stored, candidate.hexdigest())
        self.assertNotIn('203.0.113', stored)

    def test_χωρις_αρχεια_clean_σβηνονται_οι_πινακες_του_portcheck(self):
        from src.models import port_semantics as ps
        with tempfile.TemporaryDirectory() as out:
            stale = [os.path.join(out, name) for name in
                     ('port_pair_cases.csv', 'sequential_well_known.csv')]
            for path in stale:
                pd.DataFrame({'Host': ['παλιός']}).to_csv(path, index=False)
            with mock.patch.object(ps, 'port_pair_cases',
                                   return_value=(None, 0, 0)), \
                    mock.patch.object(ps, 'output_dir', return_value=out), \
                    contextlib.redirect_stdout(io.StringIO()):
                ps.run()
            for path in stale:
                self.assertFalse(os.path.exists(path), os.path.basename(path))


if __name__ == '__main__':
    unittest.main()
