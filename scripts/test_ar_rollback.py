"""Pin v4.8 AR computation while preserving v5.2 codec sources.

Run after prepare-runtime.sh. Only monitoring/cancellation statements are excluded
from the full AR source fingerprint; math, sampling, graphs and scheduling remain
covered. The reference comes from 370defa's patch on runtime 3a0dd442.
"""
from pathlib import Path
import hashlib
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / 'Audio8-Android/runtime'


def ar_computation(source):
    source = re.sub(r'^ *std::fprintf\(.*?;\n', '', source, flags=re.M | re.S)
    source = re.sub(r'^ *mobile_check_cancelled\(\);\n', '', source, flags=re.M)
    source = re.sub(r'^ *if\(step%16==0 && mobile_options.progress\) \{\n.*?^ *\}\n', '', source, flags=re.M | re.S)
    source = re.sub(r'^ *if\(mobile_options.progress\)mobile_options.progress\(.*?;\n', '', source, flags=re.M)
    source = re.sub(r'^ *const auto prefill_(?:start|done)=Clock::now\(\);\n', '', source, flags=re.M)
    return source


class RollbackTests(unittest.TestCase):
    def test_ar_computation_matches_first_preview_release(self):
        source = (RUNTIME / 'src/community_models/audio8_tts/ar.cpp').read_text()
        self.assertEqual(hashlib.sha256(ar_computation(source).encode()).hexdigest(),
                         '56f38c784bfa1ad22c4f79a0675b52df442d7bcb04245222deed4c6d2c4b4ca8')

    def test_v52_decoder_sources_unchanged(self):
        expected = {
            'Audio8-Android/runtime/src/community_models/audio8_tts/codec.cpp': '1c2d9e88e4e096811cc47ae034cc6e4b0512ab06eaf9a04bdb73193920c515b2',
            'Audio8-Android/runtime/include/engine/community_models/audio8_tts/mobile_codec.h': '66136814e98b7cafc859f17271876b3db91eea739e64786b6d5521b32eee5293',
            'Audio8-Android/app/src/main/kotlin/com/example/audio8tts/LiteCodecRuntime.kt': '1b08e331d201b1a21108ed344acaa0f70b989a397bc4ae3b605ba395e37357f2',
            'Audio8-Android/app/src/main/java/com/example/audio8tts/CodecBackend.java': 'c8ea4f0533f3660ff5a0ebb349574fd62c54e127fc10bec934f33f6265aa232d',
            'Audio8-Android/app/src/main/java/com/example/audio8tts/LiteCodecStore.java': 'f79282f3e5757a5fec6d5999995c75a6223d22919aed567b8eb636212106695c',
            'Audio8-Android/app/src/main/java/com/example/audio8tts/CodecContract.java': '2c849a7f7352143e1b933a292243d00fce00baef88c843c44e33f8fd35723f0b',
        }
        for path, digest in expected.items():
            with self.subTest(path=path):
                self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest)


if __name__ == '__main__':
    unittest.main(verbosity=2)
