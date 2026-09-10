"""Strict, toneless Wugniu -> IPA phone conversion for the Shanghai pilot.

Onsets (including affricates/aspiration) are one token. Rime components are
Unicode bases with attached combining diacritics, normalized to NFC. This is
an explicit engineering convention, not a claim of acoustic segmentation.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEME = PROJECT_ROOT / (
    "generated/recording_11_20260909/tone_pilot/alignment_diagnostics/"
    "wugniu_scheme.json"
)
TOKEN_RE = re.compile(r"([a-zɡ]+)([0-9]*)\Z")
BLANK = "<blank>"


class SpellingError(ValueError):
    """An input spelling is malformed, unknown, or has multiple parses."""


def ipa_units(ipa: str) -> list[str]:
    """Keep a nasalization/syllabicity mark with its preceding IPA base."""
    units: list[str] = []
    for char in ipa:
        if unicodedata.combining(char):
            if not units:
                raise SpellingError(f"Unattached IPA combining mark: {ipa!r}")
            units[-1] += char
        else:
            units.append(char)
    return [unicodedata.normalize("NFC", unit) for unit in units]


class Wugniu:
    def __init__(self, scheme_path: str | Path = DEFAULT_SCHEME):
        self.scheme_path = Path(scheme_path).resolve()
        self.scheme: dict[str, Any] = json.loads(
            self.scheme_path.read_text(encoding="utf-8")
        )

    def parse_syllable(self, token: str) -> dict[str, Any]:
        match = TOKEN_RE.fullmatch(token)
        if match is None:
            raise SpellingError(f"Malformed whole syllable: {token!r}")
        spelling, tone = match.groups()
        spelling = self.scheme.get("source_spelling_aliases", {}).get(
            spelling, spelling
        )
        special = self.scheme["whole_syllable_specials"]
        conditioned = self.scheme["gh_conditioned_syllables"]
        if spelling in special:
            options = [("", special[spelling]["final"], "syllabic_nasal")]
        elif spelling in conditioned:
            options = [("gh", conditioned[spelling], "conditioned_gh")]
        else:
            options = [
                (initial, spelling[len(initial):], "regular")
                for initial in sorted(self.scheme["initials"], key=len, reverse=True)
                if spelling.startswith(initial)
                and spelling[len(initial):] in self.scheme["finals"]
            ]
        if not options:
            raise SpellingError(f"Unknown spelling: {token!r}")
        if len(options) != 1:
            raise SpellingError(f"Ambiguous spelling {token!r}: {options!r}")
        initial, final, method = options[0]
        onset_ipa = self.scheme["initials"][initial]
        final_ipa = self.scheme["finals"][final]
        phones = ([unicodedata.normalize("NFC", onset_ipa)] if onset_ipa else [])
        phones += ipa_units(final_ipa)
        return {
            "raw_token": token,
            "spelling": spelling,
            "tone_suffix_raw": tone,
            "initial": initial,
            "final": final,
            "ipa": unicodedata.normalize("NFC", onset_ipa + final_ipa),
            "phones": phones,
            "parse_method": method,
        }

    def convert(self, text: str) -> dict[str, Any]:
        if not text.strip():
            raise SpellingError("Empty utterance")
        phones: list[str] = []
        syllables = []
        for number, token in enumerate(text.split()):
            syllable = self.parse_syllable(token)
            start = len(phones)
            phones.extend(syllable["phones"])
            syllables.append(dict(syllable, index=number, phone_span=[start, len(phones)]))
        return {
            "phones": phones,
            "syllables": syllables,
            "toneless_text": " ".join(s["spelling"] for s in syllables),
            "tone_suffixes_raw": [s["tone_suffix_raw"] for s in syllables],
        }

    def validate_examples(self) -> None:
        for example in self.scheme["examples"]:
            got = self.parse_syllable(example["spelling"])
            expected = unicodedata.normalize("NFC", example["ipa_candidate"])
            if got["ipa"] != expected:
                raise AssertionError((example, got))


def training_syllables(converted: dict[str, Any]) -> list[dict[str, Any]]:
    """No tone suffix, numeric tone label, raw transcript, or time targets."""
    keys = ("index", "spelling", "initial", "final", "ipa", "phone_span")
    return [{key: s[key] for key in keys} for s in converted["syllables"]]
