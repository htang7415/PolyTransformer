"""Hugging Face compatible Group SELFIES tokenizer implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from transformers import PreTrainedTokenizer


class HFGroupSelfiesTokenizer(PreTrainedTokenizer):
    """HF tokenizer wrapping GroupSELFIESTokenizer.

    The core HF tokenization path (_tokenize) expects a pre-computed Group
    SELFIES string (already converted from SMILES). For the full SMILES ->
    Group SELFIES encoding path (which requires RDKit + grammar), call
    encode_smiles() directly.

    Serialization uses two files:
      - vocab.json : plain token->id mapping (human-readable)
      - tokenizer.pkl : full tokenizer state including the grammar object
    """

    vocab_files_names = {
        "vocab_file": "vocab.json",
        "tokenizer_state_file": "tokenizer.pkl",
    }
    model_input_names = ["input_ids", "attention_mask"]

    SPECIAL_TOKENS = ["[PAD]", "[MASK]", "[BOS]", "[EOS]", "[UNK]"]

    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        max_length: int = 256,
        tokenizer_state_file: Optional[str] = None,
        vocab_file: Optional[str] = None,
        **kwargs,
    ):
        from .tokenizer import GroupSELFIESTokenizer

        # Load inner tokenizer from pickle state if available (preserves grammar)
        if tokenizer_state_file and Path(tokenizer_state_file).exists():
            self._inner = GroupSELFIESTokenizer.load(tokenizer_state_file)
        elif vocab is not None or (vocab_file and Path(vocab_file).exists()):
            loaded_vocab = vocab
            if loaded_vocab is None and vocab_file:
                with open(vocab_file, "r", encoding="utf-8") as f:
                    loaded_vocab = json.load(f)
            self._inner = GroupSELFIESTokenizer(
                grammar=None, vocab=loaded_vocab or {}, max_length=max_length
            )
        else:
            self._inner = GroupSELFIESTokenizer(grammar=None, vocab={}, max_length=max_length)

        self.max_length = self._inner.max_length
        self.vocab: Dict[str, int] = dict(self._inner.vocab)
        self.id_to_token: Dict[int, str] = {v: k for k, v in self.vocab.items()}

        pad_token = kwargs.pop("pad_token", "[PAD]")
        mask_token = kwargs.pop("mask_token", "[MASK]")
        bos_token = kwargs.pop("bos_token", "[BOS]")
        eos_token = kwargs.pop("eos_token", "[EOS]")
        unk_token = kwargs.pop("unk_token", "[UNK]")
        model_max_length = kwargs.pop("model_max_length", self.max_length)

        super().__init__(
            pad_token=pad_token,
            mask_token=mask_token,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            model_max_length=model_max_length,
            **kwargs,
        )

    def get_vocab(self) -> Dict[str, int]:
        return dict(self.vocab)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize a pre-computed Group SELFIES string."""
        return self._inner.tokenize_group_selfies(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab.get("[UNK]", 0))

    def _convert_id_to_token(self, index: int) -> str:
        return self.id_to_token.get(index, "[UNK]")

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        filtered = [t for t in tokens if t not in self.SPECIAL_TOKENS]
        return "".join(filtered)

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
    ) -> List[int]:
        if token_ids_1 is not None:
            raise NotImplementedError("Pair sequences are not supported.")
        return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            special_ids = {self.pad_token_id, self.mask_token_id, self.bos_token_id, self.eos_token_id}
            return [1 if tok in special_ids else 0 for tok in token_ids_0]
        if token_ids_1 is not None:
            raise NotImplementedError("Pair sequences are not supported.")
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
    ) -> List[int]:
        if token_ids_1 is not None:
            raise NotImplementedError("Pair sequences are not supported.")
        return [0] * (len(token_ids_0) + 2)

    # ------------------------------------------------------------------
    # Group SELFIES-specific pipeline APIs
    # ------------------------------------------------------------------
    def encode_smiles(
        self,
        smiles: str,
        add_special_tokens: bool = True,
        padding: bool = True,
        return_attention_mask: bool = True,
        canonicalize: bool = True,
    ) -> Dict[str, List[int]]:
        """Full path: p-SMILES -> Group SELFIES -> token IDs (requires grammar)."""
        return self._inner.encode(
            smiles,
            add_special_tokens=add_special_tokens,
            padding=padding,
            return_attention_mask=return_attention_mask,
            canonicalize=canonicalize,
        )

    def encode_group_selfies(
        self,
        gsf_string: str,
        add_special_tokens: bool = True,
        padding: bool = True,
        return_attention_mask: bool = True,
    ) -> Dict[str, List[int]]:
        """Encode a pre-computed Group SELFIES string to token IDs."""
        return self._inner.encode_group_selfies(
            gsf_string,
            add_special_tokens=add_special_tokens,
            padding=padding,
            return_attention_mask=return_attention_mask,
        )

    def decode(self, ids: List[int], skip_special_tokens: bool = True, **kwargs) -> str:  # type: ignore[override]
        return self._inner.decode(ids, skip_special_tokens=skip_special_tokens)

    def batch_encode(
        self,
        smiles_list: List[str],
        add_special_tokens: bool = True,
        padding: bool = True,
    ) -> Dict[str, List[List[int]]]:
        encoded = [
            self.encode_smiles(s, add_special_tokens=add_special_tokens, padding=padding)
            for s in smiles_list
        ]
        return {
            "input_ids": [r["input_ids"] for r in encoded],
            "attention_mask": [r["attention_mask"] for r in encoded],
        }

    def batch_decode(
        self,
        ids_list: List[List[int]],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in ids_list]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str, str]:
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{filename_prefix}-" if filename_prefix else ""
        vocab_path = save_dir / f"{prefix}vocab.json"
        pkl_path = save_dir / f"{prefix}tokenizer.pkl"
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, indent=2)
        self._inner.save(str(pkl_path))
        return str(vocab_path), str(pkl_path)

    def get_placeholder_token_id(self) -> Optional[int]:
        return self._inner.get_placeholder_token_id()

    def get_star_token_id(self) -> int:
        return self._inner.get_star_token_id()


def load_polymer_tokenizer(
    results_dir: Union[str, Path],
    fallback_results_dir: Optional[Union[str, Path]] = None,
) -> HFGroupSelfiesTokenizer:
    """Load tokenizer, preferring HF-pretrained assets with legacy fallback."""
    candidates: List[Path] = []
    for root in [results_dir, fallback_results_dir]:
        if root is None:
            continue
        root_path = Path(root)
        candidates.append(root_path / "tokenizer_hf")
        candidates.append(root_path / "tokenizer.pkl")

    for candidate in candidates:
        if candidate.is_dir():
            return HFGroupSelfiesTokenizer.from_pretrained(str(candidate), local_files_only=True)
        if candidate.is_file() and candidate.suffix == ".pkl":
            from .tokenizer import GroupSELFIESTokenizer
            inner = GroupSELFIESTokenizer.load(str(candidate))
            tok = HFGroupSelfiesTokenizer(vocab=inner.vocab, max_length=inner.max_length)
            tok._inner = inner
            tok.vocab = dict(inner.vocab)
            tok.id_to_token = {v: k for k, v in inner.vocab.items()}
            return tok

    searched = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Tokenizer not found. Checked: {searched}")
