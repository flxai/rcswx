"""Independent scalar ChaCha12 and exhaustive mask/CDF reference."""

import hashlib
import math
import struct

from .ordered import projection


def chacha12_words(seed):
    key = hashlib.sha256(b"rcswx-rng-v1\0" + seed.to_bytes(8, "little") + b"selection").digest()
    constants = list(struct.unpack("<4I", b"expand 32-byte k"))
    counter = 0

    def rotate(value, bits):
        return ((value << bits) | (value >> (32 - bits))) & 0xFFFFFFFF

    def quarter(x, a, b, c, d):
        x[a] = (x[a] + x[b]) & 0xFFFFFFFF
        x[d] = rotate(x[d] ^ x[a], 16)
        x[c] = (x[c] + x[d]) & 0xFFFFFFFF
        x[b] = rotate(x[b] ^ x[c], 12)
        x[a] = (x[a] + x[b]) & 0xFFFFFFFF
        x[d] = rotate(x[d] ^ x[a], 8)
        x[c] = (x[c] + x[d]) & 0xFFFFFFFF
        x[b] = rotate(x[b] ^ x[c], 7)

    while True:
        initial = (
            constants
            + list(struct.unpack("<8I", key))
            + [counter & 0xFFFFFFFF, counter >> 32, 0, 0]
        )
        block = initial.copy()
        for _ in range(6):
            for indices in (
                (0, 4, 8, 12),
                (1, 5, 9, 13),
                (2, 6, 10, 14),
                (3, 7, 11, 15),
                (0, 5, 10, 15),
                (1, 6, 11, 12),
                (2, 7, 8, 13),
                (3, 4, 9, 14),
            ):
                quarter(block, *indices)
        words = [(a + b) & 0xFFFFFFFF for a, b in zip(initial, block, strict=True)]
        for i in range(0, 16, 2):
            yield words[i] | (words[i + 1] << 32)
        counter += 1


def distribution(left, right, witness):
    edits = sum(tag != 0 for tag, _, _ in witness)
    if edits > 12:
        raise ValueError("reference mask enumeration is limited to twelve edits")
    _, _, distance = projection(left, right, witness, (1 << edits) - 1)
    cumulative, total, correction = [], 0.0, 0.0
    for mask in range(1 << edits):
        output, _, cost = projection(left, right, witness, mask)
        if not output:
            continue
        weight = 1.0 if distance == 0 else math.exp(-0.5 * ((cost / distance - 0.5) / 0.2) ** 2)
        updated = total + weight
        correction += (
            ((total - updated) + weight)
            if abs(total) >= abs(weight)
            else ((weight - updated) + total)
        )
        total = updated
        cumulative.append((mask, total + correction))
    return tuple(cumulative)


def draws(left, right, witness, seed, count):
    cumulative = distribution(left, right, witness)
    words = chacha12_words(seed)
    result = []
    for _ in range(count):
        threshold = (next(words) >> 11) * 2.0**-53 * cumulative[-1][1]
        result.append(next(mask for mask, probability in cumulative if probability > threshold))
    return result
