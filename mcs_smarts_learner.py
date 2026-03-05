#!/usr/bin/env python3
"""
SMARTS Pattern Learner using Maximum Common Substructure (MCS)
Finds common substructures in actives and evaluates discriminative power vs inactives
"""

import argparse
import csv
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set
from collections import defaultdict

try:
    from rdkit import Chem
    from rdkit import RDLogger
    from rdkit.Chem import rdFMCS
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    from rdkit import DataStructs

    RDLogger.DisableLog("rdApp.*")
except ImportError:
    raise ImportError("RDKit is required. Install with: pip install rdkit")


@dataclass
class MoleculeData:
    smiles: str
    mol: Optional[Chem.Mol]
    is_active: bool


@dataclass
class PatternResult:
    smarts: str
    mol: Optional[Chem.Mol]
    fitness: float
    ef: float
    mcc: float
    f1: float
    tanimoto_diff: float
    support_actives: int
    support_inactives: int
    num_atoms: int
    num_rings: int


class MoleculeDataset:
    def __init__(self, actives_file: str, inactives_file: str):
        self.actives: List[MoleculeData] = []
        self.inactives: List[MoleculeData] = []
        self._load_dataset(actives_file, inactives_file)

    def _load_dataset(self, actives_file: str, inactives_file: str):
        for file_path, is_active in [(actives_file, True), (inactives_file, False)]:
            with open(file_path, "r") as f:
                for line in f:
                    smiles = line.strip().split()[0]
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        data = MoleculeData(smiles=smiles, mol=mol, is_active=is_active)
                        if is_active:
                            self.actives.append(data)
                        else:
                            self.inactives.append(data)

        print(f"Loaded {len(self.actives)} actives, {len(self.inactives)} inactives")

    def get_all_mols(self) -> Tuple[List[Chem.Mol], List[Chem.Mol]]:
        return [d.mol for d in self.actives], [d.mol for d in self.inactives]


class MCSPatternFinder:
    def __init__(
        self,
        dataset: MoleculeDataset,
        min_atoms: int = 3,
        max_atoms: int = 15,
        min_actives_fraction: float = 0.3,
        ring_comparison: bool = True,
    ):
        self.dataset = dataset
        self.min_atoms = min_atoms
        self.max_atoms = max_atoms
        self.min_actives_fraction = min_actives_fraction
        self.ring_comparison = ring_comparison
        self.actives_mols = [d.mol for d in dataset.actives]
        self.min_support = int(len(self.actives_mols) * min_actives_fraction)

    def find_mcs_patterns(self, max_patterns: int = 200) -> List[str]:
        patterns: Set[str] = set()
        remaining_mols = self.actives_mols.copy()

        print(
            f"Searching for MCS patterns (min support: {self.min_support}/{len(self.actives_mols)})"
        )

        mcs_result = rdFMCS.FindMCS(
            remaining_mols,
            threshold=0.5,
            ringMatchesRingOnly=self.ring_comparison,
            completeRingsOnly=False,
            timeout=30,
        )

        if mcs_result.smartsString:
            smarts = mcs_result.smartsString
            if self._is_valid_pattern(smarts):
                patterns.add(smarts)

        for i in range(len(remaining_mols)):
            for j in range(i + 1, len(remaining_mols)):
                pair_mcs = rdFMCS.FindMCS(
                    [remaining_mols[i], remaining_mols[j]],
                    threshold=0.5,
                    ringMatchesRingOnly=self.ring_comparison,
                    completeRingsOnly=False,
                    timeout=10,
                )
                if pair_mcs.smartsString:
                    smarts = pair_mcs.smartsString
                    if self._is_valid_pattern(smarts):
                        patterns.add(smarts)
                if len(patterns) >= max_patterns:
                    break
            if len(patterns) >= max_patterns:
                break

        if len(patterns) < 10:
            for i in range(len(remaining_mols)):
                for j in range(i + 1, len(remaining_mols)):
                    for k in range(j + 1, len(remaining_mols)):
                        triple_mcs = rdFMCS.FindMCS(
                            [remaining_mols[i], remaining_mols[j], remaining_mols[k]],
                            threshold=0.3,
                            ringMatchesRingOnly=self.ring_comparison,
                            completeRingsOnly=False,
                            timeout=10,
                        )
                        if triple_mcs.smartsString:
                            smarts = triple_mcs.smartsString
                            if self._is_valid_pattern(smarts):
                                patterns.add(smarts)
                        if len(patterns) >= max_patterns:
                            break
                    if len(patterns) >= max_patterns:
                        break
                if len(patterns) >= max_patterns:
                    break

        direct_mcs = list(patterns)
        print(f"Found {len(direct_mcs)} direct MCS patterns")

        variations = self._generate_pattern_variations(direct_mcs)
        all_patterns = list(patterns | variations)
        print(f"Total patterns after variations: {len(all_patterns)}")

        return all_patterns[:max_patterns]

    def _is_valid_pattern(self, smarts: str) -> bool:
        try:
            mol = Chem.MolFromSmarts(smarts)
            if mol is None:
                return False
            num_atoms = mol.GetNumAtoms()
            if num_atoms < self.min_atoms or num_atoms > self.max_atoms:
                return False
            return True
        except:
            return False

    def _generate_pattern_variations(self, base_patterns: List[str]) -> Set[str]:
        variations: Set[str] = set()
        atom_expressions = [
            ("C", "c"),
            ("N", "n"),
            ("O", "o"),
            ("S", "s"),
            ("[CH]", "[CH2]"),
            ("[CH2]", "[CH3]"),
            ("[N", "[n"),
            ("n1", "c1"),
            ("c1", "n1"),
            ("=", "#"),
            ("-", "="),
        ]

        for smarts in base_patterns[:50]:
            for old, new in atom_expressions:
                if old in smarts and old != new:
                    var = smarts.replace(old, new, 1)
                    if self._is_valid_pattern(var):
                        variations.add(var)

            try:
                mol = Chem.MolFromSmarts(smarts)
                if mol:
                    ring_info = mol.GetRingInfo()
                    if ring_info.NumRings() > 0:
                        aromatic = Chem.MolToSmarts(mol)
                        if aromatic != smarts:
                            variations.add(aromatic)

                    for atom in mol.GetAtoms():
                        if atom.GetSymbol() in ["C", "N", "O", "S"]:
                            idx = atom.GetIdx()
                            sma = Chem.MolToSmarts(mol)
                            parts = list(sma)
                            if idx < len(parts):
                                variations.add(sma)
            except:
                pass

        base_subs = [
            "c1ccccc1",
            "c1ccc(O)cc1",
            "c1ccc(N)cc1",
            "c1ccc(F)cc1",
            "C(=O)O",
            "C(=O)N",
            "CN",
            "CO",
            "CS",
            "C#N",
            "c1cccnc1",
            "c1ccncc1",
            "c1cnccn1",
            "C1CCCCC1",
            "C1CCC1",
        ]
        for sub in base_subs:
            if self._is_valid_pattern(sub):
                variations.add(sub)

        return variations


class PatternScorer:
    def __init__(self, dataset: MoleculeDataset):
        self.dataset = dataset
        self.actives_mols = [d.mol for d in dataset.actives]
        self.inactives_mols = [d.mol for d in dataset.inactives]

    def score(
        self, smarts: str
    ) -> Tuple[float, float, float, float, float, int, int, int, int]:
        try:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is None:
                return 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0
        except Exception:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0

        try:
            hits_actives = sum(
                1 for mol in self.actives_mols if mol.HasSubstructMatch(pattern)
            )
            hits_inactives = sum(
                1 for mol in self.inactives_mols if mol.HasSubstructMatch(pattern)
            )
        except Exception:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0

        n_actives = len(self.actives_mols)
        n_inactives = len(self.inactives_mols)
        n_total = n_actives + n_inactives

        support_actives = hits_actives / n_actives if n_actives > 0 else 0
        support_inactives = hits_inactives / n_inactives if n_inactives > 0 else 0

        if hits_actives == 0 and hits_inactives == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, hits_actives, hits_inactives, 0, 0

        ef = self._enrichment_factor(
            hits_actives, hits_inactives, n_actives, n_inactives
        )
        mcc = self._matthews_corr(hits_actives, hits_inactives, n_actives, n_inactives)
        f1 = self._f1_score(hits_actives, hits_inactives, n_actives, n_inactives)
        tanimoto_diff = self._tanimoto_difference(pattern)

        fitness = (
            0.3 * self._normalize_ef(ef)
            + 0.3 * (mcc + 1) / 2
            + 0.2 * f1
            + 0.2 * tanimoto_diff
        )

        num_atoms = pattern.GetNumAtoms()
        try:
            num_rings = pattern.GetRingInfo().NumRings() if pattern else 0
        except:
            num_rings = 0

        return (
            fitness,
            ef,
            mcc,
            f1,
            tanimoto_diff,
            hits_actives,
            hits_inactives,
            num_atoms,
            num_rings,
        )

    def _enrichment_factor(self, hits_a: int, hits_i: int, n_a: int, n_i: int) -> float:
        if n_a == 0 or (hits_a + hits_i) == 0:
            return 0.0
        total_hits = hits_a + hits_i
        frac_actives = hits_a / n_a if n_a > 0 else 0
        frac_total = total_hits / (n_a + n_i) if (n_a + n_i) > 0 else 0
        if frac_total == 0:
            return 0.0
        return frac_actives / frac_total

    def _matthews_corr(self, hits_a: int, hits_i: int, n_a: int, n_i: int) -> float:
        tp = hits_a
        fp = hits_i
        fn = n_a - hits_a
        tn = n_i - hits_i

        numerator = (tp * tn) - (fp * fn)
        denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

        if denominator == 0:
            return 0.0
        return numerator / denominator

    def _f1_score(self, hits_a: int, hits_i: int, n_a: int, n_i: int) -> float:
        precision = hits_a / (hits_a + hits_i) if (hits_a + hits_i) > 0 else 0
        recall = hits_a / n_a if n_a > 0 else 0

        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def _tanimoto_difference(self, pattern: Chem.Mol) -> float:
        try:
            pattern_fp = AllChem.GetMorganFingerprintAsBitVect(pattern, 2)
        except:
            return 0.0

        active_fps = [
            AllChem.GetMorganFingerprintAsBitVect(m, 2) for m in self.actives_mols[:50]
        ]

        if not active_fps:
            return 0.0

        avg_tan = sum(
            DataStructs.TanimotoSimilarity(pattern_fp, fp) for fp in active_fps
        ) / len(active_fps)

        inactive_fps = [
            AllChem.GetMorganFingerprintAsBitVect(m, 2)
            for m in self.inactives_mols[:50]
        ]
        if inactive_fps:
            avg_tan_i = sum(
                DataStructs.TanimotoSimilarity(pattern_fp, fp) for fp in inactive_fps
            ) / len(inactive_fps)
            return max(0, avg_tan - avg_tan_i)

        return avg_tan

    def _normalize_ef(self, ef: float) -> float:
        return min(1.0, ef / 20.0)


def find_mcs_patterns_main(
    dataset: MoleculeDataset,
    min_atoms: int = 3,
    max_atoms: int = 15,
    min_fraction: float = 0.3,
    max_patterns: int = 200,
) -> List[PatternResult]:
    mcs_finder = MCSPatternFinder(
        dataset,
        min_atoms=min_atoms,
        max_atoms=max_atoms,
        min_actives_fraction=min_fraction,
    )

    patterns = mcs_finder.find_mcs_patterns(max_patterns=max_patterns)

    scorer = PatternScorer(dataset)

    results: List[PatternResult] = []
    seen_smarts = set()

    print(f"Scoring {len(patterns)} patterns...")

    for i, smarts in enumerate(patterns):
        if smarts in seen_smarts:
            continue
        seen_smarts.add(smarts)

        if (i + 1) % 50 == 0:
            print(f"  Scored {i + 1}/{len(patterns)} patterns")

        try:
            pattern_mol = Chem.MolFromSmarts(smarts)
            if pattern_mol is None:
                continue
        except:
            continue

        fitness, ef, mcc, f1, tan_diff, supp_a, supp_i, num_atoms, num_rings = (
            scorer.score(smarts)
        )

        if supp_a > 0:
            results.append(
                PatternResult(
                    smarts=smarts,
                    mol=pattern_mol,
                    fitness=fitness,
                    ef=ef,
                    mcc=mcc,
                    f1=f1,
                    tanimoto_diff=tan_diff,
                    support_actives=supp_a,
                    support_inactives=supp_i,
                    num_atoms=num_atoms,
                    num_rings=num_rings,
                )
            )

    results.sort(key=lambda x: x.fitness, reverse=True)

    print(f"Found {len(results)} valid patterns")

    unique_results = []
    seen_patterns = set()
    for r in results:
        if r.smarts not in seen_patterns:
            seen_patterns.add(r.smarts)
            unique_results.append(r)

    return unique_results


def save_results_csv(results: List[PatternResult], output_file: str, stats: dict):
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["# MCS-Based SMARTS Pattern Learning Results"])
        writer.writerow(["# Generated by mcs_smarts_learner.py"])
        writer.writerow(["#"])

        if stats:
            writer.writerow(["# Configuration Statistics"])
            for key, value in stats.items():
                writer.writerow([f"# {key}: {value}"])
            writer.writerow(["#"])

        writer.writerow(
            [
                "SMARTS,Fitness,Enrichment_Factor,MCC,F1_Score,Tanimoto_Diff,Support_Actives,Support_Inactives,Num_Atoms,Num_Rings"
            ]
        )

        for r in results:
            writer.writerow(
                [
                    r.smarts,
                    f"{r.fitness:.4f}",
                    f"{r.ef:.4f}",
                    f"{r.mcc:.4f}",
                    f"{r.f1:.4f}",
                    f"{r.tanimoto_diff:.4f}",
                    r.support_actives,
                    r.support_inactives,
                    r.num_atoms,
                    r.num_rings,
                ]
            )

    print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Learn discriminative SMARTS patterns using Maximum Common Substructure (MCS)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mcs_smarts_learner.py --actives actives.smi --inactives inactives.smi
  python mcs_smarts_learner.py -a actives.smi -i inactives.smi -o mcs_patterns.csv --max-patterns 300
  python mcs_smarts_learner.py -a actives.smi -i inactives.smi --min-atoms 4 --max-atoms 20
        """,
    )

    parser.add_argument(
        "--actives",
        "-a",
        required=True,
        help="File containing active molecules (one SMILES per line)",
    )
    parser.add_argument(
        "--inactives",
        "-i",
        required=True,
        help="File containing inactive molecules (one SMILES per line)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="mcs_patterns.csv",
        help="Output CSV file (default: mcs_patterns.csv)",
    )
    parser.add_argument(
        "--min-atoms",
        type=int,
        default=3,
        help="Minimum atoms in pattern (default: 3)",
    )
    parser.add_argument(
        "--max-atoms",
        type=int,
        default=15,
        help="Maximum atoms in pattern (default: 15)",
    )
    parser.add_argument(
        "--min-fraction",
        type=float,
        default=0.3,
        help="Minimum fraction of actives that must match (default: 0.3)",
    )
    parser.add_argument(
        "--max-patterns",
        "-n",
        type=int,
        default=200,
        help="Maximum number of patterns to generate (default: 200)",
    )
    parser.add_argument(
        "--ring-comparison",
        action="store_true",
        default=True,
        help="Enable ring comparison in MCS finding (default: True)",
    )
    parser.add_argument(
        "--no-ring-comparison",
        action="store_false",
        dest="ring_comparison",
        help="Disable ring comparison in MCS finding",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("SMARTS Pattern Learner - MCS-Based Approach")
    print("=" * 60)

    start_time = time.time()

    dataset = MoleculeDataset(args.actives, args.inactives)

    results = find_mcs_patterns_main(
        dataset=dataset,
        min_atoms=args.min_atoms,
        max_atoms=args.max_atoms,
        min_fraction=args.min_fraction,
        max_patterns=args.max_patterns,
    )

    stats = {
        "num_actives": len(dataset.actives),
        "num_inactives": len(dataset.inactives),
        "min_atoms": args.min_atoms,
        "max_atoms": args.max_atoms,
        "min_fraction": args.min_fraction,
        "max_patterns": args.max_patterns,
    }

    save_results_csv(results, args.output, stats)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f} seconds")
    print(f"Found {len(results)} valid patterns")
    print(f"Top 5 patterns:")
    for i, r in enumerate(results[:5], 1):
        print(f"  {i}. {r.smarts} (fitness: {r.fitness:.4f}, EF: {r.ef:.2f})")


if __name__ == "__main__":
    main()
