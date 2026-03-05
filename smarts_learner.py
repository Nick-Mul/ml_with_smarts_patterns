#!/usr/bin/env python3
"""
SMARTS Pattern Learner using Genetic Algorithm
Learns discriminative SMARTS patterns from active/inactive molecules
"""

import argparse
import random
import csv
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional
from collections import defaultdict

try:
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    from rdkit import DataStructs
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
    generation: int


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


class FragmentGenerator:
    def __init__(self, dataset: MoleculeDataset):
        self.dataset = dataset

    def generate_fragments(self, max_fragments: int = 200) -> List[str]:
        fragments = set()
        actives_mols, inactives_mols = self.dataset.get_all_mols()
        all_mols = actives_mols + inactives_mols

        for mol in all_mols:
            try:
                bi = {}
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, bitInfo=bi)
                for bit in fp.GetOnBits():
                    env = bi[bit]
                    atom_infos = env[0][1] if isinstance(env[0], tuple) else [env]
                    for ai in (
                        atom_infos if isinstance(atom_infos[0], list) else [atom_infos]
                    ):
                        try:
                            radius = env[0][0] if isinstance(env[0], tuple) else 2
                            amap = {}
                            sma = AllChem.GenerateSubsetsOfRadiusMolecule(
                                mol, ai[0], radius, atomMap=amap
                            )
                            if sma:
                                fragments.add(Chem.MolToSmarts(sma))
                        except:
                            pass
            except:
                pass

            try:
                for atom in mol.GetAtoms():
                    idx = atom.GetIdx()
                    try:
                        env = Chem.FindAtomEnvironmentOfRadiusN(mol, 1, idx)
                        if env:
                            amap = {}
                            submol = Chem.PathToSubmol(mol, env, atomMap=amap)
                            if submol and submol.GetNumAtoms() > 0:
                                smarts = Chem.MolToSmarts(submol)
                                if smarts:
                                    fragments.add(smarts)
                    except:
                        pass
            except:
                pass

        fragments = list(fragments)[:max_fragments]
        return fragments if fragments else ["C", "CC", "CCC", "c1ccccc1"]


class SMARTSMutator:
    ATOM_TOKENS = [
        "C",
        "N",
        "O",
        "S",
        "P",
        "F",
        "Cl",
        "Br",
        "I",
        "c",
        "n",
        "o",
        "s",
        "p",
        "[nH]",
        "[NH]",
        "[NH2]",
        "[C@@H]",
        "[C@H]",
        "[CH]",
        "[CH2]",
        "[CH3]",
        "#",
        "=",
        "@",
        ":",
        "~",
    ]

    BRANCH_TOKENS = ["(", ")", "[", "]"]

    def __init__(self):
        self.atom_expressions = [
            "",
            "!",
            "?",
            "+",
            "-",
            "^",
            "%d",
            "%e",
            "H",
            "h",
            "r",
            "x",
            "X",
            "v",
            "V",
            "z",
            "Z",
            "#",
            "$",
        ]

    def mutate(self, smarts: str) -> str:
        mutations = [
            lambda: self._add_atom(smarts),
            lambda: self._remove_atom(smarts),
            lambda: self._add_charge(smarts),
            lambda: self._remove_charge(smarts),
            lambda: self._add_ring(smarts),
            lambda: self._add_branch(smarts),
            lambda: self._change_bond(smarts),
            lambda: self._add_hydrogen(smarts),
            lambda: self._toggle_aromatic(smarts),
        ]
        return random.choice(mutations)()

    def _add_atom(self, smarts: str) -> str:
        if random.random() < 0.5:
            pos = random.randint(0, len(smarts))
            atom = random.choice(self.ATOM_TOKENS)
            return smarts[:pos] + atom + smarts[pos:]
        else:
            parts = smarts.split()
            if not parts:
                return "C"
            idx = random.randint(0, len(parts) - 1)
            atom = random.choice(self.ATOM_TOKENS)
            parts[idx] += atom
            return " ".join(parts)

    def _remove_atom(self, smarts: str) -> str:
        if len(smarts) <= 1:
            return "C"
        chars = list(smarts)
        idx = random.randint(0, len(chars) - 1)
        del chars[idx]
        result = "".join(chars)
        return result if result else "C"

    def _add_charge(self, smarts: str) -> str:
        tokens = ["[N+]", "[N-]", "[O+]", "[O-]", "[C+]", "[C-]"]
        return smarts + random.choice(tokens)

    def _remove_charge(self, smarts: str) -> str:
        for charge in ["+", "-", "++", "--", "+", "-"]:
            if charge in smarts:
                smarts = smarts.replace(charge, "", 1)
        return smarts if smarts else "C"

    def _add_ring(self, smarts: str) -> str:
        rings = ["1", "2", "3", "r", "R", "r1", "R1", "r2", "R2"]
        return smarts + random.choice(rings)

    def _add_branch(self, smarts: str) -> str:
        branches = ["()", "[]", "(C)", "(N)", "(O)", "(CC)", "(CCC)"]
        return smarts + random.choice(branches)

    def _change_bond(self, smarts: str) -> str:
        bonds = ["~", "#", "=", ":", "-"]
        bond = random.choice(bonds)
        if bond in smarts:
            idx = smarts.index(bond)
            new_bond = random.choice([b for b in bonds if b != bond])
            return smarts[:idx] + new_bond + smarts[idx + 1 :]
        return smarts + bond

    def _add_hydrogen(self, smarts: str) -> str:
        if "H" not in smarts:
            return "[CH]" + smarts
        return smarts.replace("H", "h", 1)

    def _toggle_aromatic(self, smarts: str) -> str:
        if "c" in smarts:
            return smarts.replace("c", "C", 1)
        elif "n" in smarts:
            return smarts.replace("n", "N", 1)
        return "c" + smarts


class SMARTSCrossover:
    def crossover(self, parent1: str, parent2: str) -> str:
        if random.random() < 0.5:
            return self._single_point_crossover(parent1, parent2)
        return self._fragment_swap(parent1, parent2)

    def _single_point_crossover(self, p1: str, p2: str) -> str:
        if len(p1) < 2 or len(p2) < 2:
            return p1 if random.random() < 0.5 else p2

        pt1 = random.randint(1, len(p1) - 1)
        pt2 = random.randint(1, len(p2) - 1)

        if random.random() < 0.5:
            return p1[:pt1] + p2[pt2:]
        else:
            return p2[:pt2] + p1[pt1:]

    def _fragment_swap(self, p1: str, p2: str) -> str:
        tokens1 = p1.replace("(", " ( ").replace(")", " ) ").split()
        tokens2 = p2.replace("(", " ( ").replace(")", " ) ").split()

        if not tokens1 or not tokens2:
            return p1

        sublen = random.randint(1, min(len(tokens1), len(tokens2)))
        start1 = random.randint(0, max(0, len(tokens1) - sublen))
        start2 = random.randint(0, max(0, len(tokens2) - sublen))

        child_tokens = (
            tokens1[:start1]
            + tokens2[start2 : start2 + sublen]
            + tokens1[start1 + sublen :]
        )
        return "".join(child_tokens)


class PatternScorer:
    def __init__(self, dataset: MoleculeDataset):
        self.dataset = dataset
        self.actives_mols = [d.mol for d in dataset.actives]
        self.inactives_mols = [d.mol for d in dataset.inactives]

    def score(self, smarts: str) -> Tuple[float, float, float, float, float, int, int]:
        try:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is None:
                return 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0
        except Exception:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0

        try:
            hits_actives = sum(
                1 for mol in self.actives_mols if mol.HasSubstructMatch(pattern)
            )
            hits_inactives = sum(
                1 for mol in self.inactives_mols if mol.HasSubstructMatch(pattern)
            )
        except Exception:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0

        n_actives = len(self.actives_mols)
        n_inactives = len(self.inactives_mols)
        n_total = n_actives + n_inactives

        support_actives = hits_actives / n_actives if n_actives > 0 else 0
        support_inactives = hits_inactives / n_inactives if n_inactives > 0 else 0

        if hits_actives == 0 and hits_inactives == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, hits_actives, hits_inactives

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

        return fitness, ef, mcc, f1, tanimoto_diff, hits_actives, hits_inactives

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


class GeneticAlgorithm:
    def __init__(
        self,
        dataset: MoleculeDataset,
        population_size: int = 100,
        generations: int = 50,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.5,
        elitism: int = 5,
    ):
        self.dataset = dataset
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism

        self.fragment_gen = FragmentGenerator(dataset)
        self.mutator = SMARTSMutator()
        self.crossover = SMARTSCrossover()
        self.scorer = PatternScorer(dataset)

        self.population: List[Tuple[str, float]] = []
        self.best_patterns: List[PatternResult] = []
        self.generation_stats = []

    def initialize_population(self, fragments: List[str], random_count: int = 20):
        population = []
        seen = set()

        for frag in fragments[: self.population_size - random_count]:
            if frag not in seen:
                seen.add(frag)
                fitness, _, _, _, _, _, _ = self.scorer.score(frag)
                population.append((frag, fitness))

        random_smarts = [
            "C",
            "CC",
            "CCC",
            "CCCC",
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
            "C(=O)C",
            "CC(=O)C",
            "c1cccnc1",
            "c1ccncc1",
            "c1cnccn1",
            "n1cccc1",
            "C1CCCCC1",
            "C1CCC1",
            "c1ccc2ccccc2c1",
            "C1CCOCC1",
        ]

        for smi in random_smarts[:random_count]:
            try:
                mol = Chem.MolFromSmiles(smi)
                smarts = Chem.MolToSmarts(mol)
                if smarts and smarts not in seen:
                    seen.add(smarts)
                    fitness, _, _, _, _, _, _ = self.scorer.score(smarts)
                    population.append((smarts, fitness))
            except:
                pass

        while len(population) < self.population_size:
            new_smarts = self.mutator._add_atom(random.choice(list(seen)))
            if new_smarts and new_smarts not in seen:
                seen.add(new_smarts)
                fitness, _, _, _, _, _, _ = self.scorer.score(new_smarts)
                population.append((new_smarts, fitness))

        self.population = population[: self.population_size]

    def select_parent(self) -> str:
        tournament_size = 3
        tournament = random.sample(
            self.population, min(tournament_size, len(self.population))
        )
        return max(tournament, key=lambda x: x[1])[0]

    def evolve(self):
        print(
            f"Starting GA with population {self.population_size}, generations {self.generations}"
        )

        fragments = self.fragment_gen.generate_fragments(max_fragments=150)
        self.initialize_population(fragments)

        for gen in range(self.generations):
            new_population = []

            self.population.sort(key=lambda x: x[1], reverse=True)
            elite = self.population[: self.elitism]
            new_population.extend(elite)

            while len(new_population) < self.population_size:
                if random.random() < self.crossover_rate:
                    p1 = self.select_parent()
                    p2 = self.select_parent()
                    child = self.crossover.crossover(p1, p2)
                else:
                    child = self.select_parent()

                if random.random() < self.mutation_rate:
                    child = self.mutator.mutate(child)

                fitness, ef, mcc, f1, tan_diff, supp_a, supp_i = self.scorer.score(
                    child
                )
                new_population.append((child, fitness))

            self.population = new_population[: self.population_size]

            best = max(self.population, key=lambda x: x[1])
            avg_fitness = sum(p[1] for p in self.population) / len(self.population)

            self.generation_stats.append(
                {
                    "generation": gen + 1,
                    "best_fitness": best[1],
                    "avg_fitness": avg_fitness,
                }
            )

            print(
                f"Gen {gen + 1}: Best fitness = {best[1]:.4f}, Avg = {avg_fitness:.4f}, Best pattern: {best[0]}"
            )

            for smarts, fitness in elite[:3]:
                if not any(p.smarts == smarts for p in self.best_patterns):
                    _, ef, mcc, f1, tan_diff, supp_a, supp_i = self.scorer.score(smarts)
                    self.best_patterns.append(
                        PatternResult(
                            smarts=smarts,
                            mol=None,
                            fitness=fitness,
                            ef=ef,
                            mcc=mcc,
                            f1=f1,
                            tanimoto_diff=tan_diff,
                            support_actives=supp_a,
                            support_inactives=supp_i,
                            generation=gen + 1,
                        )
                    )

        self.best_patterns.sort(key=lambda x: x.fitness, reverse=True)

    def get_results(self, top_n: int = 50) -> List[PatternResult]:
        return self.best_patterns[:top_n]


def save_results_csv(results: List[PatternResult], output_file: str, stats: List[dict]):
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["# SMARTS Pattern Learning Results"])
        writer.writerow(["# Generated by smarts_learner.py"])
        writer.writerow(["#"])

        if stats:
            writer.writerow(["# Generation Statistics"])
            writer.writerow(["# Generation,Best_Fitness,Avg_Fitness"])
            for s in stats:
                writer.writerow(
                    [
                        f"# {s['generation']},{s['best_fitness']:.4f},{s['avg_fitness']:.4f}"
                    ]
                )
            writer.writerow(["#"])

        writer.writerow(
            [
                "SMARTS,Fitness,Enrichment_Factor,MCC,F1_Score,Tanimoto_Diff,Support_Actives,Support_Inactives,Generation"
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
                    r.generation,
                ]
            )

    print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Learn discriminative SMARTS patterns using Genetic Algorithm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python smarts_learner.py --actives actives.smi --inactives inactives.smi
  python smarts_learner.py -a actives.smi -i inactives.smi -o patterns.csv -p 150 -g 100
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
        default="ranked_patterns.csv",
        help="Output CSV file (default: ranked_patterns.csv)",
    )
    parser.add_argument(
        "--population",
        "-p",
        type=int,
        default=100,
        help="Population size (default: 100)",
    )
    parser.add_argument(
        "--generations",
        "-g",
        type=int,
        default=50,
        help="Number of generations (default: 50)",
    )
    parser.add_argument(
        "--mutation-rate",
        "-m",
        type=float,
        default=0.3,
        help="Mutation rate (default: 0.3)",
    )
    parser.add_argument(
        "--crossover-rate",
        "-c",
        type=float,
        default=0.5,
        help="Crossover rate (default: 0.5)",
    )
    parser.add_argument(
        "--top-n",
        "-n",
        type=int,
        default=50,
        help="Number of top patterns to save (default: 50)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("SMARTS Pattern Learner - Genetic Algorithm")
    print("=" * 60)

    start_time = time.time()

    dataset = MoleculeDataset(args.actives, args.inactives)

    ga = GeneticAlgorithm(
        dataset=dataset,
        population_size=args.population,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
    )

    ga.evolve()

    results = ga.get_results(top_n=args.top_n)
    save_results_csv(results, args.output, ga.generation_stats)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f} seconds")
    print(f"Top 5 patterns:")
    for i, r in enumerate(results[:5], 1):
        print(f"  {i}. {r.smarts} (fitness: {r.fitness:.4f})")


if __name__ == "__main__":
    main()
