module SMARTSLearner

using RDKitMinimalLib
using Random
using Statistics
using CSV
using DataFrames
using ArgParse

export MoleculeData, PatternResult, MoleculeDataset, SMARTSLearner

mutable struct MoleculeData
    smiles::String
    mol::Union{Mol, Nothing}
    is_active::Bool
end

mutable struct PatternResult
    smarts::String
    mol::Union{Mol, Nothing}
    fitness::Float64
    ef::Float64
    mcc::Float64
    f1::Float64
    tanimoto_diff::Float64
    support_actives::Int
    support_inactives::Int
    generation::Int
end

function Base.isless(a::PatternResult, b::PatternResult)
    return a.fitness < b.fitness
end

struct MoleculeDataset
    actives::Vector{MoleculeData}
    inactives::Vector{MoleculeData}
end

function MoleculeDataset(actives_file::String, inactives_file::String)
    actives = MoleculeData[]
    inactives = MoleculeData[]
    
    for (file_path, is_active) in [(actives_file, true), (inactives_file, false)]
        for line in eachline(file_path)
            smiles = strip(split(line)[1])
            mol = get_mol(smiles)
            if mol !== nothing
                data = MoleculeData(smiles=smiles, mol=mol, is_active=is_active)
                if is_active
                    push!(actives, data)
                else
                    push!(inactives, data)
                end
            end
        end
    end
    
    println("Loaded $(length(actives)) actives, $(length(inactives)) inactives")
    return MoleculeDataset(actives, inactives)
end

function get_all_mols(dataset::MoleculeDataset)
    actives_mols = [d.mol for d in dataset.actives if d.mol !== nothing]
    inactives_mols = [d.mol for d in dataset.inactives if d.mol !== nothing]
    return actives_mols, inactives_mols
end

struct FragmentGenerator
    dataset::MoleculeDataset
end

function generate_fragments(frag_gen::FragmentGenerator; max_fragments::Int=200)
    fragments = Set{String}()
    actives_mols, inactives_mols = get_all_mols(frag_gen.dataset)
    all_mols = vcat(actives_mols, inactives_mols)
    
    for mol in all_mols
        try
            details = Dict{String, Any}("radius" => 2, "nBits" => 2048)
            fp_str = get_morgan_fp(mol, details)
        catch
            continue
        end
    end
    
    fallback_fragments = ["C", "CC", "CCC", "c1ccccc1", "CCO", "CCN", "c1ccc(O)cc1", 
                          "c1ccc(N)cc1", "c1ccc(F)cc1", "C(=O)O", "C(=O)N", "CN", "CO", 
                          "CS", "C#N", "C(=O)C", "CC(=O)C", "c1cccnc1", "c1ccncc1",
                          "c1cnccn1", "n1cccc1", "C1CCCCC1", "C1CCC1", "c1ccc2ccccc2c1",
                          "C1CCOCC1", "C(=O)CC", "CC(C)C", "CCC(C)C", "c1cc(O)ccc1O",
                          "c1cc(N)ccc1N", "c1ccccc1Cl", "c1ccccc1Br"]
    
    if isempty(fragments)
        fragments = Set(fallback_fragments)
    end
    
    result = collect(fragments)
    if length(result) > max_fragments
        result = result[1:max_fragments]
    end
    
    return isempty(result) ? ["C", "CC", "CCC", "c1ccccc1"] : result
end

struct SMARTSMutator
    atom_tokens::Vector{String}
    branch_tokens::Vector{String}
    
    function SMARTSMutator()
        atom_tokens = [
            "C", "N", "O", "S", "P", "F", "Cl", "Br", "I",
            "c", "n", "o", "s", "p", "[nH]", "[NH]", "[NH2]",
            "[C@@H]", "[C@H]", "[CH]", "[CH2]", "[CH3]",
            "#", "=", "@", ":", "~"
        ]
        branch_tokens = ["(", ")", "[", "]"]
        new(atom_tokens, branch_tokens)
    end
end

function mutate(mutator::SMARTSMutator, smarts::String)
    mutations = [
        () -> add_atom(mutator, smarts),
        () -> remove_atom(smarts),
        () -> add_charge(smarts),
        () -> remove_charge(smarts),
        () -> add_ring(smarts),
        () -> add_branch(smarts),
        () -> change_bond(smarts),
        () -> add_hydrogen(smarts),
        () -> toggle_aromatic(smarts)
    ]
    return rand(mutations)()
end

function add_atom(mutator::SMARTSMutator, smarts::String)
    if rand() < 0.5
        pos = rand(1:max(1, length(smarts)+1))
        atom = rand(mutator.atom_tokens)
        return smarts[1:pos-1] * atom * smarts[pos:end]
    else
        parts = split(smarts)
        if isempty(parts)
            return "C"
        end
        idx = rand(1:length(parts))
        atom = rand(mutator.atom_tokens)
        parts[idx] = parts[idx] * atom
        return join(parts, " ")
    end
end

function remove_atom(smarts::String)
    if length(smarts) <= 1
        return "C"
    end
    chars = collect(smarts)
    idx = rand(1:length(chars))
    deleteat!(chars, idx)
    result = join(chars)
    return isempty(result) ? "C" : result
end

function add_charge(smarts::String)
    tokens = ["[N+]", "[N-]", "[O+]", "[O-]", "[C+]", "[C-]"]
    return smarts * rand(tokens)
end

function remove_charge(smarts::String)
    result = smarts
    for charge in ["++", "--", "+", "-"]
        result = replace(result, charge => "", count=1)
    end
    return isempty(result) ? "C" : result
end

function add_ring(smarts::String)
    rings = ["1", "2", "3", "r", "R", "r1", "R1", "r2", "R2"]
    return smarts * rand(rings)
end

function add_branch(smarts::String)
    branches = ["()", "[]", "(C)", "(N)", "(O)", "(CC)", "(CCC)"]
    return smarts * rand(branches)
end

function change_bond(smarts::String)
    bonds = ["~", "#", "=", ":", "-"]
    bond = rand(bonds)
    if bond in smarts
        idx = findfirst(bond, smarts)
        new_bond = rand([b for b in bonds if b != bond])
        return smarts[1:prevind(smarts, idx)] * new_bond * smarts[nextind(smarts, idx):end]
    end
    return smarts * bond
end

function add_hydrogen(smarts::String)
    if !occursin("H", smarts)
        return "[CH]" * smarts
    end
    return replace(smarts, "H" => "h", count=1)
end

function toggle_aromatic(smarts::String)
    if occursin("c", smarts)
        return replace(smarts, "c" => "C", count=1)
    elseif occursin("n", smarts)
        return replace(smarts, "n" => "N", count=1)
    end
    return "c" * smarts
end

struct SMARTSCrossover
end

function crossover(crossover::SMARTSCrossover, parent1::String, parent2::String)
    if rand() < 0.5
        return single_point_crossover(parent1, parent2)
    else
        return fragment_swap(parent1, parent2)
    end
end

function single_point_crossover(p1::String, p2::String)
    if length(p1) < 2 || length(p2) < 2
        return rand() < 0.5 ? p1 : p2
    end
    
    pt1 = rand(2:length(p1)-1)
    pt2 = rand(2:length(p2)-1)
    
    if rand() < 0.5
        return p1[1:prevind(p1, pt1)] * p2[pt2:end]
    else
        return p2[1:prevind(p2, pt2)] * p1[pt1:end]
    end
end

function fragment_swap(p1::String, p2::String)
    tokens1 = split(replace(p1, "(" => " ( ", ")" => " ) "))
    tokens2 = split(replace(p2, "(" => " ( ", ")" => " ) "))
    
    if isempty(tokens1) || isempty(tokens2)
        return p1
    end
    
    sublen = rand(1:min(length(tokens1), length(tokens2)))
    start1 = rand(1:max(1, length(tokens1) - sublen + 1))
    start2 = rand(1:max(1, length(tokens2) - sublen + 1))
    
    child_tokens = vcat(tokens1[1:start1-1], tokens2[start2:start2+sublen-1], tokens1[start1+sublen:end])
    return join(child_tokens)
end

struct PatternScorer
    dataset::MoleculeDataset
    actives_mols::Vector{Mol}
    inactives_mols::Vector{Mol}
    actives_fps::Vector{Vector{UInt8}}
    inactives_fps::Vector{Vector{UInt8}}
end

function PatternScorer(dataset::MoleculeDataset)
    actives_mols = [d.mol for d in dataset.actives if d.mol !== nothing]
    inactives_mols = [d.mol for d in dataset.inactives if d.mol !== nothing]
    
    details = Dict{String, Any}("radius" => 2, "nBits" => 2048)
    actives_fps = Vector{UInt8}[]
    inactives_fps = Vector{UInt8}[]
    
    for mol in actives_mols[1:min(50, length(actives_mols))]
        try
            fp = get_morgan_fp_as_bytes(mol, details)
            push!(actives_fps, fp)
        catch
        end
    end
    
    for mol in inactives_mols[1:min(50, length(inactives_mols))]
        try
            fp = get_morgan_fp_as_bytes(mol, details)
            push!(inactives_fps, fp)
        catch
        end
    end
    
    PatternScorer(dataset, actives_mols, inactives_mols, actives_fps, inactives_fps)
end

function score(scorer::PatternScorer, smarts::String)
    qmol = get_qmol(smarts)
    
    if qmol === nothing
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)
    end
    
    hits_actives = 0
    for mol in scorer.actives_mols
        try
            match = get_substruct_matches(mol, qmol)
            if haskey(match, "atoms") && !isempty(match["atoms"])
                hits_actives += 1
            end
        catch
        end
    end
    
    hits_inactives = 0
    for mol in scorer.inactives_mols
        try
            match = get_substruct_matches(mol, qmol)
            if haskey(match, "atoms") && !isempty(match["atoms"])
                hits_inactives += 1
            end
        catch
        end
    end
    
    n_actives = length(scorer.actives_mols)
    n_inactives = length(scorer.inactives_mols)
    
    if hits_actives == 0 && hits_inactives == 0
        return (0.0, 0.0, 0.0, 0.0, 0.0, hits_actives, hits_inactives)
    end
    
    ef = enrichment_factor(hits_actives, hits_inactives, n_actives, n_inactives)
    mcc = matthews_corr(hits_actives, hits_inactives, n_actives, n_inactives)
    f1 = f1_score(hits_actives, hits_inactives, n_actives, n_inactives)
    tanimoto_diff = tanimoto_difference(scorer, qmol)
    
    fitness = 0.3 * normalize_ef(ef) + 0.3 * (mcc + 1) / 2 + 0.2 * f1 + 0.2 * tanimoto_diff
    
    return (fitness, ef, mcc, f1, tanimoto_diff, hits_actives, hits_inactives)
end

function enrichment_factor(hits_a::Int, hits_i::Int, n_a::Int, n_i::Int)
    if n_a == 0 || (hits_a + hits_i) == 0
        return 0.0
    end
    frac_actives = hits_a / n_a
    total_hits = hits_a + hits_i
    frac_total = total_hits / (n_a + n_i)
    if frac_total == 0
        return 0.0
    end
    return frac_actives / frac_total
end

function matthews_corr(hits_a::Int, hits_i::Int, n_a::Int, n_i::Int)
    tp = hits_a
    fp = hits_i
    fn = n_a - hits_a
    tn = n_i - hits_i
    
    numerator = (tp * tn) - (fp * fn)
    denominator = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
    if denominator == 0
        return 0.0
    end
    return numerator / denominator
end

function f1_score(hits_a::Int, hits_i::Int, n_a::Int, n_i::Int)
    precision = (hits_a + hits_i) > 0 ? hits_a / (hits_a + hits_i) : 0.0
    recall = n_a > 0 ? hits_a / n_a : 0.0
    
    if precision + recall == 0
        return 0.0
    end
    return 2 * (precision * recall) / (precision + recall)
end

function tanimoto_difference(scorer::PatternScorer, pattern::Mol)
    details = Dict{String, Any}("radius" => 2, "nBits" => 2048)
    
    try
        pattern_fp = get_morgan_fp_as_bytes(pattern, details)
    catch
        return 0.0
    end
    
    if isempty(scorer.actives_fps)
        return 0.0
    end
    
    avg_tan = 0.0
    for fp in scorer.actives_fps
        avg_tan += tanimoto_similarity(pattern_fp, fp)
    end
    avg_tan /= length(scorer.actives_fps)
    
    if !isempty(scorer.inactives_fps)
        avg_tan_i = 0.0
        for fp in scorer.inactives_fps
            avg_tan_i += tanimoto_similarity(pattern_fp, fp)
        end
        avg_tan_i /= length(scorer.inactives_fps)
        return max(0, avg_tan - avg_tan_i)
    end
    
    return avg_tan
end

function tanimoto_similarity(fp1::Vector{UInt8}, fp2::Vector{UInt8})
    if length(fp1) != length(fp2)
        return 0.0
    end
    
    intersection = 0
    union = 0
    
    for i in 1:length(fp1)
        b1 = fp1[i]
        b2 = fp2[i]
        for bit in 0:7
            mask = UInt8(1 << bit)
            has1 = (b1 & mask) != 0
            has2 = (b2 & mask) != 0
            if has1 || has2
                union += 1
            end
            if has1 && has2
                intersection += 1
            end
        end
    end
    
    return union > 0 ? intersection / union : 0.0
end

function normalize_ef(ef::Float64)
    return min(1.0, ef / 20.0)
end

mutable struct GeneticAlgorithm
    dataset::MoleculeDataset
    population_size::Int
    generations::Int
    mutation_rate::Float64
    crossover_rate::Float64
    elitism::Int
    population::Vector{Tuple{String, Float64}}
    best_patterns::Vector{PatternResult}
    generation_stats::Vector{Dict{Symbol, Any}}
end

function GeneticAlgorithm(
    dataset::MoleculeDataset;
    population_size::Int=100,
    generations::Int=50,
    mutation_rate::Float64=0.3,
    crossover_rate::Float64=0.5,
    elitism::Int=5
)
    GeneticAlgorithm(
        dataset,
        population_size,
        generations,
        mutation_rate,
        crossover_rate,
        elitism,
        Tuple{String, Float64}[],
        PatternResult[],
        Dict{Symbol, Any}[]
    )
end

function initialize_population(ga::GeneticAlgorithm, fragments::Vector{String}; random_count::Int=20)
    population = Tuple{String, Float64}[]
    seen = Set{String}()
    
    scorer = PatternScorer(ga.dataset)
    
    for frag in fragments[1:min(length(fragments), ga.population_size - random_count)]
        if !(frag in seen)
            push!(seen, frag)
            fitness, _, _, _, _, _, _ = score(scorer, frag)
            push!(population, (frag, fitness))
        end
    end
    
    random_smarts = [
        "C", "CC", "CCC", "CCCC", "c1ccccc1", "c1ccc(O)cc1", "c1ccc(N)cc1",
        "c1ccc(F)cc1", "C(=O)O", "C(=O)N", "CN", "CO", "CS", "C#N",
        "C(=O)C", "CC(=O)C", "c1cccnc1", "c1ccncc1", "c1cnccn1", "n1cccc1",
        "C1CCCCC1", "C1CCC1", "c1ccc2ccccc2c1", "C1CCOCC1"
    ]
    
    for smi in random_smarts[1:min(length(random_smarts), random_count)]
        try
            mol = get_mol(smi)
            if mol !== nothing
                smarts = get_smarts(mol)
                if smarts !== nothing && !(smarts in seen)
                    push!(seen, smarts)
                    fitness, _, _, _, _, _, _ = score(scorer, smarts)
                    push!(population, (smarts, fitness))
                end
            end
        catch
        end
    end
    
    mutator = SMARTSMutator()
    while length(population) < ga.population_size
        new_smarts = add_atom(mutator, rand(collect(seen)))
        if !(new_smarts in seen)
            push!(seen, new_smarts)
            fitness, _, _, _, _, _, _ = score(scorer, new_smarts)
            push!(population, (new_smarts, fitness))
        end
    end
    
    ga.population = population[1:min(length(population), ga.population_size)]
end

function select_parent(ga::GeneticAlgorithm)
    tournament_size = 3
    pop_indices = rand(1:length(ga.population), min(tournament_size, length(ga.population)))
    tournament = [ga.population[i] for i in pop_indices]
    return maximum(tournament, by=x->x[2])[1]
end

function evolve!(ga::GeneticAlgorithm)
    println("Starting GA with population $(ga.population_size), generations $(ga.generations)")
    
    frag_gen = FragmentGenerator(ga.dataset)
    fragments = generate_fragments(frag_gen, max_fragments=150)
    initialize_population(ga, fragments)
    
    scorer = PatternScorer(ga.dataset)
    mutator = SMARTSMutator()
    crossover = SMARTSCrossover()
    
    for gen in 1:ga.generations
        new_population = Tuple{String, Float64}[]
        
        sorted_pop = sort(ga.population, by=x->x[2], rev=true)
        elite = sorted_pop[1:min(ga.elitism, length(sorted_pop))]
        append!(new_population, elite)
        
        while length(new_population) < ga.population_size
            if rand() < ga.crossover_rate
                p1 = select_parent(ga)
                p2 = select_parent(ga)
                child = crossover(crossover, p1, p2)
            else
                child = select_parent(ga)
            end
            
            if rand() < ga.mutation_rate
                child = mutate(mutator, child)
            end
            
            fitness, ef, mcc, f1, tan_diff, supp_a, supp_i = score(scorer, child)
            push!(new_population, (child, fitness))
        end
        
        ga.population = new_population[1:min(length(new_population), ga.population_size)]
        
        best = maximum(ga.population, by=x->x[2])
        avg_fitness = mean([p[2] for p in ga.population])
        
        push!(ga.generation_stats, Dict(
            :generation => gen,
            :best_fitness => best[2],
            :avg_fitness => avg_fitness
        ))
        
        println("Gen $gen: Best fitness = $(round(best[2], digits=4)), Avg = $(round(avg_fitness, digits=4)), Best pattern: $(best[1])")
        
        for (smarts, fitness) in elite[1:min(3, length(elite))]
            if !any(p.smarts == smarts for p in ga.best_patterns)
                _, ef, mcc, f1, tan_diff, supp_a, supp_i = score(scorer, smarts)
                push!(ga.best_patterns, PatternResult(
                    smarts, nothing, fitness, ef, mcc, f1, tan_diff, supp_a, supp_i, gen
                ))
            end
        end
    end
    
    sort!(ga.best_patterns, by=x->x.fitness, rev=true)
end

function get_results(ga::GeneticAlgorithm; top_n::Int=50)
    return ga.best_patterns[1:min(top_n, length(ga.best_patterns))]
end

function save_results_csv(results::Vector{PatternResult}, output_file::String, stats::Vector{Dict{Symbol, Any}})
    open(output_file, "w") do f
        write(f, "# SMARTS Pattern Learning Results\n")
        write(f, "# Generated by smarts_learner.jl\n")
        write(f, "#\n")
        
        if !isempty(stats)
            write(f, "# Generation Statistics\n")
            write(f, "# Generation,Best_Fitness,Avg_Fitness\n")
            for s in stats
                write(f, "# $(s[:generation]),$(round(s[:best_fitness], digits=4)),$(round(s[:avg_fitness], digits=4))\n")
            end
            write(f, "#\n")
        end
        
        write(f, "SMARTS,Fitness,Enrichment_Factor,MCC,F1_Score,Tanimoto_Diff,Support_Actives,Support_Inactives,Generation\n")
        
        for r in results
            write(f, "$(r.smarts),$(round(r.fitness, digits=4)),$(round(r.ef, digits=4)),")
            write(f, "$(round(r.mcc, digits=4)),$(round(r.f1, digits=4)),")
            write(f, "$(round(r.tanimoto_diff, digits=4)),$(r.support_actives),")
            write(f, "$(r.support_inactives),$(r.generation)\n")
        end
    end
    
    println("Results saved to $output_file")
end

function parse_arguments()
    s = ArgParseSettings(
        description="Learn discriminative SMARTS patterns using Genetic Algorithm",
        epilog="""
Examples:
  julia smarts_learner.jl --actives actives.smi --inactives inactives.smi
  julia smarts_learner.jl -a actives.smi -i inactives.smi -o patterns.csv -p 150 -g 100
        """
    )
    
    @add_arg_table! s begin
        "--actives", "-a"
            arg_type = String
            required = true
            help = "File containing active molecules (one SMILES per line)"
        "--inactives", "-i"
            arg_type = String
            required = true
            help = "File containing inactive molecules (one SMILES per line)"
        "--output", "-o"
            arg_type = String
            default = "ranked_patterns.csv"
            help = "Output CSV file (default: ranked_patterns.csv)"
        "--population", "-p"
            arg_type = Int
            default = 100
            help = "Population size (default: 100)"
        "--generations", "-g"
            arg_type = Int
            default = 50
            help = "Number of generations (default: 50)"
        "--mutation-rate", "-m"
            arg_type = Float64
            default = 0.3
            help = "Mutation rate (default: 0.3)"
        "--crossover-rate", "-c"
            arg_type = Float64
            default = 0.5
            help = "Crossover rate (default: 0.5)"
        "--top-n", "-n"
            arg_type = Int
            default = 50
            help = "Number of top patterns to save (default: 50)"
    end
    
    return parse_args(s)
end

function main()
    disable_logging()
    
    args = parse_arguments()
    
    println("=" ^ 60)
    println("SMARTS Pattern Learner - Genetic Algorithm")
    println("=" ^ 60)
    
    start_time = time()
    
    dataset = MoleculeDataset(args["actives"], args["inactives"])
    
    ga = GeneticAlgorithm(
        dataset=dataset,
        population_size=args["population"],
        generations=args["generations"],
        mutation_rate=args["mutation-rate"],
        crossover_rate=args["crossover-rate"]
    )
    
    evolve!(ga)
    
    results = get_results(ga, top_n=args["top-n"])
    save_results_csv(results, args["output"], ga.generation_stats)
    
    elapsed = time() - start_time
    println("\nTotal time: $(round(elapsed, digits=1)) seconds")
    println("Top 5 patterns:")
    for (i, r) in enumerate(results[1:min(5, length(results))])
        println("  $i. $(r.smarts) (fitness: $(round(r.fitness, digits=4)))")
    end
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    SMARTSLearner.main()
end
