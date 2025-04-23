module Genetics

    using StatsBase
    using Random
    using Plots
    using LaTeXStrings
    using LinearAlgebra
    using Dates

    minimumby(f, iter) = reduce(iter) do x, y
        f(x) < f(y) ? x : y
    end
    minimum_idx_by(f, iter) = reduce(1:length(iter)) do x, y
        f(iter[x]) < f(iter[y]) ? x : y
    end

    function euclid_distance(x, y)
        v = (Float32.(x) - Float32.(y))
        return sqrt(dot(v,v))
    end
    function boolean_distance(x, y)
        return sum(target .!= x)
    end
    function hamming_distance(x, y)
        return sum(count_ones.(xor.(x,y)))
    end

    @enum LossFunctionType EuclidNorm BooleanNorm HammingNorm

    struct genetics_function
        dist
        loss
        DIMS
        genetics_function(dist, loss; DIMS=2) = new(dist, loss, DIMS)
    end

    function create_target_function(target, type = EuclidNorm)
        DIMS = length(target)
        if type == EuclidNorm
            return genetics_function(euclid_distance, x -> euclid_distance(target, x), DIMS=DIMS)
        elseif type == BooleanNorm
            return genetics_function(hamming_distance, x -> hamming_distance(target, x), DIMS=DIMS)
        else
            return genetics_function(boolean_distance, x -> boolean_distance(target, x), DIMS=DIMS)
        end        
    end

    @enum ParentSelectionType Basic Inbreading Outbreading

    struct parent_selection{T}
        func
        population
    end;

    function create_parent_selector(func::genetics_function, selection_type::ParentSelectionType)
        function __selector(population)
            return parent_selection{selection_type}(func, population)
        end
        return __selector
    end

    function create_basic_parent_selector()
        function __selector(population)
            return parent_selection{Basic}(x->x, population)
        end
        return __selector
    end

    function Base.iterate(c::parent_selection{Basic}, k=1)
        while k < length(c.population) >> 1
            first_parent_idx = 2*k-1
            second_parent_idx = -1
            k += 1
            while second_parent_idx == -1 || second_parent_idx == first_parent_idx
                second_parent_idx = rand(1:length(c.population))
            end
            return first_parent_idx => second_parent_idx, k
        end
        return nothing, k
    end

    function Base.iterate(c::parent_selection{Inbreading}, k=1)
        while k < length(c.population) >> 1
            first_parent_idx = rand(1:length(c.population))
            second_parent_idx = minimum_idx_by(x -> c.func.dist(c.population[first_parent_idx], x), c.population)
            k += 1
            return first_parent_idx => second_parent_idx, k
        end
        return nothing
    end

    function Base.iterate(c::parent_selection{Outbreading}, k=1)
        while k < length(c.population) >> 1
            first_parent_idx = rand(1:length(c.population))
            second_parent_idx = minimum_idx_by(x -> -c.func.dist(c.population[first_parent_idx], x), c.population)
            k += 1
            return first_parent_idx => second_parent_idx, k
        end
        return nothing, nothing
    end

    function select_parent_pair(c::parent_selection{Basic})
        first_parent_idx = rand(1:length(c.population))
        second_parent_idx = -1
        while second_parent_idx == -1 || second_parent_idx == first_parent_idx
            second_parent_idx = rand(1:length(c.population))
        end
        return first_parent_idx, second_parent_idx
    end

    function select_parent_pair(c::parent_selection{Inbreading})
        first_parent_idx = rand(1:length(c.population))
        second_parent_idx = minimum_idx_by(x -> c.func.dist(c.population[first_parent_idx], x), c.population)
        return first_parent_idx, second_parent_idx
    end

    function select_parent_pair(c::parent_selection{Outbreading})
        first_parent_idx = rand(1:length(c.population))
        second_parent_idx = minimum_idx_by(x -> -c.func.dist(c.population[first_parent_idx], x), c.population)
        return first_parent_idx, second_parent_idx
    end

    Base.length(s::parent_selection{T}) where T = (length(s.population) >> 1 - 1)
    Base.firstindex(s::parent_selection{T}) where T = firstindex(s.population)
    Base.getindex(s::parent_selection{T}, idx) where T = Base.iterate(s, idx)

    @enum CrossType MultiDimOnePoint OnePoint MultiPoint Linear

    struct crossover_strategy{T}
        bounds::Tuple{<:Number,<:Number}
        parent_select
        domain::Float32
        cross_points::UInt32
        function crossover_strategy{T}(bounds, parent_select; domain=1.0, cross_points=1) where T
            new(bounds, parent_select, domain, cross_points)
        end
    end;

    function create_crossover_strategy(func::genetics_function, bounds; crossover_type::CrossType=MultiDimOnePoint, selection_type::ParentSelectionType=Basic, domain=1, cross_points=1)
        if crossover_type == OnePoint
            return crossover_strategy{crossover_type}(bounds, create_parent_selector(func, selection_type))
        end    
        return crossover_strategy{crossover_type}(bounds, create_parent_selector(func, selection_type), domain=domain, cross_points=cross_points)
    end

    function crossover!(generation, s::crossover_strategy{MultiDimOnePoint})
        DIMS = length(generation[1])
        MAX_SHIFT = Base.top_set_bit(max(s.bounds[1], s.bounds[2])) - 1
        @Threads.threads for gen_idx in (length(generation) >> 1)+1:2:length(generation)
            parent_1, parent_2 = select_parent_pair(s.parent_select(generation[1:length(generation) >> 1]))
            for idx in 1:DIMS
                mask_right = (1 << rand(1:MAX_SHIFT)) - 1
                mask_left = ~mask_right
                generation[gen_idx][idx] = clamp((generation[parent_1][idx] & mask_right) | (generation[parent_2][idx] & mask_left), s.bounds[1], s.bounds[2])
                generation[gen_idx+1][idx] = clamp((generation[parent_1][idx] & mask_left) | (generation[parent_2][idx] & mask_right), s.bounds[1], s.bounds[2])
            end
        end
    end

    function crossover!(generation, s::crossover_strategy{MultiPoint})
        DIMS = length(generation[1])
        TYPE_BIT_SIZE = sizeof(generation[1][1]) << 3
        MAX_SHIFT = DIMS * TYPE_BIT_SIZE
        __shifts = sort(randperm(MAX_SHIFT)[1:s.cross_points])
        @Threads.threads for gen_idx in (length(generation) >> 1)+1:2:length(generation)
            first_parent, second_parent = select_parent_pair(s.parent_select(generation[1:length(generation) >> 1]))
            for shift in __shifts
                shift_dim_idx, shift_bit_index = ceil(Int, shift / TYPE_BIT_SIZE), shift % TYPE_BIT_SIZE
                mask_left = [idx < shift_dim_idx ? typemax(typeof(generation[1][1])) : 0 for idx in 1:DIMS]
                if(shift_bit_index != 0)
                    mask_left[shift_dim_idx] = (1 << shift_bit_index) - 1
                end
                mask_right = .~mask_left
                for dim_idx in 1:DIMS
                    generation[gen_idx][dim_idx] = (generation[first_parent][dim_idx] & mask_left[dim_idx]) | (generation[second_parent][dim_idx] & mask_right[dim_idx])
                    generation[gen_idx+1][dim_idx] = (generation[second_parent][dim_idx] & mask_left[dim_idx]) | (generation[first_parent][dim_idx] & mask_right[dim_idx])
                end            
                first_parent, second_parent = gen_idx, gen_idx+1
            end
        end
    end

    function crossover!(generation, s::crossover_strategy{Linear})
        @Threads.threads for gen_idx in (length(generation) >> 1) + 1 : length(generation)
            parent_1, parent_2 = select_parent_pair(s.parent_select(generation[1:length(generation) >> 1]))
            alpha = rand(-s.domain:0.01:1+s.domain)
            generation[gen_idx] = generation[parent_1] + alpha .* (generation[parent_2] - generation[parent_1])
        end
    end

    function crossover(population, s::crossover_strategy{Linear}; children_size=1)
        result = [population[1] for i in 1:children_size]
        @Threads.threads for gen_idx in 1:children_size
            parent_1, parent_2 = select_parent_pair(s.parent_select(population))
            alpha = rand(-s.domain:0.01:1+s.domain)
            result[gen_idx] = population[parent_1] + alpha .* (population[parent_2] - population[parent_1])
        end
        return result
    end

    @enum MutationType MultiDimOneBit MultiDim OneDim

    function mutate!(generation, mutation_type::MutationType; bounds=nothing, mut_prob=0.3)
        DIMS = length(generation[1])
        TYPE_BIT_SIZE = length(generation[1][1]) << 3
        __bounds = (1, typemax(typeof(generation[1][1])))
        if !isnothing(bounds)
            __bounds = bounds
        end
        if mutation_type == MultiDimOnePoint
            Threads.@threads for i in (length(generation) >> 1) + 1:length(generation)
                for j in 1:DIMS
                    if rand() < mut_prob
                        generation[i][j] = clamp(xor(population[i][j], 1 << rand(0:TYPE_BIT_SIZE-1)), __bounds...)
                    end
                end
            end
        elseif mutation_type == MultiDim
            Threads.@threads for i in (length(generation) >> 1) + 1:length(generation)
                for j in 1:DIMS
                    if rand() < mut_prob
                        generation[i][j] = rand(__bounds[1]:__bounds[2])
                    end
                end
            end
        else
            Threads.@threads for i in (length(generation) >> 1) + 1:length(generation)
                if rand() < mut_prob
                    generation[i][rand(1:DIMS)] = rand(__bounds[1]:__bounds[2])
                end
            end
        end
    end

    @enum SelectionType BestHalf Tournament Roullete

    function select!(generation, func::genetics_function, selection_type::SelectionType)
        if selection_type == BestHalf
            sort!(generation, by=func.loss)
        elseif selection_type == Roullete
            generation_loss = func.loss.(generation)
            generation_loss_sum = sum(generation_loss)
            for idx in eachindex(generation_loss)
                generation_loss[idx] = generation_loss_sum - generation_loss[idx]
            end
            generation_loss_sum = sum(generation_loss)
            weights = [loss/generation_loss_sum for loss in generation_loss]
            result = sample(generation, Weights(weights), length(generation) >> 1)
            for gen_idx in 1:length(generation)>>1
                generation[gen_idx] = result[gen_idx]
            end
        else
            old_gen = copy(generation)
            for gen_idx in 1:length(generation)>>1
                subset = sample(old_gen, length(generation) >> 1, replace=false)
                generation[gen_idx] = minimum_by(func.loss, subset)
            end
        end
    end

    function best_gene(generation, func::genetics_function)
        result = nothing
        best_val = nothing
        for idx in eachindex(generation)
            gene_val = func.loss(generation[idx])
            if isnothing(best_val) || best_val > gene_val
                best_val = gene_val
                result = generation[idx]
            end
        end
        return result, best_val
    end

    struct genetics_strategy
        parent_selection_type::ParentSelectionType
        crossover_type::CrossType
        mutation_type::MutationType
        selection_type::SelectionType
    end

    function genetics_min(func::genetics_function, bounds::Tuple{<:Unsigned, <:Unsigned}, gen_strategy::genetics_strategy;
                            population_size=10, max_iter=1000, mut_prob=0.3, eps=1e-3, values_type=UInt,
                            domain = 1, cross_points=2,
                            log_every=1, verbose_every=0, slow_mutation_every=300)
        generation = [
            (i <= population_size ?
                [values_type(rand(bounds[1]:bounds[2])) for j in 1:func.DIMS] :
                values_type.(zeros(func.DIMS)))
            for i in 1:population_size<<1
        ]
        cross = create_crossover_strategy(func, bounds,
            crossover_type=gen_strategy.crossover_type, selection_type=gen_strategy.parent_selection_type, domain=domain, cross_points=cross_points)
        iterations = 0
        global_min = generation[1]
        global_min_val = 0
        history = [global_min]
        start = now()
        for idx in 1:max_iter
            iterations += 1
            crossover!(generation, cross)
            mutate!(generation, gen_strategy.mutation_type, bounds=bounds, mut_prob=mut_prob)
            select!(generation, func, gen_strategy.selection_type)
            global_min, global_min_val = best_gene(generation, func)
            if slow_mutation_every > 0 && iterations % slow_mutation_every == 0
                mut_prob /= 2
            end
            if log_every > 0 && iterations % log_every == 0
                push!(history, copy(global_min))
            end
            if verbose_every > 0 && iterations % verbose_every == 0
                current = now()
                println("iteration: $(iterations), best_score: $(global_min_val), elapsed_time: $(current-start)")
                flush(stdout)
                start = current
            end
            if abs(global_min_val) < eps
                break
            end
        end
        return global_min, iterations, history
    end

    function generate_initial_population(population_size, bounds::Tuple{<:Unsigned, <:Unsigned}; DIMS=2, values_type=Int)
        return [
            [values_type(rand(bounds[1]:bounds[2])) for j in 1:DIMS]
            for i in 1:population_size
        ]
    end

    function get_default_gen_strategy()
        return genetics_strategy(Basic, MultiDimOnePoint, MultiDimOneBit, BestHalf)
    end

    function ws_genetics_min(func::genetics_function, bounds::Tuple{<:Unsigned, <:Unsigned}, population, gen_strategy::genetics_strategy;
                            max_iter=1000, mut_prob=0.3, values_type=UInt,
                            domain=1, cross_points=2, verbose_every=0)
        population_size = length(population)
        generation = [
            (i <= population_size ?
                copy(population[i]) :
                values_type.(zeros(func.DIMS)))
            for i in 1:population_size<<1
        ]
        cross = create_crossover_strategy(func, bounds,
            crossover_type=gen_strategy.crossover_type, selection_type=gen_strategy.parent_selection_type, domain=domain, cross_points=cross_points)
        iterations = 0
        global_min = generation[1]
        global_min_val = nothing
        start = now()
        for idx in 1:max_iter
            iterations += 1
            crossover!(generation, cross)
            mutate!(generation, gen_strategy.mutation_type, bounds=bounds, mut_prob=mut_prob)
            select!(generation, func, gen_strategy.selection_type)
            current_min, current_min_val = best_gene(generation, func)
            if isnothing(global_min_val) || current_min_val < global_min_val
                global_min, global_min_val = current_min, current_min_val
            end
            if verbose_every > 0 && iterations % verbose_every == 0
                current = now()
                @info "iteration: $(iterations), best_score: $(global_min_val), elapsed_time: $(current-start)"
                flush(stdout)
                start = current
            end
        end
        return copy(generation[1:population_size]), global_min, iterations
    end
end