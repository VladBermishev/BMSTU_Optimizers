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

@enum ParentSelectionType Basic Inbreading Outbreading

struct parent_selection{T}
    dist_func
    population
end;

function create_parent_selector(func, selection_type::ParentSelectionType)
    function __selector(population)
        return parent_selection{selection_type}(func, population)
    end
    return __selector
end

function Base.iterate(c::parent_selection{Basic}, k=1)
    while k <= length(c.population) >> 1
        first_parent_idx = 2*k-1
        second_parent_idx = -1
        k += 1
        while second_parent_idx == -1 || second_parent_idx == first_parent_idx
            second_parent_idx = rand(1:length(c.population))
        end
        return first_parent_idx => second_parent_idx, k
    end
    return nothing
end

function Base.iterate(c::parent_selection{Inbreading}, k=1)
    while k <= length(c.population) >> 1
        first_parent_idx = rand(1:length(c.population))
        second_parent_idx = minimum_idx_by(x -> c.dist_func(c.population[first_parent_idx], x), c.population)
        k += 1
        return first_parent_idx => second_parent_idx, k
    end
    return nothing
end

function Base.iterate(c::parent_selection{Outbreading}, k=1)
    while k <= length(c.population) >> 1
        first_parent_idx = rand(1:length(c.population))
        second_parent_idx = minimum_idx_by(x -> -c.dist_func(c.population[first_parent_idx], x), c.population)
        k += 1
        return first_parent_idx => second_parent_idx, k
    end
    return nothing
end


@enum CrossType MultiDimOnePoint OnePoint MultiPoint Linear

struct crossover_strategy{T}
    bounds::Tuple{<:Unsigned,<:Unsigned}
    generation
    parent_select
    domain::UInt32
    crossover_strategy(bounds, generation, parent_select; domain=1) = new(bounds, generation, parent_select, domain)
end;

function crossover(s::crossover_strategy{MultiDimOnePoint})
    const DIMS = length(s.generation[0])
    const MAX_SHIFT = Base.top_set_bit(max(s.bounds[1], s.bounds[2])) - 1
    gen_idx = (length(s.generation) >> 1) + 1
    Threads.@threads for (parent_1, parent_2) in s.parent_select
        for idx in 1:DIMS
            mask_right = (1 << rand(1:MAX_SHIFT)) - 1
            mask_left = ~mask_right
            s.generation[gen_idx][j] = clamp((population[pop_idx][j] & mask_right) | (population[pair_idx][j] & mask_left), bounds...)
            s.generation[gen_idx+1][j] = clamp((population[pop_idx][j] & mask_left) | (population[pair_idx][j] & mask_right), bounds...)
            gen_idx += 2
        end
    end
end

function crossover(s::crossover_strategy{MultiPoint})
    const DIMS = length(s.generation[0])
    const MAX_SHIFT = Base.top_set_bit(max(s.bounds[1], s.bounds[2])) - 1
    gen_idx = (length(s.generation) >> 1) + 1
    Threads.@threads for (parent_1, parent_2) in s.parent_select
        for idx in 1:DIMS
            mask_right = (1 << rand(1:MAX_SHIFT)) - 1
            mask_left = ~mask_right
            s.generation[gen_idx][j] = clamp((s.generation[parent_1][j] & mask_right) | (s.generation[parent_2][j] & mask_left), bounds...)
            s.generation[gen_idx+1][j] = clamp((s.generation[parent_1][j] & mask_left) | (s.generation[parent_2][j] & mask_right), bounds...)
        end
        gen_idx += 2
    end
end

function crossover(s::crossover_strategy{Linear})
    gen_idx = (length(s.generation) >> 1) + 1
    Threads.@threads for (parent_1, parent_2) in s.parent_select
        alpha = rand(-s.domain:1+s.domain)
        s.generation[gen_idx] = s.generation[parent_1] + alpha .* (s.generation[parent_2] - s.generation[parent_1])
        gen_idx += 1
    end
end