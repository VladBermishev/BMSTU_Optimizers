using Plots
target(x::Float64) = (1 - x + x^2) / (1 + x - x^2)

function unimodal(f, a, b, N=1000)
    r = range(a, b, 1000)
    last_deriv = realmin(Float64)
    for idx in range(1, 999, 999)
        deriv = (f(r[idx+1]) - f(r[idx-1]))/(r[idx+1]-r[idx-1])
        if deriv < last_deriv
            return false
        end
        last_deriv = deriv
    end
    return true
end

function min_search(f, a::Float64, b::Float64, EPS = 1e-6)
    g = Any[Any[],Any[],Any[],Any[]]
    while(b-a > EPS)
        delta = min(EPS, (b-a)/2)
        x = (a+b)/2
        f1 = f(x - delta)
        f2 = f(x + delta)
        g[1] = push!(g[1], a)
        g[2] = push!(g[2], b)
        g[3] = push!(g[3], x-delta)
        g[4] = push!(g[4], x+delta)
        println(g[2])
        if (f1 < f2)
            b = x
        else
            a = x
        end
    end
    return (b + a)/2, g
end
println("start")
x = range(-2, 2, length=100)
y1 = target.(x)
res, g = min_search(target, 0., 1.)
plot(x, y1)
plot!(g[1], target.(g[1]), seriestype=:scatter, mc=:blue)
plot!(g[2], target.(g[2]), seriestype=:scatter, mc=:blue)
plot!(g[3], target.(g[3]), seriestype=:scatter, mc=:red)
plot!(g[4], target.(g[4]), seriestype=:scatter, mc=:red)
