"""
A Tropical version of Dijkstra's shorted path algorithm
"""
module TropicalDijkstra

using LinearAlgebra



export BoolVector
struct BoolVector <: AbstractArray{Bool, 1}
    elts::Vector{UInt64}
    size::Int
    BoolVector(::UndefInitializer, size::Int) =
        new(Vector{UInt64}(undef, cld(size, 64)), size)
end

# Base.length(xs::BoolVector) = xs.size
Base.size(xs::BoolVector) = (xs.size, )

Base.getindex(xs::BoolVector, i::Int) = getindex(xs, i % UInt)
@inline function Base.getindex(xs::BoolVector, i::UInt)
    @inbounds (xs.elts[i ÷ 0x40] >> (i % 0x40)) != 0
end

Base.setindex!(xs::BoolVector, x::Bool, i::Int) = setindex!(xs, x, i % UInt)
@inline function Base.setindex!(xs::BoolVector, x::Bool, i::UInt)
    msk = ~ (UInt64(0x1) << (i % 0x40))
    bit = UInt64(x) << (i % 0x40)
    @inbounds xs.elts[i ÷ 0x40] = xs.elts[i ÷ 0x40] & msk | bit
end

function Base.fill!(rs::BoolVector, x::Bool)
    fill!(rs.elts, x ? ~ UInt64(0) : UInt64(0))
    rs
end

function Base.copyto!(rs::BoolVector, xs::BoolVector)
    n = length(rs)
    @assert length(xs) == n
    copyto!(rs.elts, xs.elts)
    rs
end

function Base.map!(f::Function, rs::BoolVector, xs::BoolVector)
    n = length(rs)
    @assert length(xs) >= n
    @inbounds @simd for i in 1:n
        rs[i] = f(xs[i])
    end
    rs
end
function Base.map!(::typeof(identity), rs::BoolVector, xs::BoolVector)
    copyto!(rs, xs)
end
function Base.map!(::typeof(~), rs::BoolVector, xs::BoolVector)
    n = length(rs)
    @assert length(xs) == n
    @inbounds @simd for i in eachindex(rs.elts)
        rs.elts[i] = ~ xs.elts[i]
    end
    rs
end
function Base.map!(f::Function, rs::BoolVector, xs::BoolVector, ys::BoolVector)
    n = length(rs)
    @assert length(xs) >= n
    @assert length(ys) >= n
    @inbounds @simd for i in 1:n
        rs[i] = f(xs[i], ys[i])
    end
    rs
end
function Base.map!(::typeof(&), rs::BoolVector, xs::BoolVector, ys::BoolVector)
    n = length(rs)
    @assert length(xs) >= n
    @assert length(ys) >= n
    @inbounds @simd for i in eachindex(rs.elts)
        rs.elts[i] = xs.elts[i] & ys.elts[i]
    end
    rs
end
function Base.map!(::typeof(|), rs::BoolVector, xs::BoolVector, ys::BoolVector)
    n = length(rs)
    @assert length(xs) >= n
    @assert length(ys) >= n
    @inbounds @simd for i in eachindex(rs.elts)
        rs.elts[i] = xs.elts[i] | ys.elts[i]
    end
    rs
end
function Base.map!(::typeof(xor),
                   rs::BoolVector, xs::BoolVector, ys::BoolVector)
    n = length(rs)
    @assert length(xs) >= n
    @assert length(ys) >= n
    @inbounds @simd for i in eachindex(rs.elts)
        rs.elts[i] = xor(xs.elts[i], ys.elts[i])
    end
    rs
end

function Base.mapreduce(f::Function, op::Function, xs::BoolVector; init)
    n = length(xs)
    r = init
    @inbounds @simd for i in 1:n
        r = f(r, op(xs.elts[i]))
    end
    r
end
function Base.mapreduce(f::Function, op, xs::BoolVector, ys::BoolVector; init)
    n = length(xs)
    @assert length(ys) >= n
    r = init
    @inbounds @simd for i in 1:n
        r = f(r, op(xs.elts[i], ys.elts[i]))
    end
    r
end



export matmul_or_and
function matmul_or_and(f::Function,
                       A::AbstractArray{Bool, 2},
                       x::AbstractArray{Bool, 1})
    n, m = size(A)
    @assert size(x, 1) == n
    r = similar(x, m)
    @inbounds for i in 1:m
        s = false
        @simd for j in 1:n
            s = s | f(j, i, A[j,i]) & x[j]
        end
        r[i] = s
    end
    r
end

export matmul_plus_times
function matmul_plus_times(f::Function,
                           A::AbstractArray{<:Real, 2},
                           x::AbstractArray{<:Real, 1})
    n, m = size(A)
    @assert size(x, 1) == n
    R = promote_type(eltype(A), eltype(x))
    r = similar(x, R, m)
    @inbounds for i in 1:m
        s = R(0)
        @simd for j in 1:n
            s = s + f(j, i, A[j,i]) * x[j]
        end
        r[i] = s
    end
    r
end

export matmul_min_plus
@inline function matmul_min_plus(f::Function,
                         A::AbstractArray{<:Real, 2},
                         x::AbstractArray{<:Real, 1})
    n, m = size(A)
    @assert size(x, 1) == n
    R = promote_type(eltype(A), eltype(x))
    r = similar(x, R, m)
    @inbounds for i in 1:m
        s = R(Inf)
        @simd for j in 1:n
            s = min(s, f(j, i, A[j,i]) + x[j])
        end
        r[i] = s
    end
    r
end



export find_connected_component

function find_connected_component(f::Function,
                                  G::AbstractArray{Bool, 2},
                                  i::Int = 1)
    n = LinearAlgebra.checksquare(G)
    x = falses(n)
    if n == 0
        return x
    end
    @assert 1 <= i <= n
    x[i] = true
    while true
        xold = x
        x = xold .| matmul_or_and(f, G, xold)
        isequal(x, xold) && break # reached transitive closure
    end
    x
end

function find_connected_component(f::Function,
                                  G::AbstractArray{<:Real, 2},
                                  i::Int = 1)
    n = LinearAlgebra.checksquare(G)
    R = eltype(G)
    x = fill(R(Inf), n)
    if n == 0
        return x
    end
    @assert 1 <= i <= n
    x[i] = R(0)
    while true
        xold = x
        x = min.(xold, matmul_min_plus(f, G, xold))
        isequal(x, xold) && break # reached transitive closure
    end
    x
end



"""
find_shortest_paths
    G: Graph adjacency matrix
    src, dst: Source and destination vertices
"""
function find_shortest_paths(G::AbstractArray{Bool, 2},
                             src::AbstractArray{Bool, 1},
                             dst::AbstractArray{Bool, 1})
    n = LinearAlgebra.checksquare(G)
    @assert size(src, 1) == size(dst, 1) == n
end

export find_shortest_paths

function find_shortest_paths(f::Function,
                             G::AbstractArray{<:Real, 2},
                             srci::Int,
                             dsti::Int)
    n = LinearAlgebra.checksquare(G)
    @assert 1 <= srci <= n
    @assert 1 <= dsti <= n

    R = typeof(f(1, 1, eltype(G)(0)))
    dists = fill(R(Inf), n)
    # Source is reachable with distance 0
    dists[srci] = R(0)

    dsti == srci && return dists

    # Interior of visited domain, i.e. vertices where all neighbours
    # have been visited
    interior = falses(n)
    visited = dists .< R(Inf)
    boundary = visited .> interior

    while true
        olddists = dists
        oldinterior = interior
        oldvisited = visited

        dists = min.(dists, matmul_min_plus(f, G, olddists))
        visited = dists .< R(Inf)

        interior = oldvisited
        boundary = visited .> interior

        # The termination condition is:
        # - destination vertex is in the interior
        # - the minimum of all boundary distances is at least the
        #   destination distance
        !any(boundary) && break
        if interior[dsti]
            dst_dist = dists[dsti]
            bnd_dist = minimum(dists[i] for i in 1:n if boundary[i])
            bnd_dist >= dst_dist && break
        end

        @assert !isequal(dists, olddists) # reached transitive closure
    end
    dists
end

function find_shortest_paths(f::Function,
                             G::AbstractArray{<:Real, 2},
                             srci::Int,
                             dst::AbstractArray{Bool, 1})
    n = LinearAlgebra.checksquare(G)
    @assert 1 <= srci <= n

    R = typeof(f(1, 1, eltype(G)(0)))
    dists = fill(R(Inf), n)
    # Source is reachable with distance 0
    dists[srci] = R(0)

    !any(dst) && return dists
    dst[srci] && count(dst) == 1 && return dists

    # Interior of visited domain, i.e. vertices where all neighbours
    # have been visited
    interior = falses(n)
    visited = dists .< R(Inf)
    boundary = visited .> interior

    iter = 0
    while true
        iter += 1
        println("Iteration: $iter")
        println("    interior points: $(count(interior))")
        println("    boundary points: $(count(boundary))")
        println("    visited destinations: $(all(dst .<= visited))")
        println("    destinations are interior: $(all(dst .<= interior))")

        olddists = dists
        oldinterior = interior
        oldvisited = visited

        dists = min.(dists, matmul_min_plus(f, G, olddists))
        visited = dists .< R(Inf)

        interior = oldvisited
        boundary = visited .> interior

        # The termination condition is:
        # - destination vertex is in the interior
        # - the minimum of all boundary distances is at least the
        #   destination distance
        !any(boundary) && break
        if all(dst .<= interior)
            dst_dist = maximum(dists[i] for i in 1:n if dst[i])
            bnd_dist = minimum(dists[i] for i in 1:n if boundary[i])
            bnd_dist >= dst_dist && break
        end

        @assert !isequal(dists, olddists) # reached transitive closure
    end
    dists
end



identity3(i,j,x) = identity(x)
# inv3(i,j,x) = inv(x)
inv3(i,j,x) = x ? 1.0 : Inf

function main()
    println("setup G")
    n = 2^17
    # G = rand(n, n) .< 3.0/n
    G = falses(n, n)
    @inbounds for j in 1:n
        j % 1024 == 1 && println("  $j/$n")
        for i in 1:n
            G[i,j] = rand() < 0.1
        end
    end

    println("simply-connect G")
    ps = collect(1:n)
    for i in 1:n-1
        j = rand(i+1:n)
        ps[i],ps[j] = ps[j],ps[i]
    end
    for i in 1:n-1
        G[ps[i],ps[i+1]] = true
    end
    G[ps[n],ps[1]] = true
    di = 512                    # one cache line
    dj = 256                    # to fit into L1 data cache size
    @inbounds for j0 in 1:dj:n, i0 in 1:di:n
        i0 == 1 && j0 % 1024 == 1 && println("  $j0/$n")
        for j in j0:min(j0+dj-1,n)
            @simd for i in i0:min(i0+di-1,n)
                G[i,j] |= G[j,i]
            end
        end
    end

    # println("check simply-connect G")
    # x = find_connected_component(identity3, G)
    # @assert all(x)

    println("find paths G")
    i = rand(1:n)
    js = rand(n) .< 10.0/n
    x = find_shortest_paths(inv3, G, i, js)
    # @show x
    # @show i x[i]
    println(x[js])

    println("Done.")
end

end
