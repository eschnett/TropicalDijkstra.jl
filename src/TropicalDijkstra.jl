"""
A Tropical version of Dijkstra's shorted path algorithm
"""
module TropicalDijkstra

using LinearAlgebra



################################################################################



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
    @inbounds @simd ivdep for i in 1:n
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
    @inbounds @simd ivdep for i in eachindex(rs.elts)
        rs.elts[i] = ~ xs.elts[i]
    end
    rs
end
function Base.map!(f::Function, rs::BoolVector, xs::BoolVector, ys::BoolVector)
    n = length(rs)
    @assert length(xs) >= n
    @assert length(ys) >= n
    @inbounds @simd ivdep for i in 1:n
        rs[i] = f(xs[i], ys[i])
    end
    rs
end
function Base.map!(::typeof(&), rs::BoolVector, xs::BoolVector, ys::BoolVector)
    n = length(rs)
    @assert length(xs) >= n
    @assert length(ys) >= n
    @inbounds @simd ivdep for i in eachindex(rs.elts)
        rs.elts[i] = xs.elts[i] & ys.elts[i]
    end
    rs
end
function Base.map!(::typeof(|), rs::BoolVector, xs::BoolVector, ys::BoolVector)
    n = length(rs)
    @assert length(xs) >= n
    @assert length(ys) >= n
    @inbounds @simd ivdep for i in eachindex(rs.elts)
        rs.elts[i] = xs.elts[i] | ys.elts[i]
    end
    rs
end
function Base.map!(::typeof(xor),
                   rs::BoolVector, xs::BoolVector, ys::BoolVector)
    n = length(rs)
    @assert length(xs) >= n
    @assert length(ys) >= n
    @inbounds @simd ivdep for i in eachindex(rs.elts)
        rs.elts[i] = xor(xs.elts[i], ys.elts[i])
    end
    rs
end

function Base.mapreduce(f::Function, op::Function, xs::BoolVector; init)
    n = length(xs)
    r = init
    @inbounds @simd ivdep for i in 1:n
        r = f(r, op(xs.elts[i]))
    end
    r
end
function Base.mapreduce(f::Function, op, xs::BoolVector, ys::BoolVector; init)
    n = length(xs)
    @assert length(ys) >= n
    r = init
    @inbounds @simd ivdep for i in 1:n
        r = f(r, op(xs.elts[i], ys.elts[i]))
    end
    r
end



################################################################################



export matmul_or_and
function matmul_or_and(f::Function,
                       A::AbstractArray{Bool, 2},
                       x::AbstractArray{Bool, 1})
    m, n = size(A)
    @assert size(x, 1) == n
    r = similar(x, m)

    # @inbounds for i in 1:m
    #     s = false
    #     @simd ivdep for j in 1:n
    #         s = s | f(j, i, A[j,i]) & x[j]
    #     end
    #     r[i] = s
    # end

    @inbounds @simd ivdep for i in 1:m
        r[i] = false
    end
    @inbounds for j in 1:n
        if x[j]
            @simd ivdep for i in 1:m
                r[i] |= f(i, j, A[i,j])
            end
        end
    end

    r
end
function matmul_or_and(f::Function,
                       A::AbstractArray{Bool, 2},
                       xs::AbstractArray{Bool, 2})
    m, n = size(A)
    p = size(xs, 1)
    @assert size(xs, 2) == n
    rs = similar(xs, p, m)

    @inbounds for i in 1:m
        @simd ivdep for k in 1:p
            rs[k,i] = false
        end
    end
    @inbounds for j in 1:n
        any_xj = false
        @simd ivdep for k in 1:p
            any_xj |= xs[k,j]
        end
        if any_xj
            for i in 1:m
                y = f(i, j, A[i,j])
                if y
                    @simd ivdep for k in 1:p
                        rs[k,i] |= xs[k,j]
                    end
                end
            end
        end
    end

    rs
end

export matmul_plus_times
function matmul_plus_times(f::Function,
                           A::AbstractArray{<:Real, 2},
                           x::AbstractArray{<:Real, 1})
    m, n = size(A)
    @assert size(x, 1) == n
    R = promote_type(eltype(A), eltype(x))
    r = similar(x, R, m)

    # @inbounds for i in 1:m
    #     s = R(0)
    #     @simd ivdep for j in 1:n
    #         s = s + f(j, i, A[j,i]) * x[j]
    #     end
    #     r[i] = s
    # end

    @inbounds @simd ivdep for i in 1:m
        r[i] = R(0)
    end
    @inbounds for j in 1:n
        xj = x[j]
        if xj != R(0)
            @simd ivdep for i in 1:m
                r[i] += f(i, j, A[i,j]) * xj
            end
        end
    end

    r
end
function matmul_plus_times(f::Function,
                           A::AbstractArray{<:Real, 2},
                           xs::AbstractArray{<:Real, 2})
    m, n = size(A)
    p = size(xs, 1)
    @assert size(xs, 2) == n
    R = promote_type(eltype(A), eltype(xs))
    rs = similar(xs, R, p, m)

    @inbounds for i in 1:m
        @simd ivdep for k in 1:p
            rs[k,i] = R(0)
        end
    end
    @inbounds for j in 1:n
        any_xj = false
        @simd ivdep for k in 1:p
            any_xj |= xs[k,j] != R(0)
        end
        if any_xj
            for i in 1:m
                y = f(i, j, A[i,j])
                if y != R(0)
                    @simd ivdep for k in 1:p
                        rs[k,i] += y * xs[k,j]
                    end
                end
            end
        end
    end

    rs
end

export matmul_min_plus
function matmul_min_plus(f::Function,
                         A::AbstractArray{<:Real, 2},
                         x::AbstractArray{<:Real, 1})
    m, n = size(A)
    @assert size(x, 1) == n
    R = promote_type(eltype(A), eltype(x))
    r = similar(x, R, m)

    # @inbounds for i in 1:m
    #     s = R(Inf)
    #     @simd ivdep for j in 1:n
    #         s = min(s, f(j, i, A[j,i]) + x[j])
    #     end
    #     r[i] = s
    # end

    @inbounds @simd ivdep for i in 1:m
        r[i] = R(Inf)
    end
    @inbounds for j in 1:n
        xj = x[j]
        if xj != R(Inf)
            @simd ivdep for i in 1:m
                r[i] = min(r[i], f(i, j, A[i,j]) + xj)
            end
        end
    end

    r
end
function matmul_min_plus(f::Function,
                         A::AbstractArray{<:Real, 2},
                         xs::AbstractArray{<:Real, 2})
    m, n = size(A)
    p = size(xs, 1)
    @assert size(xs, 2) == n
    R = promote_type(eltype(A), eltype(xs))
    rs = similar(xs, R, p, m)

    @inbounds for i in 1:m
        @simd ivdep for k in 1:p
            rs[k,i] = R(Inf)
        end
    end
    @inbounds for j in 1:n
        any_xj = false
        @simd ivdep for k in 1:p
            any_xj |= xs[k,j] != R(Inf)
        end
        if any_xj
            for i in 1:m
                y = f(i, j, A[i,j])
                if y != R(Inf)
                    @simd ivdep for k in 1:p
                        rs[k,i] = min(rs[k,i], y + xs[k,j])
                    end
                end
            end
        end
    end

    rs
end



################################################################################



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



################################################################################



export find_shortest_paths

"""
find_shortest_paths
    G: Graph adjacency matrix
    src, dst: Source and destination vertices
"""
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
    @assert size(dst, 1) == n

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
        olddists = dists
        oldinterior = interior
        oldvisited = visited

        dists = min.(dists, matmul_min_plus(f, G, olddists))
        visited = dists .< R(Inf)

        interior = oldvisited
        boundary = visited .> interior

        iter += 1
        println("Iteration: $iter")
        println("    interior points: $(count(interior))")
        println("    boundary points: $(count(boundary))")
        println("    visited destinations: $(all(dst .<= visited))")
        println("    destinations are interior: $(all(dst .<= interior))")

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

function find_shortest_paths(f::Function,
                             G::AbstractArray{<:Real, 2},
                             srci::Int,
                             dsts::Vector{Int})
    n = LinearAlgebra.checksquare(G)
    @assert 1 <= srci <= n
    @assert all(1 <= i <= n for i in dsts)

    R = typeof(f(1, 1, eltype(G)(0)))
    dists = fill(R(Inf), n)
    # Source is reachable with distance 0
    dists[srci] = R(0)

    all(i == srci for i in dsts) && return dists

    # Interior of visited domain, i.e. vertices where all neighbours
    # have been visited
    interior = falses(n)
    visited = dists .< R(Inf)
    boundary = visited .> interior

    iter = 0
    while true
        olddists = dists
        oldinterior = interior
        oldvisited = visited

        dists = min.(dists, matmul_min_plus(f, G, olddists))
        visited = dists .< R(Inf)

        interior = oldvisited
        boundary = visited .> interior

        iter += 1
        println("Iteration: $iter")
        println("    interior points: $(count(interior))")
        println("    boundary points: $(count(boundary))")
        println("    visited destinations: $(all(visited[dsts]))")
        println("    destinations are interior: $(all(interior[dsts]))")

        # The termination condition is:
        # - destination vertex is in the interior
        # - the minimum of all boundary distances is at least the
        #   destination distance
        !any(boundary) && break
        if all(interior[dsts])
            dst_dist = maximum(dists[dsts])
            bnd_dist = minimum(dists[i] for i in 1:n if boundary[i])
            bnd_dist >= dst_dist && break
        end

        @assert !isequal(dists, olddists) # reached transitive closure
    end
    dists
end

function find_shortest_paths(f::Function,
                             G::AbstractArray{<:Real, 2},
                             srcs::Vector{Int},
                             dsts::Vector{Int})
    println("find_shortest_paths: ",
            "$(length(srcs)) sources, $(length(dsts)) destinations")

    n = LinearAlgebra.checksquare(G)
    p = length(srcs)
    @assert all(1 <= i <= n for i in srcs)
    @assert all(1 <= i <= n for i in dsts)

    R = typeof(f(1, 1, eltype(G)(0)))
    dists = fill(R(Inf), p, n)
    # Sources are reachable with distance 0
    for (k,i) in enumerate(srcs)
        dists[k,i] = R(0)
    end

    # Interior of visited domain, i.e. vertices where all neighbours
    # have been visited
    interior = falses(p, n)
    visited = dists .< R(Inf)
    boundary = visited .> interior

    iter = 0
    while true
        olddists = dists
        oldinterior = interior
        oldvisited = visited

        dists = min.(dists, matmul_min_plus(f, G, olddists))
        visited = dists .< R(Inf)

        interior = oldvisited
        boundary = visited .> interior

        iter += 1
        println("Iteration: $iter")
        println("    interior points: $(count(interior))")
        println("    boundary points: $(count(boundary))")
        println("    visited destinations: ",
                "$(count(visited[:,dsts]))/$(length(srcs)*length(dsts))")
        println("    destinations are interior: ",
                "$(count(all(interior[:,dsts], dims=2)))/$(length(srcs))")

        # The termination condition is:
        # - destination vertex is in the interior
        # - the minimum of all boundary distances is at least the
        #   destination distance
        !any(boundary) && break
        if all(interior[:, dsts])
            dst_dist = maximum(dists[:, dsts], dims=2)
            bnd_dist =
                minimum((dists[:, i] for i in 1:n if boundary[i]), dims=2)
            all(bnd_dist >= dst_dist) && break
        end

        @assert !isequal(dists, olddists) # reached transitive closure
    end
    dists
end



################################################################################



# identity3(i,j,x) = identity(x)
identity3(i,j,x) = x
# inv3(i,j,x) = inv(x)
inv3(i,j,x) = x ? 1.0 : Inf

function main()
    println("setup G")
    n = 2^16

    G = BitArray(undef, n, n)
    di = dj = 256               # to fit into L1 data cache size
    @inbounds for j0 in 1:dj:n, i0 in 1:di:j0
        i0 == 1 && j0 % 1024 == 1 && println("  $j0/$n")
        for j in j0:min(j0+dj-1,n), i in i0:min(i0+di-1,j)
            G[i,j] = G[j,i] = rand() < 0.005
        end
    end

    println("simply-connect G")
    ps = collect(1:n)
    for i in 1:n-1
        j = rand(i+1:n)
        ps[i],ps[j] = ps[j],ps[i]
    end
    for m in 1:n
        i, j = ps[m], ps[m == n ? 1 : m + 1]
        G[i,j] = G[j,i] = true
    end

    # println("check simply-connect G")
    # x = find_connected_component(identity3, G)
    # @assert all(x)

    println("find paths G")
    i = rand(1:n)
    # is = [i]
    is = falses(n)
    is[i] = true
    is = is .| matmul_or_and(identity3, G, is)
    is = [i for (i,b) in enumerate(is) if b]
    # js = rand(n) .< 10.0/n
    # Choose a point and its neighbours
    j = rand(1:n)
    # js = [j]
    js = falses(n)
    js[j] = true
    js = js .| matmul_or_and(identity3, G, js)
    js = [j for (j,b) in enumerate(js) if b]
    @time xs = find_shortest_paths(inv3, G, is, js)
    println("Minimum for each source: ", minimum(xs[:,js], dims=2))
    println("Minimum for each destination: ", minimum(xs[:,js], dims=1))
    println("Overall minimum distance: ", minimum(xs[:,js]))

    println("Done.")
end

end
