using Test

using TropicalDijkstra

identity3(i,j,x) = identity(x)
inv3(i,j,x) = inv(x)

@testset "matmul" begin
    G = BitArray(rand(Bool, 5, 4))
    x = BitArray(rand(Bool, 5))
    r = matmul_or_and(identity3, G, x)
    @test Array(r) == (Array(G' * x) .!= 0)
end

@testset "find_connected_component" begin
    G = BitArray([1 1 1; 1 1 0; 1 0 1])
    x = find_connected_component(identity3, G)
    @test all(x)

    G = BitArray([1 1 0; 1 1 0; 0 0 1])
    x = find_connected_component(identity3, G)
    @test x == BitArray([1, 1, 0])
    x = find_connected_component(identity3, G, 3)
    @test x == BitArray([0, 0, 1])
end

@testset "find_shortest_paths" begin
    G = BitArray([1 0 0 0 0;
                  1 1 0 0 0;
                  0 1 1 0 0;
                  0 0 1 1 0;
                  0 0 0 1 1])
    x = find_shortest_paths(inv3, G, 1, 1)
    @test isequal(x, [0.0, Inf, Inf, Inf, Inf])

    G = BitArray([1 1 0 0 0;
                  0 1 1 0 0;
                  0 0 1 1 0;
                  0 0 0 1 1;
                  0 0 0 0 1])
    x = find_shortest_paths(inv3, G, 1, 3)
    @test isequal(x, [0.0, 1.0, 2.0, 3.0, Inf])

    G = rand(100, 100) .< 0.1
    i, j = rand(1:100, 2)
    x = find_shortest_paths(inv3, G, i, j)
    if all(find_connected_component(identity3, G))
        @test x[j] < Inf
    else
        @test x[j] == Inf
    end
end

@testset "find_shortest_paths" begin
    G = BitArray([1 0 0 0 0;
                  1 1 0 0 0;
                  0 1 1 0 0;
                  0 0 1 1 0;
                  0 0 0 1 1])
    x = find_shortest_paths(inv3, G, 1, BitArray([1, 0, 0, 0, 0]))
    @test isequal(x, [0.0, Inf, Inf, Inf, Inf])

    G = BitArray([1 1 0 0 0;
                  0 1 1 0 0;
                  0 0 1 1 0;
                  0 0 0 1 1;
                  0 0 0 0 1])
    x = find_shortest_paths(inv3, G, 1, BitArray([0, 0, 1, 0, 0]))
    @test isequal(x, [0.0, 1.0, 2.0, 3.0, Inf])

    G = rand(100, 100) .< 0.1
    i = rand(1:100)
    js = rand(100) .< 0.1
    x = find_shortest_paths(inv3, G, i, js)
    if all(find_connected_component(identity3, G))
        @test all(x[js] .< Inf)
    else
        @test any(x[js] .== Inf)
    end
end

@time TropicalDijkstra.main()
