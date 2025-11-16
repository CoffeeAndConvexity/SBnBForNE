using JuMP, JLD2, KNITRO, LinearAlgebra, HDF5, Gurobi

ALG_KNITRO = 1 # 1 for Interior/Direct and 2 for Interior/CG, default is 1

if IMPLEMENTATION_TYPE == :full
    @info "[🐘 ] full implementation activated"
    TIME_LIMIT_KNITRO = 600 # default is 600 (ie 10 min)
elseif IMPLEMENTATION_TYPE == :heuristic
    @info "[🐘 ] heuristic implementation activated"
    TIME_LIMIT_KNITRO = 15 # only 15 s 
else
    @error "IMPLEMENTATION_TYPE not recognized"
end

## Projections onto probability simplex 

## Special iterator functions

function iterators(m, i, k_i)
    ret = [1:m_i for m_i in m]
    ret[i] = k_i:k_i
    return ret
end

function iterators_full(m, i)
    ret = [1:m_i for m_i in m]
    # ret[i] = k_i:k_i
    return ret
end

## Function to find a feasible solution to the NE problem

function NE_computation_bnb_solver_complete_continuous_version(
    # data
    # ====
    A, n, m, util_range, 
    # warm-start points 
    p_ws, v_ws, ϖ_ws
    ; 
    # options
    # ======
    solver = :knitro, # options are :gurobi and :knitro 
    objective_type = :penalized, # options are :penalized and :exact
    FeasibilityTol_KNITRO = 1e-3,
    OptimalityTol_KNITRO = 1e-3,
    MIPOptimalityTolKNITRO = 1e-2
    # good_enough_feasibility_tolerance = 0,
    )

    if solver == :knitro

        @info "[🚀 ] activating KNITRO"

        modelGame = Model(
            optimizer_with_attributes(
                KNITRO.Optimizer,
                "convex" => 0,
                "strat_warm_start" => 1,
                # the last settings below are for larger N
                # you can comment them out if preferred but not recommended
                "honorbnds" => 1,
                # "bar_feasmodetol" => 1e-3,
                "feastol" => FeasibilityTol_KNITRO,
                # "feastol_abs" => FeasibilityTol_KNITRO,
                #"infeastol" => 1e-12,
                "opttol" => OptimalityTol_KNITRO,
                "mip_opt_gap_rel" => MIPOptimalityTolKNITRO,
                "maxtime" => TIME_LIMIT_KNITRO,
                "algorithm" => ALG_KNITRO # 1 for primal-dual interior point method
                # "maxit" => 30000
                # "opttol_abs" => OptimalityTol_KNITRO,
                # "mip_multistart" => 1,
                #"opttolabs" => 1e-1
            )
        )

        elseif solver == :gurobi

            @info "[🐌 ] globally optimal solution finder activated, solution method: spatial branch and bound"

            modelGame = Model(Gurobi.Optimizer)
            # using direct_model results in smaller memory allocation
            # we could also use
            # Model(Gurobi.Optimizer)
            # but this requires more memory

            set_optimizer_attribute(modelGame, "NonConvex", 2)
            # "NonConvex" => 2 tells Gurobi to use its nonconvex algorithm

            set_optimizer_attribute(modelGame, "MIPFocus", 1)
            # If you are more interested in good quality feasible solutions, you can select MIPFocus=1.
            # If you believe the solver is having no trouble finding the optimal solution, and wish to focus more
            # attention on proving optimality, select MIPFocus=2.
            # If the best objective bound is moving very slowly (or not at all), you may want to try MIPFocus=3 to focus on the bound.

            # 🐑: other Gurobi options one can play with
            # ------------------------------------------

            # turn off all the heuristics (good idea if the warm-starting point is near-optimal)
            # set_optimizer_attribute(modelGame, "Heuristics", 0)
            # set_optimizer_attribute(modelGame, "RINS", 0)

            # other termination epsilons for Gurobi
            # set_optimizer_attribute(modelGame, "MIPGapAbs", 1e-4)

            # set_optimizer_attribute(modelGame, "MIPGap", 1e-2) # 99% optimal solution, because Gurobi will provide a result associated with a global lower bound within this tolerance, by polishing the result, we can find the exact optimal solution by solving a convex SDP

            # set_optimizer_attribute(modelGame, "FuncPieceRatio", 0) # setting "FuncPieceRatio" to 0, will ensure that the piecewise linear approximation of the nonconvex constraints lies below the original function

            # set_optimizer_attribute(modelGame, "Threads", 64) # how many threads to use at maximum
            #
            set_optimizer_attribute(modelGame, "FeasibilityTol", FeasibilityTol_KNITRO)
            #
            set_optimizer_attribute(modelGame, "OptimalityTol", OptimalityTol_KNITRO)

            set_optimizer_attribute(modelGame, "TimeLimit", 1200)

    else

        @error "solver not recognized"
        return

    end

    # # Define decision variables
    
    @info "[🐆 ] Defining the decision variables"

    # defining $p$
    # =========

    @variable(modelGame, 0 <= p[i=1:n, k_i=1:m[i]] <= 1)


    # defining $v$
    # =========

    @variable(modelGame, util_range[1] <= v[i=1:n] <= util_range[2])


    # define epigraph variable  $\varpi$
    # =========


    @variable(modelGame, ϖ >= 0)

    # add constraint 
    # $v_{i}-\sum_{k_{1}\in[1:m_{1}]}\cdots\sum_{k_{i-1}\in[1:m_{i-1}]}\sum_{k_{i+1}\in[1:m_{i+1}]}\cdots\sum_{k_{n}\in[1:m_{n}]}\mathcal{A}_{k_{1},k_{2},\ldots,k_{n}}^{[i]}\prod_{\ell\in[1:n]:\ell\neq i}p_{\ell,k_{\ell}}\geq0,\quad\forall i\in[1:n],\forall k_{i}\in M_{i}$


    @constraint(modelGame, con_v[i=1:n, k_i=1:m[i]], 
        v[i] - sum(
            A[i, k...] * prod(p[j,k[j]] for j in 1:n if j != i)
            for k in Iterators.product(iterators(m, i, k_i)...)
        ) >= 0
    )


    ## Add the constraints one by one 

    @info "[🐉 ] Defining the constraints"

    @constraint(modelGame, con_p_sum[i=1:n], sum(p[i,j] for j in 1:m[i]) == 1)

    if objective_type == :penalized

        @info "[🐏 ] penalized objective function activated"


        # Add constraint 
        #  $\varpi\geq p_{i,k_{i}}\left(v_{i}-\sum_{k_{1}\in[1:m_{1}]}\cdots\sum_{k_{i-1}\in[1:m_{i-1}]}\sum_{k_{i+1}\in[1:m_{i+1}]}\cdots\sum_{k_{n}\in[1:m_{n}]}\mathcal{A}_{k_{1},k_{2},\ldots,k_{n}}^{[i]}\prod_{\ell\in[1:n]:\ell\neq i}p_{\ell,k_{\ell}}\right),\quad i\in[1:n],k_{i}\in M_{i}$



        @constraint(modelGame, con_v_penalty_pos[i=1:n, k_i=1:m[i]], 
        ϖ >= p[i,k_i]*(v[i] - sum(
            A[i, k...] * prod(p[j,k[j]] for j in 1:n if j != i)
            for k in Iterators.product(iterators(m, i, k_i)...)))
        )

        # Add constraint 
        # $\varpi\geq-\left(p_{i,k_{i}}\left(v_{i}-\sum_{k_{1}\in[1:m_{1}]}\cdots\sum_{k_{i-1}\in[1:m_{i-1}]}\sum_{k_{i+1}\in[1:m_{i+1}]}\cdots\sum_{k_{n}\in[1:m_{n}]}\mathcal{A}_{k_{1},k_{2},\ldots,k_{n}}^{[i]}\prod_{\ell\in[1:n]:\ell\neq i}p_{\ell,k_{\ell}}\right)\right),\quad i\in[1:n],k_{i}\in M_{i}$


        
        @constraint(modelGame, con_v_penalty_neg[i=1:n, k_i=1:m[i]], 
        ϖ >= -(p[i,k_i]*(v[i] - sum(
            A[i, k...] * prod(p[j,k[j]] for j in 1:n if j != i)
            for k in Iterators.product(iterators(m, i, k_i)...))))
        )


        # @constraint(modelGame, con_good_enough_feasibility, 
        # τ_max <= good_enough_feasibility_tolerance)

        #@constraint(modelGame, con_good_enough_feasibility, sum(τ[i,k_i] for i in 1:n, k_i in 1:m[i]) >= good_enough_feasibility_tolerance)

        @objective(modelGame, Min, ϖ
        )


    else

        @error "objective type not recognized"
        return

    end

    ## Warm-start if the solver is Gurobi 

    if solver == :gurobi

        @info "[🐌 ] warm-starting activated"

        # warm-start $p$

        for i in 1:n
            for k_i in 1:m[i]
                set_start_value(p[i,k_i], p_ws[i,k_i])
            end
        end

        # warm-start $v$

        for i in 1:n
            set_start_value(v[i], v_ws[i])
        end

        # warm-start ϖ

        set_start_value(ϖ, ϖ_ws)

        ## Check if all variables are warm-starteed
        # ==============

        if any(isnothing, start_value.(all_variables(modelGame))) == true
            @error "not all the variables are warm-started"
            return
        else
            @info "[😃 ] all the variables are warm-started"
        end

    end

    ## Time to find a feasible solution 

    optimize!(modelGame)

    sol_time = solve_time(modelGame)

    ## Extract the optimal values of the decision variables

    p_ws = value.(p)

    v_ws = value.(v)

    ϖ_ws = value.(ϖ)
    
    return p_ws, v_ws, ϖ_ws, termination_status(modelGame), sol_time

end

function NE_computation_bnb_solver_complete_continuous_with_complementarity(
    # data
    # ====
    A, n, m, util_range,
    # warm-start points
    # ==============
    p_ws, v_ws, ϖ_ws;
    # options
    # ======
    solver = :knitro, # options are :gurobi and :knitro 
    FeasibilityTol_KNITRO = 1e-3,
    OptimalityTol_KNITRO = 1e-3,
    MIPOptimalityTolKNITRO = 1e-2
    )

    if solver == :knitro

        @info "[🚀 ] activating KNITRO"

        modelGame = Model(
            optimizer_with_attributes(
                KNITRO.Optimizer,
                "convex" => 0,
                "strat_warm_start" => 1,
                # the last settings below are for larger N
                # you can comment them out if preferred but not recommended
                "honorbnds" => 1,
                # "bar_feasmodetol" => 1e-3,
                "feastol" => FeasibilityTol_KNITRO,
                # "feastol_abs" => FeasibilityTol_KNITRO,
                #"infeastol" => 1e-12,
                "opttol" => OptimalityTol_KNITRO,
                "mip_opt_gap_rel" => MIPOptimalityTolKNITRO,
                "maxtime" => TIME_LIMIT_KNITRO,
                "algorithm" => ALG_KNITRO # 1 for primal-dual interior point method, 2 for CG based method
                # "maxit" => 30000
                # "opttol_abs" => OptimalityTol_KNITRO,
                # "mip_multistart" => 1,
                #"opttolabs" => 1e-1
            )
        )
    else

        @error "solver not recognized"
        return

    end

    # # Define decision variables
    
    @info "[🐆 ] Defining the decision variables"

    # defining $p$
    # =========

    @variable(modelGame, 0 <= p[i=1:n, k_i=1:m[i]] <= 1)

    # warm-start $p$

    for i in 1:n
        for k_i in 1:m[i]
            set_start_value(p[i,k_i], p_ws[i,k_i])
        end
    end

    # defining $v$
    # =========

    @variable(modelGame, util_range[1] <= v[i=1:n] <= util_range[2])

    # warm-start $v$

    for i in 1:n
        set_start_value(v[i], v_ws[i])
    end

    ## Check if all variables are warm-starteed
    # ==============

    if any(isnothing, start_value.(all_variables(modelGame))) == true
        @error "not all the variables are warm-started"
        return
    else
        @info "[😃 ] all the variables are warm-started"
    end

    ## Add the constraints one by one 

    @info "[🐉 ] Defining the constraints"

    # add constraint 
    # $v_{i}-\sum_{k_{1}\in[1:m_{1}]}\cdots\sum_{k_{i-1}\in[1:m_{i-1}]}\sum_{k_{i+1}\in[1:m_{i+1}]}\cdots\sum_{k_{n}\in[1:m_{n}]}\mathcal{A}_{k_{1},k_{2},\ldots,k_{n}}^{[i]}\prod_{\ell\in[1:n]:\ell\neq i}p_{\ell,k_{\ell}}\geq0,\quad\forall i\in[1:n],\forall k_{i}\in M_{i}$


    @constraint(modelGame, con_v[i=1:n, k_i=1:m[i]], 
        v[i] - sum(
            A[i, k...] * prod(p[j,k[j]] for j in 1:n if j != i)
            for k in Iterators.product(iterators(m, i, k_i)...)
        ) >= 0
    )

    @constraint(modelGame, con_p_sum[i=1:n], sum(p[i,j] for j in 1:m[i]) == 1)

    ## Add the complementarity constraint
    # $\left(v_{i}-\sum_{k_{1}\in[1:m_{1}]}\cdots\sum_{k_{i-1}\in[1:m_{i-1}]}\sum_{k_{i+1}\in[1:m_{i+1}]}\cdots\sum_{k_{n}\in[1:m_{n}]}\mathcal{A}_{k_{1},k_{2},\ldots,k_{n}}^{[i]}\prod_{\ell\in[1:n]:\ell\neq i}p_{\ell,k_{\ell}}\right)\perp p_{i,k_{i}}=0,\quad i\in[1:n],k_{i}\in M_{i}$


    @constraint(modelGame, con_complementarity[i=1:n, k_i=1:m[i]], 
        (v[i] - sum(
            A[i, k...] * prod(p[j,k[j]] for j in 1:n if j != i)
            for k in Iterators.product(iterators(m, i, k_i)...))) ⟂ p[i,k_i] 
    )

    ## Time to find a feasible solution 

    optimize!(modelGame)

    @info "[📠 ] Termination status for nonlinear interior point method is $(termination_status(modelGame))" #$

    if termination_status(modelGame) != LOCALLY_SOLVED

        @warn "[👿 ] The solver did not find an optimal solution, returning the warm-started values with complementarity constraints gap $(ϖ_ws)" #$

        p_star = p_ws 

        v_star = v_ws

        ϖ_star = ϖ_ws

    else

        @info "[🌋 ] The solver found a Nash Equilibrium!"

        p_star = value.(p)

        v_star = value.(v)

        ϖ_star = 0.0 # since we are not using the penalized objective function, we set ϖ_star to 0.0

    end

    sol_time = solve_time(modelGame)

    return p_star, v_star, ϖ_star, termination_status(modelGame), sol_time

end

## Computing quality of Nash equilibrium given a mixed strategy 

function compute_ϵ_for_mixed_strategy(A, n, m, p_ws)
    # Compute 
    # $V_{i}=\max_{k_{i}\in[1:m_{i}]}\sum_{k_{1}\in[1:m_{1}]}\cdots\sum_{k_{i-1}\in[1:m_{i-1}]}\sum_{k_{i+1}\in[1:m_{i+1}]}\cdots\sum_{k_{n}\in[1:m_{n}]}\mathcal{A}_{k_{1},k_{2},\ldots,k_{n}}^{[i]}\prod_{j\in[1:n]:j\neq i}p_{j,k_{j}}$

    
    V_i_array = []
    
    for i in 1:n


        V_i = maximum(
            [ sum(
                A[i, k...] * prod(p_ws[j,k[j]] for j in 1:n if j != i)
                for k in Iterators.product(iterators(m, i, k_i)...)
            )
             for k_i in 1:m[i]]
        )   
        
        push!(V_i_array, V_i)

    end

    # Compute 
    #  $E_{i}=\sum_{k_{1}\in[1:m_{1}]}\cdots\sum_{k_{i-1}\in[1:m_{i-1}]}\sum_{k_{i}\in[1:m_{i}]}\sum_{k_{i+1}\in[1:m_{i+1}]}\cdots\sum_{k_{n}\in[1:m_{n}]}\mathcal{A}_{k_{1},k_{2},\ldots,k_{n}}^{[i]}\prod_{j\in[1:n]}p_{j,k_{j}}$

    E_i_array = []

    for i in 1:n

        E_i = sum(
            A[i, k...] * prod(p_ws[j,k[j]] for j in 1:n)
            for k in Iterators.product(iterators_full(m, i)...)
        )

        push!(E_i_array, E_i)

    end
 
    # Compute 
    # $\epsilon_{\textup{Nash}}=\max_{i\in[1:n]}V_{i}-E_{i}$

    # Construct V_i - E_i array

    V_i_minus_E_i_array = [V_i_array[i] - E_i_array[i] for i in 1:n]

    ϵ_Nash = maximum(V_i_minus_E_i_array)   

    return ϵ_Nash

end  

## Define the NE solver spatial branch and bound
# ==============

## Define a function to normalize the payoff matrix

# function normalize_payoff_matrix(str_input; 
#     normalize_type = :zero_one # options are :zero_one and :minusOne_One
#     )

#     dset = load_utility_matrices(str_input);

#     dset = permutedims(dset, reverse(1:ndims(dset)));

#     n_players, m_actions = size(dset)[1], size(dset)[2]

#     if normalize_type == :zero_one

#         dset = dset .-  minimum(dset); # to ensure that the minimum is 0

#         dset = dset ./ maximum(dset); # to ensure that the maximum is 1

#     elseif normalize_type == :minusOne_One

#         max_entry_dset = max(minimum(abs.(dset)), maximum(abs.(dset)))

#         dset = dset ./ max_entry_dset; # to ensure that the entries are between -1 and 1

#     else 

#         @error "normalize type not recognized"
#         return

#     end

#     util_range = (minimum(dset), maximum(dset))

#     return n_players, m_actions, dset, util_range

# end

function normalize_payoff_matrix(str_input; 
    normalize_type = :zero_one # options are :zero_one and :minusOne_One
    )

    dset = load_utility_matrices(str_input);

    dset = permutedims(dset, reverse(1:ndims(dset)));

    # n_players, m_actions = size(dset)[1], size(dset)[2]

    s = size(dset)

    n_players = s[1]

    m_actions = collect(s[2:end])

    if normalize_type == :zero_one

        dset = dset .-  minimum(dset); # to ensure that the minimum is 0

        dset = dset ./ maximum(dset); # to ensure that the maximum is 1

    elseif normalize_type == :minusOne_One

        max_entry_dset = max(minimum(abs.(dset)), maximum(abs.(dset)))

        dset = dset ./ max_entry_dset; # to ensure that the entries are between -1 and 1

    else 

        @error "normalize type not recognized"
        return

    end

    util_range = (minimum(dset), maximum(dset))

    return n_players, m_actions, dset, util_range

end

## Funciton to load the utility matrices from the HDF5 file
 
function load_utility_matrices(filename::AbstractString)
    h5open(filename, "r") do file
        return read(file["utilities"])
    end
end


function compute_NE_master_via_complementarity(input_file_name_with_path)

    n_players, m_actions, A, util_range = normalize_payoff_matrix(input_file_name_with_path; normalize_type= :zero_one)  # options are :zero_one and :minusOne_One);

    # n = n_players # data_from_Python["n_players"]
    # m_uniform = m_actions# data_from_Python["m_actions"]
    # m = convert(Array{Int}, m_uniform*ones(n))

    n = n_players # data_from_Python["n_players"]
    # m_uniform = m_actions# data_from_Python["m_actions"]
    m = m_actions # convert(Array{Int}, m_uniform*ones(n))

    p_ws, v_ws, ϖ_ws = undef, undef, undef

    @info "=============================="
    @info "[🐉 ] STARTING STAGE 1"
    @info "=============================="


    p_ws, v_ws, ϖ_ws, term_status_stage_1, sol_time_1 = NE_computation_bnb_solver_complete_continuous_version(
    # data
    # ====
    A, n, m, util_range,
    p_ws, v_ws, ϖ_ws
    ; 
    # options
    # ======
    solver = :knitro, # options are :gurobi and :knitro 
    objective_type = :penalized, # options are :penalized and :exact
    FeasibilityTol_KNITRO = 1e-6,
    OptimalityTol_KNITRO = 1e-6,
    MIPOptimalityTolKNITRO = 1e-6
    )

    @show term_status_stage_1

    ϵ_Nash = compute_ϵ_for_mixed_strategy(A, n, m, p_ws)

    @info "[⏱️ ] ϵ-Nash after stage 1 is $(ϵ_Nash)" #$ 

    @info "=============================="
    @info "[🐉 ] STARTING STAGE 2"
    @info "=============================="

    p_ws, v_ws, ϖ_ws, term_status_stage_2, sol_time_2 = NE_computation_bnb_solver_complete_continuous_with_complementarity(
    # data
    # ====
    A, n, m, util_range,
    # warm-start points
    # ==============
    p_ws, v_ws, ϖ_ws;
    # options
    # ======
    solver = :knitro, # options are :gurobi and :knitro 
    FeasibilityTol_KNITRO = 1e-6,
    OptimalityTol_KNITRO = 1e-4,
    MIPOptimalityTolKNITRO = 1e-4
    )

    ϵ_Nash = compute_ϵ_for_mixed_strategy(A, n, m, p_ws)

    @info "[⏱️ ] ϵ-Nash after stage 2 is $(ϵ_Nash)" #$ 
        
    if term_status_stage_2 != LOCALLY_SOLVED && ϵ_Nash > 1e-4

        # Means tha quality of solution produced by stage 1 and stage 2 is not good enough, so we run the final spatial branch and bound stage
        @info "=============================="
        @info "[🐉 ] STARTING STAGE 3"
        @info "=============================="

        p_ws, v_ws, ϖ_ws, sol_time_3 = NE_computation_bnb_solver_complete_continuous_version(
            # data
            # ====
            A, n, m, util_range,
            p_ws, v_ws, ϖ_ws
            ;
            # options
            # ======
            solver=:gurobi, # options are :gurobi and :knitro 
            objective_type=:penalized, # options are :penalized and :exact
            FeasibilityTol_KNITRO=1e-4,
            OptimalityTol_KNITRO=1e-4,
            MIPOptimalityTolKNITRO=1e-4
        )

        # Polish the best found solution found by Gurobi

        p_ws, v_ws, ϖ_ws, term_status_stage_3, sol_time_4 = NE_computation_bnb_solver_complete_continuous_with_complementarity(
            # data
            # ====
            A, n, m, util_range,
            # warm-start points
            # ==============
            p_ws, v_ws, ϖ_ws;
            # options
            # ======
            solver=:knitro, # options are :gurobi and :knitro 
            FeasibilityTol_KNITRO=1e-6,
            OptimalityTol_KNITRO=1e-4,
            MIPOptimalityTolKNITRO=1e-4
        )

        ϵ_Nash = compute_ϵ_for_mixed_strategy(A, n, m, p_ws)

        @info "[⏱️ ] ϵ-Nash after stage 3 is $(ϵ_Nash)" #$

    else

        # no need to run the final stage, stage 1 or 2 found a solution

        sol_time_3 = 0
        sol_time_4 = 0
        # No need to run the final stage, stage 1 or 2 found a solution

    end



    @info "[⏱️ ] Final ϵ_Nash for the computed approximate Nash equilibrium is $(ϵ_Nash)"#$


    @info "=============================="
    @info "[🐉 ] DONE, STORING SOLUTION"
    @info "=============================="

    total_sol_time = sol_time_1 + sol_time_2 + sol_time_3 + sol_time_4
    #+ sol_time_4 

    return n_players, m_actions, p_ws, v_ws, ϖ_ws, total_sol_time, ϵ_Nash

end

function compute_NE_master_via_complementarity_heuristic(input_file_name_with_path)

    n_players, m_actions, A, util_range = normalize_payoff_matrix(input_file_name_with_path; normalize_type= :zero_one)  # options are :zero_one and :minusOne_One);

    # n = n_players # data_from_Python["n_players"]
    # m_uniform = m_actions# data_from_Python["m_actions"]
    # m = convert(Array{Int}, m_uniform*ones(n))

    n = n_players # data_from_Python["n_players"]
    # m_uniform = m_actions# data_from_Python["m_actions"]
    m = m_actions # convert(Array{Int}, m_uniform*ones(n))

    p_ws, v_ws, ϖ_ws = undef, undef, undef
    
    # @info "=============================="
    # @info "[🐉 ] STARTING STAGE 0"
    # @info "=============================="

    # p_ws, v_ws, z_ws, s_ws, τ_ws, sol_time_0 = NE_computation_find_feasible(
    # # data
    # # ====
    # A, n, m, util_range;
    # # options
    # # ======
    # solver = :knitro, # options are :gurobi and :knitro 
    # FeasibilityTol_KNITRO = 1e-6,
    # OptimalityTol_KNITRO = 1e-6,
    # MIPOptimalityTolKNITRO = 1e-2
    # )

    @info "=============================="
    @info "[🐉 ] STARTING STAGE 1"
    @info "=============================="


    p_ws, v_ws, ϖ_ws,  sol_time_1 = NE_computation_bnb_solver_complete_continuous_version(
    # data
    # ====
    A, n, m, util_range,
    p_ws, v_ws, ϖ_ws
    ; 
    # options
    # ======
    solver = :knitro, # options are :gurobi and :knitro 
    objective_type = :penalized, # options are :penalized and :exact
    FeasibilityTol_KNITRO = 1e-6,
    OptimalityTol_KNITRO = 1e-6,
    MIPOptimalityTolKNITRO = 1e-6
    )

    ϵ_Nash = compute_ϵ_for_mixed_strategy(A, n, m, p_ws)

    @info "[⏱️ ] ϵ_Nash for the Nash equilibrium is $(ϵ_Nash)"#$


    @info "=============================="
    @info "[🐉 ] DONE, STORING SOLUTION"
    @info "=============================="

    total_sol_time = sol_time_1 
    #+ sol_time_4 

    return n_players, m_actions, p_ws, v_ws, ϖ_ws, total_sol_time, ϵ_Nash

end

