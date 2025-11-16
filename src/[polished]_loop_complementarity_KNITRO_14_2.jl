IMPLEMENTATION_TYPE = :full # options are :full and :heuristic, when it is full, time limit for KNITRO is 600 s (10 min), when it is heuristic, time limit for KNITRO is 15 s (0.25 min)

include("[polished]_master_file_KNITRO_14_2.jl")

input_dir = "RandomGraphicalGame/RandomGraphicalGame/"

# create a list of all files in the input directory 

file_list = readdir(input_dir)

# create an empty dictionary to store the results
results = Dict() 

for i in 1:length(file_list)

    # Load file name
    file_name_input = file_list[i]
    println("Processing file: ", file_name_input)
    
    # concatenate the input directory and file name to get the full path
    input_file_name_with_path = joinpath(input_dir, file_name_input)

    # run the master file
    n_players, m_actions, p_ws, v_ws, ϖ_ws,total_sol_time, ϵ_Nash = compute_NE_master_via_complementarity(input_file_name_with_path)

    # log10_max_constraint_violation = log10(max(max_constraint_violation,1e-16)) # to avoid log(0) error

    # Save the results to the dictionary results, 
    # whre the key is the file name and the value is a tuple of the results
    results[file_name_input] = (n_players, m_actions, p_ws, v_ws, ϖ_ws,total_sol_time, ϵ_Nash)

end

# save the dicationary in JLD2 format 

using JLD2, FileIO, Dates

filename_part_1 = "results"

filename_part_2 = string(today())

filename_part_3 = ".jld2"

results_file_name = filename_part_1 * "_" * filename_part_2 * filename_part_3

save(results_file_name, results)

## Pretty tables 
# -----------------

result = load(results_file_name)

array_max_constraint_violations = []

n_players_array = []

m_actions_array = []

sol_time_array = []

file_name_array = []

ϵ_Nash_array = []

input_dir = "RandomGraphicalGame/RandomGraphicalGame/"

file_list = readdir(input_dir)

for i in 1:length(keys(results))

    # Load file name
    file_name_input = file_list[i]

    push!(file_name_array, file_name_input)
    
    specific_results = results[file_name_input]

    n_players, m_actions, p_ws, v_ws, ϖ_ws,total_sol_time, ϵ_Nash = specific_results
    
    if abs(ϖ_ws) <= 1e-16
        ϖ_ws = 1e-16 # Treat very small values as zero
    end 

    log_10_max_constraint_violation = log10(ϖ_ws) # to avoid log(0) error

    # Append the max_constraint_violation to the array

    push!(array_max_constraint_violations, log_10_max_constraint_violation)

    push!(n_players_array, n_players)

    push!(m_actions_array, m_actions)

    push!(sol_time_array, total_sol_time)

    push!(ϵ_Nash_array, ϵ_Nash)

end


using PrettyTables

data = [file_name_array ϵ_Nash_array sol_time_array]

header = (["File name", "ϵ Nash", "Sol Time (s)" ])

my_table = pretty_table(data; backend = Val(:markdown),  header = header)

