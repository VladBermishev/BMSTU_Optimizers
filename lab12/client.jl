using StatsBase
using HTTP.WebSockets
using JSON
using ArgParse
include("genetics_algo.jl")
using .Genetics: create_target_function, generate_initial_population, ws_genetics_min, get_default_gen_strategy
include("structures.jl")
using .CommonStructures: ClientConfig, ClientConfigRequest, GeneticsTaskRequest, GeneticsTaskResponce, PopulationRequest, PopulationResponce, MigratePopulationRequest, BestResultRequest, BestResultResponce
include("image_generation.jl")
using .Images
 


image_vector = Images.read_image("target_image.png")
image_tf = create_target_function(image_vector)
@info "Target image read"

config = nothing
current_iteration = 0
best_result = 1000000.0
population = nothing
gen_strategy = nothing



function set_config(ws, message)
    global config = ClientConfig(message["config"]["isolation_time"], message["config"]["population_size"], message["config"]["max_iter"])
    global population = generate_initial_population(config.population_size, (0x00, 0xff), DIMS=image_tf.DIMS, values_type=UInt8)
    global gen_strategy = get_default_gen_strategy()
    global current_iteration = 0
    global best_result = 1000000.0
    @info "Config set $config"
end

function start_work(ws, message)
    if !isnothing(config)
        if current_iteration < config.max_iter
            @info "Work started"
            result = ws_genetics_min(image_tf, (0x00, 0xff), population, gen_strategy;
             max_iter=config.isolation_time, mut_prob=0.01, values_type=UInt8, verbose_every=10)
            global population = result[1]
            global best_result = image_tf.loss(result[2])
            global current_iteration += result[3]
            @info "Work finished"
        end
        send(ws, json(GeneticsTaskResponce(copy(population))))
    end
end

function migrate(ws, message)
    if !isnothing(config)
        global population[length(population) / 2 + 1:end] = sample(message["population"], length(population) >> 1, replace=false)
        send(ws, json(MigratePopulationResponse()))
    end
end

function send_best_result(ws, message)
    send(ws, json(BestResultResponce(best_result)))
end

endpoints = Dict([
    ("/set-config", set_config),
    ("/work-start", start_work),
    ("/migrate-population", migrate),
    ("/get-best-result", send_best_result)
])

function process_message(ws, msg)
    try
        message = JSON.parse(msg) 
        endpoints[message["topic"]](ws, message)
    catch x
        @error x
    end
end

params = ArgParseSettings()
@add_arg_table! params begin
    "--addr", "-a"
    help = "WebSocket server address"
    arg_type = String
    default = "127.0.0.1"
    "--port", "-p"
    help = "WebSocket server port"
    arg_type = Int
    default = 8124
end

args = parse_args(ARGS, params)

WebSockets.open("ws://$(args["addr"]):$(args["port"])") do ws
    @info "Connected to ws://$(args["addr"]):$(args["port"])"
    for msg in ws
        @debug "$(ws.id): $msg"
        process_message(ws, msg)
    end
end

