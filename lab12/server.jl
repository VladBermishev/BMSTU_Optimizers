using HTTP.WebSockets
using JSON
using ArgParse
include("structures.jl")
using .CommonStructures: ClientConfig, ClientConfigRequest, GeneticsTaskRequest, GeneticsTaskResponce,
 PopulationRequest, PopulationResponce, MigratePopulationRequest, BestResultRequest, BestResultResponce

const SERVER_ADDR = "127.0.0.1"
@enum ClientState Free Working Migrating

mutable struct Client
    socket
    config::ClientConfig
    state::ClientState
    population
    best::Float32
    Client(socket, config::ClientConfig, state::ClientState) = new(socket, config, state, [], 0.0)
end

active_clients = Vector{Client}()
cancellation_token = Threads.Atomic{Bool}(false)
start_migration = Threads.Atomic{Bool}(false)

function register(ws; config::ClientConfig = ClientConfig())
    @info "new client connected: $(ws.id)"
    new_client = Client(ws, config, Free)
    push!(active_clients, new_client)
    send(ws, json(ClientConfigRequest(new_client.config)))
end

function unregister(ws)
    for client_idx in eachindex(active_clients)
        if active_clients[client_idx].socket.id == ws.id
            deleteat!(active_clients, client_idx)
            return
        end
    end
    @warn "unregister on unknown socket: $(ws.id)"
end

function get_best_result(ws)
    for client in active_clients
        if client.socket.id == ws.id && client.state == Free
            send(ws, json(BestResultRequest()))
        end
    end
end

function start_work(ws)
    for client in active_clients
        if client.socket.id == ws.id && client.state == Free
            send(client.socket, json(GeneticsTaskRequest()))
            client.state = Working
            return
        end
    end
    @warn "Cant start work for client: $(ws.id)"
end

function migrate(from_client, to_client)
    while to_client.state != Free
        sleep(0.05)
    end
    send(to_client.socket, MigratePopulationRequest(from_client.population))
    to_client.state = Migrating
end

function save_result(ws, message)
    for client in active_clients
        if client.socket.id == ws.id
            client.best = message["result"]
            @info "Client{$(ws.id)} -> best result: $(client.best)"
        end
    end
end

function work_finished(ws, message)
    can_migrate = false
    for client in active_clients
        if client.socket.id == ws.id && client.state == Working
            client.population = copy(message["population"])
            client.state = Free
            @info "client: $(client.socket.id) finished work"
        end
        can_migrate &= client.state == Free
    end
    start_migration[] = can_migrate
end

function migration_ended(ws, message)
    for client in active_clients
        if client.socket.id == ws.id && client.state == Migrating
            client.state = Free
        end
    end
end

endpoints = Dict([
    ("/work-finished", work_finished),
    ("/best-result", save_result),
    ("/migration-ended", migration_ended)
])

function process_message(ws, msg)
    try
        message = JSON.parse(msg) 
        endpoints[message["topic"]](ws, message)
    catch x
        @error x
    end
end

function cmd_processor_task()
    while true
        print(":>")
        cmd = readline()
        if cmd == "q" || cmd == "quit"
            cancellation_token[] = true
            break
        elseif cmd == "s" || cmd == "step"
            for client in active_clients
                start_work(client.socket)
                @info "client: $(client.socket.id) started work"
            end
        elseif cmd == "i" || cmd == "info"
            for client in active_clients
                get_best_result(client.socket)
            end
        else
            println("commands:\n\tq | quit - stop the server\n\ti | info - get best results from clients\n\ts | step - migrate and start another round")
        end
    end
end

params = ArgParseSettings()
@add_arg_table! params begin
   "--port", "-p"
   help = "WebSocket server port"
   arg_type = Int
   default = 8124
   "--isolation-time", "-t"
   help = "isolation time for each client"
   arg_type = Int
   default = 200
   "--population-size"
   help = "population size for each client"
   arg_type = Int
   default = 100
   "--max-iter"
   help = "population size for each client"
   arg_type = Int
   default = 1000
end    

args = parse_args(ARGS, params)

ws_server = WebSockets.listen!(SERVER_ADDR, args["port"]) do ws
    register(ws, config=ClientConfig(args["isolation-time"], args["population-size"], args["max-iter"]))
    for msg in ws
        process_message(ws, msg)
    end
    unregister(ws)
end

Threads.@spawn cmd_processor_task()

while !cancellation_token[]
    if start_migration[]
        @info "Migration started"
        if length(active_clients) > 1
            prev_idx = length(active_clients)
            for idx in eachindex(active_clients)
                migrate(active_clients[idx], active_clients[prev_idx])
                migrate(active_clients[prev_idx], active_clients[idx])
                next_idx = idx == length(active_clients) ? 1 : idx
                migrate(active_clients[idx], active_clients[next_idx])
                migrate(active_clients[next_idx], active_clients[idx])
                prev_idx = idx
                @info "Client $idx processed"
            end
        end
        @info "Migration ended"
        start_migration[] = false
    end
    sleep(0.05)
end

close(ws_server)

