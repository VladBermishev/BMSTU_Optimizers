using HTTP.WebSockets
using JSON
using ArgParse


const SERVER_ADDR = "127.0.0.1"
@enum ClientState Free Working Migrating

mutable struct Client
    socket
    state::ClientState
end

active_clients = Vector{Client}()
client_config = Dict([("isolation_time", 200), ("population_size", 100), ("max_iter", 1000)])

function register(ws)
    push!(active_clients, Client(ws, Free))
    send(ws, JSON.json(client_config))
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

function work_start(ws)
    for client in active_clients
        if client.socket.id == ws.id
            client.state = Working
        end
    end
end

function done(ws)
    for client in active_clients
        if client.socket.id == ws.id
            client.state = Free
        end
    end
end

endpoints = Dict([
    ("/start", work_start),
    ("/done", done),
])

function process_message(ws, msg)
    try
        message = JSON.parse(msg) 
        endpoints[message["topic"]](ws)
    catch x
        @error x
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
client_config = Dict([("isolation-time", args["isolation-time"]), ("population-size", args["population-size"]), ("max-iter", args["max-iter"])])

ws_server = WebSockets.listen!(SERVER_ADDR, args["port"]) do ws
    register(ws)
    for msg in ws
        @debug "$(ws.id): $msg"
        process_message(ws, msg)
    end
    unregister(ws)
end

while true
    cmd = readline()
    if cmd == "q" || cmd == "quit"
        break
    end
    if cmd == "i" || cmd == "info"
        break
    end
end
close(ws_server)

