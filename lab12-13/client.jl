using HTTP.WebSockets
using JSON
using ArgParse
include("genetics_algo.jl")
using .genetics


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
    for msg in ws
        
    end
end

