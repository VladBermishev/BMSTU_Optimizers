module CommonStructures
    struct ClientConfig
        isolation_time
        population_size
        max_iter
        ClientConfig(isolation_time=200, population_size=100, max_iter=1000) = new(isolation_time, population_size, max_iter)
    end

    struct ClientConfigRequest
        topic
        config
        ClientConfigRequest(config) = new("/set-config", config)
    end

    struct GeneticsTaskRequest
        topic
        GeneticsTaskRequest() = new("/work-start")
    end

    struct GeneticsTaskResponce
        topic
        population
        GeneticsTaskResponce(population) = new("/work-finished", population)
    end

    struct PopulationRequest
        topic
        PopulationRequest() = new("/get-population")
    end

    struct PopulationResponce
        topic
        population
        PopulationResponce(population) = new("/population", population)
    end

    struct MigratePopulationRequest
        topic
        population
        MigratePopulationRequest(population) = new("/migrate-population", population)
    end

    struct MigratePopulationResponse
        topic
        MigratePopulationResponse() = new("/migration-ended")
    end

    struct BestResultRequest
        topic
        BestResultRequest() = new("/get-best-result")
    end

    struct BestResultResponce
        topic
        result
        BestResultResponce(result) = new("/best-result", result)
    end
end