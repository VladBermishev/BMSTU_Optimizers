module Images
    using ImageMagick
    unpack_val(x) = [x.r.i, x.g.i, x.b.i]

    function generate_target_img(size=3)
        result = zeros(UInt8, size^2) .+ 0x6
        for idx in 1:size
            result[trunc(UInt8, size/2)+1 + (idx-1)*size] = 3
            result[(trunc(UInt8, size/2))*size + idx] = 3
            result[(idx-1)*size+idx] = 0
            result[size-idx+1 + (idx-1)*size] = 0
        end
        return result
    end

    function generate_colored_img(size=3)
        result = zeros(UInt8, (size^2,3)) .+ 0x7
        for idx in 1:size
            result[trunc(UInt8, size/2)+1 + (idx-1)*size,:] = UInt8.(rand(0:7, 3))
            result[(trunc(UInt8, size/2))*size + idx,:] = UInt8.(rand(0:7, 3))
            result[(idx-1)*size+idx,:] = [0x0, 0x0, 0x0]
            result[size-idx+1 + (idx-1)*size,:] = [0x0, 0x0, 0x0]
        end
        return reshape(result, 3*size^2)
    end

    function read_image(filename="target_image.png")
        target_img = open(filename) do io
            ImageMagick.load(io)
        end
        return reduce(vcat, reshape(unpack_val.(target_img), prod(size(target_img))))
    end
end