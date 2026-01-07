# This function writes data to a text file in a structured format so that it can be plotted, e.g., using TikZ/pgfplots.
# The first argument is the filename, followed by name-value pairs.
# Each name is a string, and each value is a vector of data.
function data2txt(filename::AbstractString, name::AbstractString, value, pairs...)
    # Check if the number of name value pairs match.
    if isodd(length(pairs)) || !all(isa.(pairs[1:2:end], AbstractString))
        throw(ArgumentError("Arguments must be name-value pairs"))
    end

    # Determine the maximum length of the input vectors.
    max_length = length(value)
    for i in 2:2:length(pairs)
        if length(pairs[i]) > max_length
            max_length = length(pairs[i])
        end
    end

    # Write to file.
    open(filename, "w") do file
        # Write the header.
        write(file, string(name))
        for j in 1:2:length(pairs)
            write(file, " " * string(pairs[j]))
        end
        write(file, "\n")

        # Write data row by row.
        for i in 1:max_length
            if i <= length(value)
                write(file, string(value[i]))
            else
                write(file, " ")
            end
            for j in 2:2:length(pairs)
                if i <= length(pairs[j])
                    write(file, " " * string(pairs[j][i]))
                else
                    write(file, "  ")
                end
            end
            write(file, "\n")
        end
    end
end