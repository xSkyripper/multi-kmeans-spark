if __name__ == '__main__':
    with open("iris_constraints.txt", mode="rt") as input_file_object:
        with open("iris_constraints.txt", mode="wt") as output_file_object:
            for line in input_file_object:
                values = line.split("\t")
                parsed_values = " ".join(values)
                output_file_object.write(parsed_values)
