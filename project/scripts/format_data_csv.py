'''
Format data file
'''
input_path = '/home/burntice/0_repositories/CZ4042/project/data/aligned_gender.txt'
output_path = input_path[:-4] + '.csv'

with open(input_path) as input_file, open(output_path, 'w') as output_file:
    line = input_file.readline()
    output_file.write(line)

    for line in input_file:
        line = line[5:]
        output_file.write(line)

print('done')
