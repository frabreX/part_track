import csv


def get_data_at_position(filepath, row_index, col_index):
    with open(filepath, "r") as output:
        csv_reader = csv.reader(output)
        next(csv_reader)  # skip header
        rows = list(csv_reader)

        if -len(rows) <= row_index < len(rows):
            row = rows[row_index]
            if -len(row) <= col_index < len(row):
                return row[col_index]
            else:
                print(f"Column index {col_index} is out of bounds for row {row_index}.")
                return None
        else:
            print(f"Row index {row_index} is out of bounds.")


#data that must be known to get MSD in SI um
video_height = 2000
video_widght = 4000

#the um of this data (e.g. micron) determines the output
#micron is recomended

sensor_height = 2
sensor_widght = 4

magnification = 400

# Some usefull variables

with open("fps.txt", 'r') as fps:
    fps = int(round(float(fps.read())))

number_particles = int(get_data_at_position("output.csv", -1, 1))

with open("video_lenght.txt", 'r') as video_lenght:
    video_lenght = video_lenght.read()

number_frames_per_particle = int(get_data_at_position("output.csv", -1, 0)) -  fps + 1 #minus to subtract the 1 second skip and plus one to count the initial one
print(number_frames_per_particle, fps, video_lenght, number_particles)



#converting csv in a list of lists
data = []

with (open("output.csv", "r") as output):
    csv_reader = csv.reader(output)
    next(csv_reader)
    for line in csv_reader:
        frame, item_id, x, y, w, h = map(int, line)

        data.append([item_id, frame, x, y,])


sorted_data = sorted(data, key=lambda a: (a[0], a[1]))

print(sorted_data)

#computing MSD
x = 0
y = 0
x0 = 0
y0 = 0

MSD_particles_list = []
MSD_total_sum = 0

for particle in range(1, number_particles +1):
    a = (particle  - 1) * number_frames_per_particle
    b = particle * number_frames_per_particle

    MSD_sum = 0

    for position in range(a, b):
        x = sorted_data[position][2]
        y = sorted_data[position][3]

        if position == a:
            y0 = y
            x0 = x

        MSD_t = (x - x0) ** 2 + (y - y0) ** 2

        x0 = x
        y0 = y

        MSD_sum += MSD_t

    MSD_particles = MSD_sum/(number_frames_per_particle - 1) #number_frames_per_particle - 1 displacements (because displacements are between consecutive frames)
    MSD_total_sum += MSD_particles
    MSD_particles_list.append((particle, MSD_particles))

print(f"The MSD corresponding to the the particles id are: {MSD_particles_list}")

MSD = MSD_total_sum /number_particles

print(MSD)

# Convert MSD from pixels ** 2/frame to m ** 2/s (or micron meters)

pizel_height = sensor_height / (video_height * magnification)
pizel_widght = sensor_widght / (video_widght * magnification)

# the pizel size must be a sqare otherwise the video proportion are distorted
if pizel_height == pizel_widght:

    pixel_size = pizel_height
    MSD_SI = MSD_particles * (pixel_size ** 2) * fps

    print(f"MSD_SI: {MSD_SI}")
