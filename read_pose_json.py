import cv2
import numpy as np
import json
import glob
import os
import csv



class BoundingBox(object):
    """
    A 2D bounding box
    """

    def __init__(self, points):
        if len(points) == 0:
            raise ValueError("Can't compute bounding box of empty list")
        self.minx, self.miny = float("inf"), float("inf")
        self.maxx, self.maxy = float("-inf"), float("-inf")
        self.extension = 30  # Extend the size of the bounding box
        for joint in points:
            # Set min coords

            if joint[0] < self.minx and joint[0] > 0:
                self.minx = joint[0]
            if joint[1] < self.miny and joint[1] > 0:
                self.miny = joint[1]
            # Set max coords
            if joint[0] > self.maxx:
                self.maxx = joint[0]
            elif joint[1] > self.maxy:
                self.maxy = joint[1]

        self.minx = self.minx - self.extension
        self.miny = self.miny - self.extension
        self.maxx = self.maxx + self.extension
        self.maxy = self.maxy + self.extension

    @property
    def width(self):
        return self.maxx - self.minx

    @property
    def height(self):
        return self.maxy - self.miny

    def __repr__(self):
        return "BoundingBox({}, {}, {}, {})".format(
            self.minx, self.maxx, self.miny, self.maxy)

    def getCoordinates(self):
        return self.minx, self.miny, self.width, self.height


def draw_bounding_box(input_path, bounding_box_dimension, output_path):
    # Read the image from the path

    img = cv2.imread(input_path)

    # Draw the rectange based on the minx and miny coordinate for the first point then minx + width, miny + height for second point
    #
    cv2.rectangle(img, (int(bounding_box_dimension[0]), int(bounding_box_dimension[1])), (
    int(bounding_box_dimension[0]) + int(bounding_box_dimension[2]),
    int(bounding_box_dimension[1]) + int(bounding_box_dimension[3])), (255, 0, 0), 2)
    cv2.imwrite(output_path, img)
    return


def calculate_largest_bb(list_bb):
    max_width = 0
    max_height = 0
    largest_bb_index = 0
    for i in range(len(list_bb)):
        width = list_bb[i][2]
        height = list_bb[i][3]

        if width > max_width and height > max_height:
            max_width = width
            max_height = height
            largest_bb_index = i

    return largest_bb_index

"""
read_json
Description : Read the json files from a FOLDER - eg: Drive_1_json folder
Input : Path of folder
Outputs : folder_people, folder_bb, list_people 
        folder_people consists of all the person detected in each image - list of list
        folder_bb similar to folder_people but is used for the bounding boxes
        list_people consists of a list of the largest people in each image - index 0 = image 0 largest person
"""
#Read the json files from a particular folder
def read_json(path):
    #json_files = [json_file for json_file in os.listdir(path) if json_file.endswith('.json')]
    json_files = []
    subfolders = [folder for folder in os.listdir(path)]

    #Retrieve all the json files from one subfolder
    for subfolder in subfolders[1:]:
        subdir_json_files = []
        subdir_json_files=[subfolder + '/' + json_file for json_file in os.listdir(os.path.join(path,subfolder)) if json_file.endswith('.json')]
        json_files += subdir_json_files

    folder_people = []
    folder_bb = []
    list_largest_people = [] #This list only contains the largest person from each image

    for index, json_file_name in enumerate(json_files):
        with open(os.path.join(path,json_file_name)) as json_file:
            data = json.load(json_file)
            #JSON Data loaded into data for each file
            #Read the data into variables
            list_people = []
            list_bb = []
            for i in range(len(data['people'])):


                body_part = data['people'][i]['body_parts']

                nose, neck, r_shoulder, r_elbow, r_wrist, l_shoulder, l_elbow, l_wrist, r_hip, r_knee, r_ankle, l_hip, l_knee, l_ankle, r_eye, l_eye, r_ear, l_ear = ([] for i in range(18))
                people = nose, neck, r_shoulder, r_elbow, r_wrist, l_shoulder, l_elbow, l_wrist, r_hip, r_knee, r_ankle, l_hip, l_knee, l_ankle, r_eye, l_eye, r_ear, l_ear
                people = list(people)
                i = 0

                for joint in people:
                    joint.append(body_part[i])
                    i += 1
                    joint.append(body_part[i])
                    i += 2

                # list of people, add a complete person - complete meaning with their joints
                list_people.append(people)

                # Get the bounding box of the person
                coords = BoundingBox(people)

                bounding_box_dimension = coords.getCoordinates()
                list_bb.append(bounding_box_dimension)

            largest_bb_index = calculate_largest_bb(list_bb)
            largest_people_index = largest_bb_index
            largest_people = list_people[largest_people_index]

            list_largest_people.append(largest_people)
            folder_people.append(list_people)
            folder_bb.append(list_bb)

    return folder_people,folder_bb, list_largest_people

"""
write_to_csv
Description : Writes the coordinates of the joint of a person into a CSV
Input : Directory of the CSV path, List of person
Output : CSV
"""


def write_to_csv(csv_dir, list_person):

    csv_file = open(csv_dir,"w")
    row = ''
    for person in(list_person):
        row = ''
        for joints in person:
            row += str(joints[0]) + "," + str(joints[1]) + ","
        row += "\n"
        csv_file.write(row)
    return

"""
adjust_coordinates 
Description : Converts the coordinates of the joints of a person to be relative to the NOSE
Input : List_Person : List of people where each person is containing a list of joints
Output : Person : List containing list of joints that have been adjusted
"""


def adjust_coordinates(list_person):


    for person in list_person:

        noseX = person[0][0]  # Nose, x coordinate
        noseY = person[0][1]  # Nose, y coordinate

        for joints in person:

            joints[0] = joints[0] - noseX
            joints[1] = joints[1] - noseY

    return list_person

"""
Read from csv
Description : Retrieves data from CSV of the same format based on the output. The CSV should contain normalize coordinates of only the athletes in each frame
Input : Directory of the CSV file
Outputs : Returns the the list of people with their normalize joints coordinate
"""

def read_from_csv(csv_dir):

    with open(csv_dir,"rw") as csv_data:
        csv_data = csv.reader(csv_data, delimiter=',', quotechar="'")
        folder_people = [] #Only until 35 as it is the last item in the excel sheet

        for row in csv_data:
            folder_people.append(row)

    return folder_people

"""
Main code section
"""
"""
list_people = []

list_bb = []
#A folder consists of list of people from each image
folder_people = []
folder_bb = []
list_largest_people = [] #List of largest person for each image - each person is from 1 image

# Read json of the poses
json_data_files = glob.glob('output/*.json')
json_data_files.sort()


json_file_dir = '/Users/thammingkeat/PycharmProjects/Drive/Json/Drive_1/Drive_1_000000000001_pose.json'

with open(json_file_dir) as data_file:
    data = json.load(data_file)


# 0 is the first index for the body part in people
# returns a list
# Read image/frame of video
path = '/Users/thammingkeat/PycharmProjects/Drive_1_images/Drive_1_000000000001_rendered.png'
img = cv2.imread(path)

img_dir = '/Users/thammingkeat/PycharmProjects/Drive/Images/'
json_dir = '/Users/thammingkeat/PycharmProjects/Drive/Json/'
csv_dir = '/Users/thammingkeat/PycharmProjects/Drive_1_athlete.csv'

#For reading all the json files in a folder
#Returns a folder of people - consisting of list of people from each image
folder_people,folder_bb, list_largest_people = read_json(json_dir)
"""
# for i in range(len(data['people'])):
#
#     body_part = data['people'][i]['body_parts']
#
#     nose, neck, r_shoulder, r_elbow, r_wrist, l_shoulder, l_elbow, l_wrist, r_hip, r_knee, r_ankle, l_hip, l_knee, l_ankle, r_eye, l_eye, r_ear, l_ear = ([] for i in range(18))
#     people = nose, neck, r_shoulder, r_elbow, r_wrist, l_shoulder, l_elbow, l_wrist, r_hip, r_knee, r_ankle, l_hip, l_knee, l_ankle, r_eye, l_eye, r_ear, l_ear
#     people = list(people)
#     i = 0
#
#     for joint in people:
#         #For each joint in the people, add their coordinates from the data read from the json
#         joint.append(body_part[i])
#         i += 1
#         joint.append(body_part[i])
#         i += 2
#
#
#     # nose.append(body_part[0])
#     # nose.append(body_part[1])  # ,body_part[2]
#     # neck.append(body_part[3])
#     # neck.append(body_part[4])  # ,body_part[5]
#     # r_shoulder.append(body_part[6])
#     # r_shoulder.append(body_part[7])  # ,body_part[8]
#     # r_elbow = body_part[9], body_part[10]  # ,body_part[11]
#     # r_wrist = body_part[12], body_part[13]  # ,body_part[14]
#     # l_shoulder = body_part[15], body_part[16]  # ,body_part[17]
#     # l_elbow = body_part[18], body_part[19]  # ,body_part[20]
#     # l_wrist = body_part[21], body_part[22]  # ,body_part[23]
#     # r_hip = body_part[24], body_part[25]  # ,body_part[26]
#     # r_knee = body_part[27], body_part[28]  # ,body_part[29]
#     # r_ankle = body_part[30], body_part[31]  # ,body_part[32]
#     # l_hip = body_part[33], body_part[34]  # ,body_part[35]
#     # l_knee = body_part[36], body_part[37]  # ,body_part[38]
#     # l_ankle = body_part[39], body_part[40]  # ,body_part[41]
#     # r_eye = body_part[42], body_part[43]  # ,body_part[44]
#     # l_eye = body_part[45], body_part[46]  # ,body_part[47]
#     # r_ear = body_part[48], body_part[49]  # ,body_part[50]
#     # l_ear = body_part[51], body_part[52]  # ,body_part[53]
#     #
#     # people = nose, neck, r_shoulder, r_elbow, r_wrist, l_shoulder, l_elbow, l_wrist, r_hip, r_knee, r_ankle, l_hip, l_knee, l_ankle, r_eye, l_eye, r_ear, l_ear
#
#     #list of people, add a complete person - complete meaning with their joints
#     list_people.append(people)
#
#     #Get the bounding box of the person
#     coords = BoundingBox(people)
#
#     bounding_box_dimension = coords.getCoordinates()
#     list_bb.append(bounding_box_dimension)
#
#     # Draw the rectange based on the minx and miny coordinate for the first point then minx + width, miny + height for second point
#     # Draw for all the human poses
#     # cv2.rectangle(img,(int(bounding_box_dimension[0]),int(bounding_box_dimension[1])),(int(bounding_box_dimension[0]) + int(bounding_box_dimension[2]), int(bounding_box_dimension[1]) + int(bounding_box_dimension[3])), (255,0,0), 2)
#     # output = '/Users/thammingkeat/PycharmProjects/output/bb.png'
#     # cv2.imwrite(output,img)
#
# # Calculate the largest bounding box and draw ONLY the largest
# largest_bbox_index = calculate_largest_bb(list_bb)
# largest_people_index = largest_bbox_index
# largest_people = list_people[largest_people_index]
# # Get the largest person index from the largest bounding box index
# # Will be the identical list as the number of bounding boxes = number of people
#
# bounding_box_dimension = list_bb[largest_bbox_index]
# cv2.rectangle(img, (int(bounding_box_dimension[0]), int(bounding_box_dimension[1])), (
# int(bounding_box_dimension[0]) + int(bounding_box_dimension[2]),
# int(bounding_box_dimension[1]) + int(bounding_box_dimension[3])), (255, 0, 0), 2)
# output = '/Users/thammingkeat/PycharmProjects/output/bb.png'
# cv2.imwrite(output, img)

"""

    Use the nose as the central point
    Each joint would have to minus the nose's coordinate to become relative
    
"""
"""
list_largest_people = adjust_coordinates(list_largest_people)

write_to_csv(csv_dir, list_largest_people)
csv_data = read_from_csv(csv_dir)
"""

