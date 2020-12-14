folder = "C:\Users\molenaar\OneDrive - Stichting Deltares\Documents\Thesis - Deltares\Data"
name = "IMG_SW00_OPT_MS4_1C_20180612T092821_20180612T092823_TOU_1234_a2dc_R1C1"

file = folder + "/" + name + ".nc"
dataset = Dataset(file, "r")
var_values = {}
var_units = {}
var_names = {}
lst = list(dataset.variables.keys())
print(lst)